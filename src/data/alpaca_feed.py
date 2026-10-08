"""
Live market data from Alpaca.

This fills in the "swap the data feed" extension point described in the
README with a real one: `AlpacaLiveFeed` implements the same
`MarketDataFeed` interface as `SyntheticFeed`/`YFinanceFeed`, so nothing in
`Orchestrator` needs to change to use it.

Two Alpaca API surfaces are involved, and they work differently:

- `StockHistoricalDataClient` (REST) backs `get_history()` -- a normal
  request/response call used once, to warm up each ticker's rolling window
  before trading starts.
- `StockDataStream` (websocket, asyncio-based) backs `stream()` -- but
  `MarketDataFeed.stream()` is specified as a plain synchronous iterator,
  and the rest of the codebase (Orchestrator.run()) just does
  `for bar in feed.stream(tickers): ...` with no asyncio in sight. To
  bridge that gap, the websocket client runs on a background thread with
  its own event loop; its async bar handler pushes each bar onto a
  thread-safe queue, and stream() is a plain generator that blocks on that
  queue. This is a standard asyncio-in-a-thread pattern, not anything
  Alpaca-specific -- the same approach works for any websocket vendor.

Requires the `alpaca-py` package (`pip install alpaca-py`) and a paper (or
live) API key/secret from https://app.alpaca.markets. Only `DataFeed.IEX`
is used here, which is what a free Alpaca account gets: real-time (not
delayed) trades and quotes from the IEX exchange specifically, not the
full consolidated SIP tape every exchange trades on. That means prices can
occasionally lag or differ slightly from what you'd see on, say, a
broker's own SPY chart, since IEX is one exchange among many -- fine for
testing a strategy's plumbing, worth knowing about before reading too much
into small price discrepancies.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .feed import Bar, MarketDataFeed

# alpaca-py logs its own connect/auth/subscribe/disconnect lifecycle (see
# alpaca.data.live.websocket) at INFO level via the stdlib `logging` module --
# but INFO is below the default WARNING threshold, so with no handler
# configured those lines are silently dropped and only failures (logged at
# WARNING/ERROR via log.warning()/log.exception()) ever reach stdout. That
# made a real incident ("tickers missing from the Performance Summary")
# needlessly hard to diagnose: the log was full of "connection limit
# exceeded" tracebacks but had zero evidence of which (if any) connection
# attempts actually succeeded, for how long, or when. Raising this logger's
# level and giving it a handler surfaces those previously-invisible
# "starting ... websocket connection" / "connected to ..." lines too, so a
# future read of run_live_alpaca_stdout.log can show the full connect /
# subscribe / drop cycle, not just the failures.
_alpaca_logger = logging.getLogger("alpaca")
if not _alpaca_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    _alpaca_logger.addHandler(_handler)
_alpaca_logger.setLevel(logging.INFO)


class AlpacaLiveFeed(MarketDataFeed):
    def __init__(self, tickers: Sequence[str], api_key: str, secret_key: str,
                 feed: str = "iex", history_days: int = 10,
                 reconnect_backoff_s: float = 5.0):
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical.stock import StockHistoricalDataClient

        self.tickers = list(tickers)
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_feed = DataFeed.IEX if feed.lower() == "iex" else DataFeed.SIP
        self.history_days = history_days
        self.reconnect_backoff_s = reconnect_backoff_s

        self._hist_client = StockHistoricalDataClient(api_key, secret_key)
        self._bar_queue: "queue.Queue[Bar]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cursor = {t: 0 for t in self.tickers}  # bars already handed out via get_history

    @staticmethod
    def _todays_session_open_utc(now_utc: datetime) -> datetime:
        """09:30 America/New_York, on `now_utc`'s calendar date in that
        timezone, as a UTC-aware datetime. Doesn't special-case weekends/
        holidays -- on a non-trading day this just comes out earlier than
        any bar Alpaca has for "today", which get_history() below treats
        the same as "market hasn't opened yet" and falls back to the
        multi-day lookback instead."""
        eastern = ZoneInfo("America/New_York")
        now_et = now_utc.astimezone(eastern)
        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        return open_et.astimezone(timezone.utc)

    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now(timezone.utc)
        session_open = self._todays_session_open_utc(end)
        if session_open < end:
            # Anchor the warm-up backfill to today's market open instead
            # of a rolling `history_days`-day window. The multi-day window
            # made bar counts inconsistent across tickers warming up at
            # the exact same moment (one real run showed 118 bars for one
            # symbol and 65 for another): with `limit=lookback` capping
            # the request, a few thinly-IEX-traded days of history could
            # eat into the limit for a less liquid ticker before today's
            # bars were even reached, while a liquid ticker filled the
            # limit from today's bars alone. Anchoring everyone to today's
            # open means every ticker gets the same window -- whatever
            # bars actually printed in it -- so a lower count now reflects
            # that ticker's real liquidity today, not an artifact of how
            # far back the request happened to look.
            start = session_open
        else:
            # Before today's open (premarket) there's no "today" window
            # yet -- fall back to the old multi-day lookback so warm-up
            # still has something to work with.
            start = end - timedelta(days=self.history_days)
        req = StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
            start=start, end=end, feed=self.data_feed, limit=lookback,
        )
        barset = self._hist_client.get_stock_bars(req)
        bars = barset[ticker] if ticker in barset else []
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = [{"open": float(b.open), "high": float(b.high), "low": float(b.low),
                 "close": float(b.close), "volume": float(b.volume)} for b in bars]
        idx = pd.DatetimeIndex([pd.Timestamp(b.timestamp) for b in bars])
        df = pd.DataFrame(rows, index=idx).iloc[-lookback:]
        self._cursor[ticker] = len(df)
        return df

    def _run_stream_forever(self) -> None:
        from alpaca.data.live.stock import StockDataStream

        print(f"[AlpacaLiveFeed] subscribing to {len(self.tickers)} tickers: {self.tickers}")

        while True:
            try:
                stream = StockDataStream(self.api_key, self.secret_key, feed=self.data_feed)

                async def on_bar(bar, _q=self._bar_queue):
                    _q.put(Bar(
                        ticker=bar.symbol, timestamp=pd.Timestamp(bar.timestamp),
                        open=float(bar.open), high=float(bar.high),
                        low=float(bar.low), close=float(bar.close), volume=float(bar.volume),
                    ))

                stream.subscribe_bars(on_bar, *self.tickers)
                stream.run()  # blocks this thread; internally manages its own event loop
                # NOTE: alpaca-py's StockDataStream.run() retries dropped
                # connections FOREVER inside its own internal event loop
                # (with its own exponential backoff) and only ever returns
                # here on a clean stop or on a small set of fatal errors
                # (e.g. "insufficient subscription") -- a transient error
                # like "connection limit exceeded" is caught and retried
                # *inside* run() and never reaches this `except` below. So
                # this handler is NOT the thing recovering from reconnects
                # in normal operation; see the logging config above for
                # how to actually observe those reconnect cycles.
            except Exception as exc:  # noqa: BLE001 -- keep the feed alive across transient drops
                print(f"[AlpacaLiveFeed] stream error ({exc!r}); reconnecting in "
                      f"{self.reconnect_backoff_s:.0f}s")
                time.sleep(self.reconnect_backoff_s)

    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_stream_forever, daemon=True)
            self._thread.start()
        seen: set[str] = set()
        while True:
            bar = self._bar_queue.get()  # blocks until the next real-time bar arrives
            if bar.ticker not in seen:
                seen.add(bar.ticker)
                # The one thing the earlier silent failure mode made
                # impossible to tell at a glance: which subscribed tickers
                # are actually receiving live bars at all, vs. which never
                # get past the websocket's auth/connect/subscribe step.
                print(f"[AlpacaLiveFeed] first live bar received for {bar.ticker} "
                      f"({len(seen)}/{len(self.tickers)} tickers streaming so far)")
            yield bar
