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

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .feed import Bar, HistoricalReplayFeed, MarketDataFeed

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


def bar_to_json(bar: Bar) -> str:
    return json.dumps({"ticker": bar.ticker, "timestamp": bar.timestamp.isoformat(),
                       "open": bar.open, "high": bar.high, "low": bar.low,
                       "close": bar.close, "volume": bar.volume})


def bar_from_json(line: str) -> Bar:
    d = json.loads(line)
    return Bar(ticker=d["ticker"], timestamp=pd.Timestamp(d["timestamp"]),
               open=float(d["open"]), high=float(d["high"]), low=float(d["low"]),
               close=float(d["close"]), volume=float(d["volume"]))


class AlpacaLiveFeed(MarketDataFeed):
    """`publish_path` / `source_path` let two bots share one data stream.

    Alpaca's free IEX feed allows one real-time websocket per Alpaca login,
    and paper accounts under the same login share it: with two bots, the
    second one loops on "connection limit exceeded" all day. So one process
    (the live trader) owns the websocket and appends every bar it receives
    to `publish_path` as a JSON line, and the other (an experiment arm) sets
    `source_path` to the same file and tails it instead of opening its own
    websocket. Both then trade on identical bars. History warm-up still goes
    through each bot's own REST client, which has no such limit.
    """

    # Class-level defaults so instances built without __init__ (the tests)
    # behave like a plain websocket feed.
    publish_path: str | None = None
    source_path: str | None = None
    _publish_lock: threading.Lock | None = None
    _publish_failed: bool = False

    def __init__(self, tickers: Sequence[str], api_key: str, secret_key: str,
                 feed: str = "iex", history_days: int = 10,
                 reconnect_backoff_s: float = 5.0,
                 publish_path: str | None = None, source_path: str | None = None):
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
        if publish_path and source_path:
            raise ValueError("set publish_path or source_path, not both")
        self.publish_path = publish_path
        self.source_path = source_path
        self._publish_lock = threading.Lock()

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

    def _fetch_bars(self, ticker: str, start: datetime, end: datetime, lookback: int) -> list:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=ticker, timeframe=TimeFrame.Minute,
            start=start, end=end, feed=self.data_feed, limit=lookback,
        )
        barset = self._hist_client.get_stock_bars(req)
        # NOT `ticker in barset` / `barset[ticker]`: BarSet is a pydantic
        # BaseModel, and pydantic's own __iter__ (which yields the model's
        # (field_name, value) pairs, for dict(model) support) shadows any
        # dict-like "does this symbol have bars" membership test -- `in`
        # falls back to that field iteration and a ticker string never
        # matches a (field_name, value) tuple, so `ticker in barset` is
        # always False, unconditionally. This was the real reason
        # get_history() always returned "no bars": not a windowing issue
        # at all, every call silently took the `else []` branch regardless
        # of what Alpaca actually sent back. Going through `.data` (the
        # plain dict BarSet wraps) is the one access pattern that isn't
        # shadowed by pydantic's own protocol methods.
        return barset.data.get(ticker, [])

    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        session_open = self._todays_session_open_utc(end)

        bars = []
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
            bars = self._fetch_bars(ticker, session_open, end, lookback)

        if not bars:
            # Either premarket (no "today" window yet -- session_open was
            # not < end above) or, observed live, Alpaca's historical bars
            # endpoint simply not having anything for the still-in-progress
            # current session yet even well after the open -- in both
            # cases, fall back to the old rolling multi-day window rather
            # than leaving warm-up with nothing. This only ever fires when
            # the today-anchored request above came back empty, so it
            # can't reintroduce the inconsistent-bar-count problem on a
            # day where today's bars ARE being served -- it only prevents
            # a regression to zero bars on a day/account where they aren't.
            fallback_start = end - timedelta(days=self.history_days)
            bars = self._fetch_bars(ticker, fallback_start, end, lookback)

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
                    b = Bar(
                        ticker=bar.symbol, timestamp=pd.Timestamp(bar.timestamp),
                        open=float(bar.open), high=float(bar.high),
                        low=float(bar.low), close=float(bar.close), volume=float(bar.volume),
                    )
                    self._publish(b)
                    _q.put(b)

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

    def _publish(self, bar: Bar) -> None:
        """Append `bar` to publish_path. A failed write only costs the
        follower that bar, so it never stops this bot's own trading."""
        if not self.publish_path:
            return
        try:
            with self._publish_lock:
                with open(self.publish_path, "a", encoding="utf-8") as f:
                    f.write(bar_to_json(bar) + "\n")
            if self._publish_failed:
                self._publish_failed = False
                print(f"[AlpacaLiveFeed] publishing bars to {self.publish_path} again")
        except OSError as exc:
            if not self._publish_failed:  # once per outage, not once per bar
                self._publish_failed = True
                print(f"[AlpacaLiveFeed] could not publish bars to {self.publish_path}: {exc!r}")

    def _tail_source_forever(self, poll_s: float = 1.0) -> None:
        """Feed the queue from bars another bot appends to source_path.
        Starts at the file's current end (only new bars; history comes from
        REST), waits for the file to appear, and starts over from the top
        if the file shrinks (truncated or replaced)."""
        tickers = set(self.tickers)
        print(f"[AlpacaLiveFeed] reading bars published to {self.source_path} "
              f"instead of opening a websocket")
        f = None
        skip_existing = True
        pending = ""
        while True:
            if f is None:
                try:
                    f = open(self.source_path, "r", encoding="utf-8")
                except FileNotFoundError:
                    if skip_existing:
                        print(f"[AlpacaLiveFeed] {self.source_path} doesn't exist yet; waiting "
                              f"for the bot that owns the data stream to publish bars")
                    skip_existing = False  # created after we started: all of it is new
                    time.sleep(poll_s)
                    continue
                if skip_existing:
                    f.seek(0, os.SEEK_END)
                skip_existing = False
                pending = ""
            chunk = f.readline()
            if chunk:
                pending += chunk
                if not pending.endswith("\n"):
                    continue  # the writer is mid-line; the rest comes on a later read
                line, pending = pending.strip(), ""
                if not line:
                    continue
                try:
                    bar = bar_from_json(line)
                except (ValueError, KeyError) as exc:
                    print(f"[AlpacaLiveFeed] skipping unreadable bar line ({exc!r})")
                    continue
                if bar.ticker in tickers:
                    self._bar_queue.put(bar)
                continue
            time.sleep(poll_s)
            try:
                shrunk = os.path.getsize(self.source_path) < f.tell()
            except FileNotFoundError:
                shrunk = True
            if shrunk:
                f.close()
                f = None

    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        if self._thread is None:
            target = self._tail_source_forever if self.source_path else self._run_stream_forever
            self._thread = threading.Thread(target=target, daemon=True)
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


class AlpacaHistoricalFeed(HistoricalReplayFeed):
    """Real historical intraday bars from Alpaca between `start` and `end`,
    replayed as if live -- for backtesting periods older than the ~60 days
    of 15-minute history yfinance serves. Regular session (09:30-16:00 ET)
    only, to match yfinance's intraday bars."""

    def __init__(self, tickers: Sequence[str], api_key: str, secret_key: str,
                 start: datetime, end: datetime, minutes: int = 15, feed: str = "iex",
                 warmup_days: int = 45):
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(api_key, secret_key)
        data_feed = DataFeed.IEX if feed.lower() == "iex" else DataFeed.SIP
        series: dict[str, pd.DataFrame] = {}
        for t in tickers:
            req = StockBarsRequest(symbol_or_symbols=t, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                                   start=start - timedelta(days=warmup_days), end=end, feed=data_feed)
            bars = client.get_stock_bars(req).data.get(t, [])  # see _fetch_bars for why .data
            df = pd.DataFrame(
                [{"timestamp": b.timestamp, "open": b.open, "high": b.high, "low": b.low,
                  "close": b.close, "volume": b.volume} for b in bars],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            ).set_index("timestamp")
            df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
            series[t] = df.between_time("09:30", "15:59").dropna()
        # Bars before `start` are only used to warm up (see HistoricalReplayFeed).
        replay_from = pd.Timestamp(start)
        replay_from = replay_from.tz_localize("America/New_York") if replay_from.tzinfo is None \
            else replay_from.tz_convert("America/New_York")
        super().__init__(series, replay_from=replay_from)
