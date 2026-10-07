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

import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Sequence

import pandas as pd

from .feed import Bar, MarketDataFeed


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

    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now(timezone.utc)
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
            except Exception as exc:  # noqa: BLE001 -- keep the feed alive across transient drops
                print(f"[AlpacaLiveFeed] stream error ({exc!r}); reconnecting in "
                      f"{self.reconnect_backoff_s:.0f}s")
                time.sleep(self.reconnect_backoff_s)

    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_stream_forever, daemon=True)
            self._thread.start()
        while True:
            yield self._bar_queue.get()  # blocks until the next real-time bar arrives
