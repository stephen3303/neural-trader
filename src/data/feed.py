"""
Market data feeds.

The system is built around a tiny abstract interface (`MarketDataFeed`) so the
rest of the pipeline never cares whether bars are coming from a CSV, a
synthetic generator, or a real broker/data-vendor websocket. Swapping in a
live feed (Alpaca, Polygon, IBKR, etc.) later means writing one new class
here -- nothing else changes.

Two concrete feeds are provided out of the box:

- `SyntheticFeed`   -- no network required, generates a plausible multi-regime
                       price series (trend + mean-reversion + volatility
                       clustering + occasional shocks). Use this for the smoke
                       test and for unit tests, since it's deterministic given
                       a seed and never depends on an external API being up.
- `YFinanceFeed`    -- pulls real historical bars via `yfinance`. This is
                       still NOT a live broker feed -- it's a convenient stand
                       in for "real but delayed/free" data during paper
                       trading and backtesting.

Both feeds expose the same two operations the rest of the system needs:

    get_history(ticker, lookback)      -> pd.DataFrame of past bars (warm-up)
    stream(tickers)                    -> generator yielding one new bar at a
                                           time across tickers, in time order,
                                           simulating the live market clock

`stream()` is what makes "continuous monitoring" concrete: the orchestrator
just iterates it forever. For a true live feed, a real implementation would
block on a websocket message instead of advancing an internal pointer.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bar:
    ticker: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataFeed(abc.ABC):
    """Abstract base class every data source implements."""

    @abc.abstractmethod
    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        """Return the most recent `lookback` bars for `ticker` as a
        DataFrame indexed by timestamp with columns
        [open, high, low, close, volume]. Used for warm-up / backtesting."""

    @abc.abstractmethod
    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        """Yield bars, one at a time, across all `tickers`, advancing in
        time order. In a real live feed this would block until the next
        tick/bar arrives; here it plays back history (or synthetic data)
        to simulate that clock."""


class SyntheticFeed(MarketDataFeed):
    """Deterministic synthetic multi-regime price generator.

    Each ticker gets its own regime-switching geometric Brownian motion:
    drift and volatility periodically jump to a new (mu, sigma) pair, which
    stands in for the kind of regime shift a real continual-learning system
    has to detect and adapt to. A small number of volume-correlated jumps are
    injected to resemble news shocks.
    """

    def __init__(self, tickers: Sequence[str], n_bars: int = 5000, seed: int = 7,
                 bar_seconds: int = 60):
        self.tickers = list(tickers)
        self.n_bars = n_bars
        self.bar_seconds = bar_seconds
        self._series: dict[str, pd.DataFrame] = {}
        rng = np.random.default_rng(seed)
        for t in self.tickers:
            self._series[t] = self._generate(rng, n_bars)
        self._cursor = {t: 0 for t in self.tickers}

    def _generate(self, rng: np.random.Generator, n_bars: int) -> pd.DataFrame:
        price = 100.0 + rng.normal(0, 5)
        regime_len = max(50, n_bars // 10)
        prices, vols = [], []
        mu, sigma = 0.0, 0.01
        for i in range(n_bars):
            if i % regime_len == 0:
                mu = rng.normal(0, 0.0004)
                sigma = abs(rng.normal(0.008, 0.004)) + 0.002
            shock = 0.0
            if rng.random() < 0.002:
                shock = rng.normal(0, 0.05)
            ret = rng.normal(mu, sigma) + shock
            price *= (1 + ret)
            price = max(price, 0.5)
            prices.append(price)
            vols.append(abs(rng.normal(1_000_000, 300_000) * (1 + 10 * abs(ret))))

        prices = np.array(prices)
        opens = prices * (1 + rng.normal(0, 0.0005, size=n_bars))
        highs = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.0015, size=n_bars)))
        lows = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.0015, size=n_bars)))
        idx = pd.date_range("2024-01-01", periods=n_bars, freq="min")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": vols},
            index=idx,
        )

    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        df = self._series[ticker]
        cur = self._cursor[ticker]
        if cur == 0:
            # Pre-stream warm-up call (Orchestrator.run() does this before its
            # main loop starts): hand back the first `lookback` bars *and*
            # advance the cursor so stream() continues right after them,
            # instead of replaying the same bars a second time.
            self._cursor[ticker] = min(lookback, len(df))
            return df.iloc[: self._cursor[ticker]].copy()
        start = max(0, cur - lookback)
        return df.iloc[start:cur].copy()

    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        # Advance a shared pointer across all tickers in round-robin, so
        # bars interleave in (approximately) the same order a real
        # multi-symbol feed would deliver them. Starts from each ticker's
        # current cursor, so a prior get_history() warm-up call is honored.
        start_i = max(self._cursor[t] for t in tickers)
        max_len = min(len(self._series[t]) for t in tickers)
        for i in range(start_i, max_len):
            for t in tickers:
                self._cursor[t] = i + 1
                row = self._series[t].iloc[i]
                yield Bar(t, self._series[t].index[i], row.open, row.high,
                          row.low, row.close, row.volume)


class HistoricalReplayFeed(MarketDataFeed):
    """Replays a fixed set of per-ticker OHLCV DataFrames as if live.
    Subclasses only have to fill `self._series` and `self._cursor`.

    `replay_from`, when given, is the first timestamp to replay: bars
    before it are only used to answer the warm-up `get_history()` call, so
    a backtest asked to start on a date really starts trading on it
    instead of spending its first `lookback` bars warming up."""

    def __init__(self, series: dict[str, pd.DataFrame], replay_from=None):
        self.tickers = list(series)
        self._series = series
        self._cursor = {t: 0 for t in self.tickers}
        self._first = {t: 0 for t in self.tickers}
        if replay_from is not None:
            for t, df in series.items():
                ts = pd.Timestamp(replay_from)
                if df.index.tz is not None and ts.tzinfo is None:
                    ts = ts.tz_localize(df.index.tz)
                self._first[t] = int(df.index.searchsorted(ts))

    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        df = self._series[ticker]
        cur = self._cursor[ticker]
        if cur == 0:
            first = self._first[ticker]
            if first > 0:
                self._cursor[ticker] = first
                return df.iloc[max(0, first - lookback):first].copy()
            # See SyntheticFeed.get_history: advance the cursor on this
            # warm-up call so stream() doesn't replay the same bars twice.
            self._cursor[ticker] = min(lookback, len(df))
            return df.iloc[: self._cursor[ticker]].copy()
        start = max(0, cur - lookback)
        return df.iloc[start:cur].copy()

    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        # Merge by timestamp, not by row number: a ticker with a missing
        # interval (common in IEX data) would otherwise drift out of step,
        # replaying one ticker's later bar before another's earlier one.
        order = {t: k for k, t in enumerate(tickers)}
        events = []
        for t in tickers:
            df = self._series[t]
            begin = max(self._cursor[t], self._first[t])
            events.extend((df.index[i], order[t], i) for i in range(begin, len(df)))
        events.sort(key=lambda e: (e[0], e[1]))
        for ts, k, i in events:
            t = tickers[k]
            self._cursor[t] = i + 1
            row = self._series[t].iloc[i]
            yield Bar(t, ts, float(row.open), float(row.high),
                      float(row.low), float(row.close), float(row.volume))


class CSVFeed(HistoricalReplayFeed):
    """Real bars saved earlier to <data_dir>/<TICKER>.csv (columns
    timestamp,open,high,low,close,volume), replayed as if live -- so a
    backtest can rerun on exactly the same history without re-downloading
    it. Replay starts at `start` (YYYY-MM-DD), with the bars before it used
    only for warm-up, and stops before `end` (exclusive)."""

    def __init__(self, tickers: Sequence[str], data_dir: str, start: str | None = None,
                 end: str | None = None):
        series: dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = pd.read_csv(f"{data_dir}/{t}.csv")
            df.index = pd.to_datetime(df.pop("timestamp"), utc=True).dt.tz_convert("America/New_York")
            df = df.sort_index()
            if end:
                df = df[df.index < pd.Timestamp(end, tz="America/New_York")]
            series[t] = df[["open", "high", "low", "close", "volume"]].dropna()
        super().__init__(series, replay_from=pd.Timestamp(start, tz="America/New_York") if start else None)


class YFinanceFeed(HistoricalReplayFeed):
    """Real historical bars via `yfinance`, replayed as if live.

    This is appropriate for paper trading / backtesting against real
    history. It is NOT a production live-data connection: yfinance data is
    delayed/unofficial and has no delivery guarantees. Swap this class for a
    real broker/vendor websocket client before trading with real money.
    """

    def __init__(self, tickers: Sequence[str], period: str = "60d", interval: str = "15m"):
        import yfinance as yf  # local import: optional dependency

        series: dict[str, pd.DataFrame] = {}
        for t in tickers:
            # multi_level_index=False: recent yfinance returns (field, ticker)
            # MultiIndex columns by default, which breaks the rename/select below.
            df = yf.download(t, period=period, interval=interval, progress=False,
                             multi_level_index=False, auto_adjust=True)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            series[t] = df.dropna()
        super().__init__(series)

