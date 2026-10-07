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


class YFinanceFeed(MarketDataFeed):
    """Real historical bars via `yfinance`, replayed as if live.

    This is appropriate for paper trading / backtesting against real
    history. It is NOT a production live-data connection: yfinance data is
    delayed/unofficial and has no delivery guarantees. Swap this class for a
    real broker/vendor websocket client before trading with real money.
    """

    def __init__(self, tickers: Sequence[str], period: str = "60d", interval: str = "15m"):
        import yfinance as yf  # local import: optional dependency

        self.tickers = list(tickers)
        self._series: dict[str, pd.DataFrame] = {}
        for t in self.tickers:
            df = yf.download(t, period=period, interval=interval, progress=False)
            df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            df = df.dropna()
            self._series[t] = df
        self._cursor = {t: 0 for t in self.tickers}

    def get_history(self, ticker: str, lookback: int) -> pd.DataFrame:
        df = self._series[ticker]
        cur = self._cursor[ticker]
        if cur == 0:
            # See SyntheticFeed.get_history: advance the cursor on this
            # warm-up call so stream() doesn't replay the same bars twice.
            self._cursor[ticker] = min(lookback, len(df))
            return df.iloc[: self._cursor[ticker]].copy()
        start = max(0, cur - lookback)
        return df.iloc[start:cur].copy()

    def stream(self, tickers: Sequence[str]) -> Iterator[Bar]:
        start_i = max(self._cursor[t] for t in tickers)
        max_len = min(len(self._series[t]) for t in tickers)
        for i in range(start_i, max_len):
            for t in tickers:
                self._cursor[t] = i + 1
                row = self._series[t].iloc[i]
                yield Bar(t, self._series[t].index[i], float(row.open), float(row.high),
                          float(row.low), float(row.close), float(row.volume))
