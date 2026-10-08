"""
Level 2 (order book depth) data feeds.

**Read this before wiring anything up:** Alpaca does NOT provide Level 2
depth for US equities -- `StockDataStream` only exposes top-of-book NBBO
quotes (`subscribe_quotes`), which is Level 1. Verified directly against
the installed `alpaca-py` SDK (v0.44.0) while building this: `CryptoDataStream`
has `subscribe_orderbooks`/`Orderbook`/`OrderbookQuote`, `StockDataStream`
does not. So for SPY/QQQ/AAPL -- the tickers this project actually
trades -- there is no real L2 feed available through Alpaca at all. A
true equities L2 feed needs a different vendor; see `EquityL2FeedStub`
below for what that integration would require.

What IS real here: Alpaca's crypto data API genuinely does provide L2
depth (`AlpacaCryptoOrderBookFeed`), so if this project ever trades crypto
pairs, that path is a working integration, not a stand-in. For equities,
`SyntheticOrderBookFeed` is what lets the feature-engineering and model
pipeline be built and tested today, ahead of (and independent of) a real
equities L2 subscription.

Mirrors the `MarketDataFeed` pattern in `feed.py` on purpose: the rest of
the system should never care whether a book snapshot came from a real
exchange feed or a synthetic generator, so swapping one `OrderBookFeed`
implementation for another is the only thing a real integration should
require.
"""

from __future__ import annotations

import abc
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    """bids: best-to-worst (descending price). asks: best-to-worst
    (ascending price). Both may be shorter than the venue's full depth --
    callers should treat a missing level as "no liquidity reported there",
    not "zero liquidity there"."""

    ticker: str
    timestamp: pd.Timestamp
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]

    @property
    def best_bid(self) -> OrderBookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> OrderBookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price

    @property
    def spread_bps(self) -> float | None:
        mid, spread = self.mid_price, self.spread
        if mid is None or spread is None or mid == 0:
            return None
        return spread / mid * 10_000


class OrderBookFeed(abc.ABC):
    """Abstract base every L2 source implements -- see module docstring
    for which of the concrete classes below are real vs. synthetic."""

    @abc.abstractmethod
    def get_snapshot(self, ticker: str) -> OrderBookSnapshot | None:
        """One current snapshot for `ticker`, or None if unavailable."""

    @abc.abstractmethod
    def stream(self, tickers: Sequence[str]) -> Iterator[OrderBookSnapshot]:
        """Yield snapshots, one at a time, across all `tickers`, advancing
        in time order -- same shape as `MarketDataFeed.stream()`."""


class SyntheticOrderBookFeed(OrderBookFeed):
    """Deterministic synthetic L2 book generator, for building and testing
    the feature pipeline without needing a real depth subscription.

    Builds a plausible multi-level book around a synthetic mid-price
    random walk: spread widens when the synthetic volatility regime is
    high, depth decays geometrically away from the best level (thinner
    liquidity further from the touch, as in a real book), and a small
    random size imbalance between bid/ask sides stands in for the
    short-term order flow pressure a real book would show.
    """

    def __init__(self, tickers: Sequence[str], n_snapshots: int = 5000, seed: int = 11,
                 n_levels: int = 10, base_spread_bps: float = 2.0, tick_size: float = 0.01,
                 depth_decay: float = 0.7, base_size: float = 500.0,
                 snapshot_seconds: int = 1):
        self.tickers = list(tickers)
        self.n_levels = n_levels
        self.snapshot_seconds = snapshot_seconds
        self._series: dict[str, list[OrderBookSnapshot]] = {}
        rng = np.random.default_rng(seed)
        for t in self.tickers:
            self._series[t] = self._generate(
                rng, t, n_snapshots, n_levels, base_spread_bps, tick_size, depth_decay, base_size
            )
        self._cursor = {t: 0 for t in self.tickers}

    @staticmethod
    def _generate(rng, ticker, n_snapshots, n_levels, base_spread_bps, tick_size,
                   depth_decay, base_size) -> list[OrderBookSnapshot]:
        mid = 100.0 + rng.normal(0, 5)
        sigma = 0.01
        snapshots = []
        start = pd.Timestamp("2024-01-01")
        for i in range(n_snapshots):
            if i % 500 == 0:
                sigma = abs(rng.normal(0.008, 0.004)) + 0.001
            mid *= 1 + rng.normal(0, sigma)
            mid = max(mid, tick_size * n_levels * 2)

            spread_bps = max(base_spread_bps * (1 + sigma * 50) + rng.normal(0, 0.5), base_spread_bps * 0.5)
            half_spread = mid * spread_bps / 10_000 / 2
            best_bid = round(mid - half_spread, 2)
            best_ask = round(mid + half_spread, 2)

            # Random short-term imbalance between the two sides -- this is
            # the signal order-book-imbalance features are meant to pick up.
            imbalance = np.clip(rng.normal(0, 0.3), -0.9, 0.9)
            bid_scale = base_size * (1 + imbalance)
            ask_scale = base_size * (1 - imbalance)

            bids = tuple(
                OrderBookLevel(
                    price=round(best_bid - lvl * tick_size, 2),
                    size=max(1.0, bid_scale * (depth_decay ** lvl) * abs(rng.normal(1.0, 0.2))),
                )
                for lvl in range(n_levels)
            )
            asks = tuple(
                OrderBookLevel(
                    price=round(best_ask + lvl * tick_size, 2),
                    size=max(1.0, ask_scale * (depth_decay ** lvl) * abs(rng.normal(1.0, 0.2))),
                )
                for lvl in range(n_levels)
            )
            ts = start + pd.Timedelta(seconds=i)
            snapshots.append(OrderBookSnapshot(ticker, ts, bids, asks))
        return snapshots

    def get_snapshot(self, ticker: str) -> OrderBookSnapshot | None:
        series = self._series[ticker]
        cur = self._cursor[ticker]
        if cur >= len(series):
            return None
        snap = series[cur]
        self._cursor[ticker] = cur + 1
        return snap

    def get_history(self, ticker: str, lookback: int) -> list[OrderBookSnapshot]:
        """Analogous to MarketDataFeed.get_history: hands back the most
        recent `lookback` snapshots for warm-up, advancing the cursor on
        first call so a later stream() doesn't replay them (same fix as
        feed.py's SyntheticFeed/YFinanceFeed -- see those for why this
        matters)."""
        series = self._series[ticker]
        cur = self._cursor[ticker]
        if cur == 0:
            self._cursor[ticker] = min(lookback, len(series))
            return series[: self._cursor[ticker]]
        start = max(0, cur - lookback)
        return series[start:cur]

    def stream(self, tickers: Sequence[str]) -> Iterator[OrderBookSnapshot]:
        start_i = max(self._cursor[t] for t in tickers)
        max_len = min(len(self._series[t]) for t in tickers)
        for i in range(start_i, max_len):
            for t in tickers:
                self._cursor[t] = i + 1
                yield self._series[t][i]


class AlpacaCryptoOrderBookFeed(OrderBookFeed):
    """REAL L2 depth via Alpaca's crypto data API -- this is not a stand-in
    like SyntheticOrderBookFeed. Alpaca genuinely streams multi-level order
    books for crypto pairs (`CryptoDataStream.subscribe_orderbooks`),
    unlike equities. Use this if/when this project trades crypto; for the
    SPY/QQQ/AAPL equities this project currently trades, it does not apply
    -- see the module docstring and `EquityL2FeedStub`.

    `symbols` are Alpaca crypto pair symbols, e.g. "BTC/USD", not stock
    tickers.
    """

    def __init__(self, symbols: Sequence[str], api_key: str, secret_key: str,
                 n_levels: int = 10, reconnect_backoff_s: float = 5.0):
        self.symbols = list(symbols)
        self.api_key = api_key
        self.secret_key = secret_key
        self.n_levels = n_levels
        self.reconnect_backoff_s = reconnect_backoff_s
        self._snap_queue: queue.Queue = queue.Queue()
        self._stream_thread: threading.Thread | None = None

    @staticmethod
    def _to_snapshot(ob, n_levels: int) -> OrderBookSnapshot:
        bids = tuple(OrderBookLevel(q.price, q.size) for q in (ob.bids or [])[:n_levels])
        asks = tuple(OrderBookLevel(q.price, q.size) for q in (ob.asks or [])[:n_levels])
        ts = pd.Timestamp(ob.timestamp) if ob.timestamp is not None else pd.Timestamp.utcnow()
        return OrderBookSnapshot(ob.symbol, ts, bids, asks)

    def get_snapshot(self, ticker: str) -> OrderBookSnapshot | None:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoLatestOrderbookRequest

        client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        result = client.get_crypto_latest_orderbook(CryptoLatestOrderbookRequest(symbol_or_symbols=ticker))
        ob = result.get(ticker)
        return self._to_snapshot(ob, self.n_levels) if ob is not None else None

    def _run_stream_forever(self) -> None:
        from alpaca.data.live.crypto import CryptoDataStream

        while True:
            try:
                stream = CryptoDataStream(self.api_key, self.secret_key)

                async def on_orderbook(ob):
                    self._snap_queue.put(self._to_snapshot(ob, self.n_levels))

                stream.subscribe_orderbooks(on_orderbook, *self.symbols)
                stream.run()  # blocks until disconnected
            except Exception as e:  # noqa: BLE001 -- reconnect on anything, log and retry
                print(f"AlpacaCryptoOrderBookFeed stream error: {e!r}; "
                      f"reconnecting in {self.reconnect_backoff_s:.0f}s...")
            time.sleep(self.reconnect_backoff_s)

    def stream(self, tickers: Sequence[str]) -> Iterator[OrderBookSnapshot]:
        if self._stream_thread is None:
            self._stream_thread = threading.Thread(target=self._run_stream_forever, daemon=True)
            self._stream_thread.start()
        while True:
            yield self._snap_queue.get()


class EquityL2FeedStub(OrderBookFeed):
    """Skeleton for a real US-equities L2 feed. NOT functional -- Alpaca
    cannot be wired up here (it doesn't offer equities depth at all, see
    the module docstring); this needs a vendor that actually sells L2/ITCH
    depth. Options, roughly cheapest/simplest to most complete:

    - Polygon.io -- "Launchpad"/higher tiers add full market depth for
      stocks; simplest REST+websocket integration of this list.
    - Databento -- sells raw MBP-10 (market-by-price, 10 levels) and
      MBO (market-by-order) feeds per exchange; you reconstruct the book
      from their normalized schema.
    - IEX Cloud DEEP -- IEX's own full order book depth (IEX's slice of
      the market only, not consolidated across all exchanges).
    - Direct exchange feeds (Nasdaq TotalView, NYSE OpenBook, etc.) --
      most complete and most expensive; typically needs co-location or a
      specialized feed handler, well beyond what this project needs.

    Whichever vendor, a real implementation has to add things
    SyntheticOrderBookFeed glosses over entirely:
    - binary/ITCH-style protocol parsing (most L2 feeds are not JSON)
    - book reconstruction from add/cancel/modify/execute messages, with
      sequence-number gap detection (a dropped message silently corrupts
      the whole reconstructed book)
    - snapshot + delta reconciliation on reconnect
    - per-symbol subscription/entitlement limits and cost -- L2 equities
      data is usually priced per symbol and is not cheap at any real
      scale
    """

    def __init__(self, api_key: str, secret_key: str, **kwargs):
        raise NotImplementedError(
            "EquityL2FeedStub is a placeholder. Alpaca does not provide equities "
            "L2 depth -- wire this up against a real L2 vendor (see class docstring "
            "for options) before using this class."
        )

    def get_snapshot(self, ticker: str) -> OrderBookSnapshot | None:
        raise NotImplementedError

    def stream(self, tickers: Sequence[str]) -> Iterator[OrderBookSnapshot]:
        raise NotImplementedError
