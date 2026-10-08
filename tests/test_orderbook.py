import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.data.orderbook import (
    AlpacaCryptoOrderBookFeed,
    EquityL2FeedStub,
    OrderBookLevel,
    OrderBookSnapshot,
    SyntheticOrderBookFeed,
)


def _snap(bids, asks, ts="2024-01-01"):
    import pandas as pd
    return OrderBookSnapshot(
        "TEST", pd.Timestamp(ts),
        tuple(OrderBookLevel(p, s) for p, s in bids),
        tuple(OrderBookLevel(p, s) for p, s in asks),
    )


def test_snapshot_derived_properties():
    snap = _snap(bids=[(99.0, 10), (98.5, 20)], asks=[(101.0, 5), (101.5, 15)])
    assert snap.best_bid == OrderBookLevel(99.0, 10)
    assert snap.best_ask == OrderBookLevel(101.0, 5)
    assert snap.mid_price == pytest.approx(100.0)
    assert snap.spread == pytest.approx(2.0)
    assert snap.spread_bps == pytest.approx(2.0 / 100.0 * 10_000)


def test_snapshot_handles_empty_side():
    snap = _snap(bids=[], asks=[(101.0, 5)])
    assert snap.best_bid is None
    assert snap.mid_price is None
    assert snap.spread is None
    assert snap.spread_bps is None


def test_synthetic_feed_is_deterministic_given_seed():
    a = SyntheticOrderBookFeed(["SPY"], n_snapshots=50, seed=42)
    b = SyntheticOrderBookFeed(["SPY"], n_snapshots=50, seed=42)
    snap_a = a.get_snapshot("SPY")
    snap_b = b.get_snapshot("SPY")
    assert snap_a.bids == snap_b.bids
    assert snap_a.asks == snap_b.asks
    assert snap_a.timestamp == snap_b.timestamp


def test_synthetic_feed_book_shape_and_ordering():
    feed = SyntheticOrderBookFeed(["SPY"], n_snapshots=20, n_levels=10, seed=1)
    snap = feed.get_snapshot("SPY")
    assert len(snap.bids) == 10
    assert len(snap.asks) == 10
    # bids descending, asks ascending -- same convention real venues use
    bid_prices = [lvl.price for lvl in snap.bids]
    ask_prices = [lvl.price for lvl in snap.asks]
    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ask_prices == sorted(ask_prices)
    assert snap.best_bid.price < snap.best_ask.price
    assert all(lvl.size > 0 for lvl in snap.bids + snap.asks)


def test_synthetic_feed_depth_decays_away_from_touch():
    # Liquidity should generally thin out away from the best price, same
    # shape as a real book -- not required to be strictly monotonic every
    # single snapshot (randomness), but the top level should usually beat
    # the average of the rest.
    feed = SyntheticOrderBookFeed(["SPY"], n_snapshots=200, n_levels=10, seed=3)
    top_bigger_count = 0
    for _ in range(200):
        snap = feed.get_snapshot("SPY")
        if snap is None:
            break
        rest_avg = sum(lvl.size for lvl in snap.bids[1:]) / max(1, len(snap.bids) - 1)
        if snap.bids[0].size > rest_avg:
            top_bigger_count += 1
    assert top_bigger_count > 150  # clearly the common case, not a coin flip


def test_get_history_then_stream_does_not_duplicate():
    # Same bug class as feed.py's SyntheticFeed/YFinanceFeed cursor fix:
    # a warm-up get_history() call must not cause stream() to replay bars.
    feed = SyntheticOrderBookFeed(["SPY"], n_snapshots=30, seed=5)
    history = feed.get_history("SPY", lookback=10)
    assert len(history) == 10
    streamed = list(feed.stream(["SPY"]))
    assert len(streamed) == 20  # the remaining 20, not 30
    # no timestamp overlap between the warm-up history and the stream
    history_ts = {s.timestamp for s in history}
    streamed_ts = {s.timestamp for s in streamed}
    assert history_ts.isdisjoint(streamed_ts)


def test_stream_interleaves_multiple_tickers_in_time_order():
    feed = SyntheticOrderBookFeed(["A", "B"], n_snapshots=5, seed=9)
    out = list(feed.stream(["A", "B"]))
    assert len(out) == 10
    tickers_seen = [s.ticker for s in out]
    assert tickers_seen == ["A", "B"] * 5


def test_equity_l2_stub_refuses_to_construct():
    with pytest.raises(NotImplementedError):
        EquityL2FeedStub(api_key="x", secret_key="y")


def test_alpaca_crypto_feed_constructs_without_network():
    feed = AlpacaCryptoOrderBookFeed(["BTC/USD"], api_key="x", secret_key="y")
    assert feed.symbols == ["BTC/USD"]
