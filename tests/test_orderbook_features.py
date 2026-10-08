import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.orderbook import OrderBookLevel, OrderBookSnapshot
from src.data.orderbook_features import ORDERBOOK_FEATURE_COLUMNS, compute_orderbook_features


def _snap(ts, bids, asks):
    return OrderBookSnapshot(
        "TEST", pd.Timestamp(ts),
        tuple(OrderBookLevel(p, s) for p, s in bids),
        tuple(OrderBookLevel(p, s) for p, s in asks),
    )


def test_single_snapshot_hand_computed_values():
    # best bid 99 (size 30), best ask 101 (size 10) -> mid 100, spread 2
    # obi_l1 = (30-10)/(30+10) = 0.5
    # microprice = (99*10 + 101*30) / 40 = (990 + 3030)/40 = 100.5
    #   -> micro_dev_bps = (100.5-100)/100 * 10000 = 50 bps
    snap = _snap("2024-01-01", bids=[(99.0, 30), (98.5, 5)], asks=[(101.0, 10), (101.5, 5)])
    df = compute_orderbook_features([snap], levels=2)

    assert list(df.columns) == ORDERBOOK_FEATURE_COLUMNS
    row = df.iloc[0]
    assert row["spread_bps"] == pytest.approx(200.0)  # 2 / 100 * 10000
    assert row["obi_l1"] == pytest.approx(0.5)
    assert row["microprice_dev_bps"] == pytest.approx(50.0)
    # obi_l5 (here really "top 2", levels=2): bid depth=35, ask depth=15
    # -> (35-15)/(35+15) = 0.4
    assert row["obi_l5"] == pytest.approx(0.4)


def test_balanced_book_has_zero_imbalance():
    snap = _snap("2024-01-01", bids=[(99.0, 20), (98.5, 20)], asks=[(101.0, 20), (101.5, 20)])
    df = compute_orderbook_features([snap], levels=2)
    row = df.iloc[0]
    assert row["obi_l1"] == pytest.approx(0.0)
    assert row["obi_l5"] == pytest.approx(0.0)
    assert row["microprice_dev_bps"] == pytest.approx(0.0)


def test_missing_side_produces_nan_not_a_crash():
    snap = _snap("2024-01-01", bids=[], asks=[(101.0, 10)])
    df = compute_orderbook_features([snap], levels=2)
    row = df.iloc[0]
    assert np.isnan(row["spread_bps"])
    assert np.isnan(row["obi_l1"])
    assert np.isnan(row["microprice_dev_bps"])


def test_depth_z_score_is_nan_during_warmup_then_finite():
    # constant depth for the first few snapshots, then a clear spike --
    # the z-score should be NaN while the rolling window isn't full yet,
    # and clearly positive once the spike is inside the window.
    snaps = []
    for i in range(5):
        snaps.append(_snap(f"2024-01-01 00:{i:02d}:00", bids=[(99.0, 100)], asks=[(101.0, 100)]))
    snaps.append(_snap("2024-01-01 00:05:00", bids=[(99.0, 1000)], asks=[(101.0, 100)]))

    df = compute_orderbook_features(snaps, levels=1, depth_z_window=3)
    assert np.isnan(df.iloc[0]["depth_bid_z"])  # window not full yet
    assert df.iloc[-1]["depth_bid_z"] > 1.0  # spike stands out from recent history


def test_output_is_causal_truncating_snapshots_does_not_change_earlier_rows():
    # Feature at row i must not depend on snapshots after i (no lookahead).
    snaps = []
    for i in range(10):
        bid_size = 100 + i * 5
        snaps.append(_snap(f"2024-01-01 00:{i:02d}:00", bids=[(99.0, bid_size)], asks=[(101.0, 100)]))

    full = compute_orderbook_features(snaps, levels=1, depth_z_window=3)
    truncated = compute_orderbook_features(snaps[:6], levels=1, depth_z_window=3)

    pd.testing.assert_frame_equal(full.iloc[:6], truncated)


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
