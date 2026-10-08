"""
Feature engineering over Level 2 order book snapshots.

Same design rules as `features.py`: every feature is causal (uses only
the snapshot at time t and, for the rolling z-scores, snapshots strictly
before it) and expressed as a ratio/z-score/bounded measure rather than a
raw price or size, so it's stationary-ish and comparable across tickers
and venues. This module is deliberately independent of `features.py` --
see the README section "Order book (Level 2) features" for how to concat
its output onto the existing per-bar feature matrix if/when you wire it
into the model (that's a `n_features` + retrain change, not done here).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.data.orderbook import OrderBookSnapshot

ORDERBOOK_FEATURE_COLUMNS = [
    "spread_bps",
    "obi_l1",
    "obi_l5",
    "microprice_dev_bps",
    "depth_bid_z",
    "depth_ask_z",
]


def _top_n_size(levels, n: int) -> float:
    return float(sum(lvl.size for lvl in levels[:n]))


def _microprice(snap: OrderBookSnapshot) -> float | None:
    """Size-weighted price that leans toward whichever side has LESS size
    at the touch (the classic microprice formula): a thin ask relative to
    the bid implies the next trade is more likely to lift the ask, so the
    "fair" price sits closer to the ask, and vice versa. A standard,
    well-studied short-term price-pressure signal in market microstructure."""
    bb, ba = snap.best_bid, snap.best_ask
    if bb is None or ba is None:
        return None
    denom = bb.size + ba.size
    if denom == 0:
        return None
    return (bb.price * ba.size + ba.price * bb.size) / denom


def compute_orderbook_features(snapshots: Sequence[OrderBookSnapshot], levels: int = 5,
                                depth_z_window: int = 30) -> pd.DataFrame:
    """snapshots: time-ordered L2 snapshots for ONE ticker (mirrors
    compute_features()'s per-ticker contract in features.py -- call this
    once per ticker, same as the price-feature pipeline).

    Returns a DataFrame indexed by timestamp with ORDERBOOK_FEATURE_COLUMNS,
    with NaN warm-up rows at the start (same convention as compute_features
    -- caller drops/skips those, build_windows() already does via its
    np.isnan(chunk).any() check)."""
    rows = []
    index = []
    raw_bid_depth, raw_ask_depth = [], []

    for snap in snapshots:
        mid = snap.mid_price
        spread_bps = snap.spread_bps

        bb, ba = snap.best_bid, snap.best_ask
        obi_l1 = np.nan
        if bb is not None and ba is not None and (bb.size + ba.size) > 0:
            obi_l1 = (bb.size - ba.size) / (bb.size + ba.size)

        bid_depth_l = _top_n_size(snap.bids, levels)
        ask_depth_l = _top_n_size(snap.asks, levels)
        obi_l5 = np.nan
        if (bid_depth_l + ask_depth_l) > 0:
            obi_l5 = (bid_depth_l - ask_depth_l) / (bid_depth_l + ask_depth_l)

        micro = _microprice(snap)
        micro_dev_bps = np.nan
        if micro is not None and mid is not None and mid != 0:
            micro_dev_bps = (micro - mid) / mid * 10_000

        raw_bid_depth.append(bid_depth_l)
        raw_ask_depth.append(ask_depth_l)
        rows.append({
            "spread_bps": spread_bps if spread_bps is not None else np.nan,
            "obi_l1": obi_l1,
            "obi_l5": obi_l5,
            "microprice_dev_bps": micro_dev_bps,
        })
        index.append(snap.timestamp)

    df = pd.DataFrame(rows, index=pd.Index(index, name="timestamp"))

    # Depth is a size, not a ratio -- z-score it against its own trailing
    # window (same pattern as vol_z in features.py) so the network sees a
    # "is liquidity thinner/thicker than usual right now" signal instead of
    # a raw share count that has no fixed scale across tickers.
    bid_depth = pd.Series(raw_bid_depth, index=df.index)
    ask_depth = pd.Series(raw_ask_depth, index=df.index)
    df["depth_bid_z"] = (bid_depth - bid_depth.rolling(depth_z_window).mean()) / \
        bid_depth.rolling(depth_z_window).std().replace(0, np.nan)
    df["depth_ask_z"] = (ask_depth - ask_depth.rolling(depth_z_window).mean()) / \
        ask_depth.rolling(depth_z_window).std().replace(0, np.nan)

    return df[ORDERBOOK_FEATURE_COLUMNS]
