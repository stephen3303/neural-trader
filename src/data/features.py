"""
Feature engineering: turns raw OHLCV bars into the fixed-width numeric
windows the neural network consumes, plus the forward-looking labels used
to train it.

Design choices, and why:

- All features are *causal* (computed only from bars up to and including
  time t) so there is no lookahead leakage -- critical for anything that
  will eventually size real orders.
- Features are expressed as returns / z-scores / bounded oscillators
  rather than raw price levels, so the network sees stationary-ish inputs
  and the same weights generalize across tickers with wildly different
  price levels ($5 vs $500).
- The label is the realized forward return over a configurable horizon,
  discretized into {sell, hold, buy} with a transaction-cost-aware
  deadband -- the model is never rewarded for calling a move too small to
  clear costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_15",
    "sma_10_dev", "sma_30_dev",
    "ema_12_dev", "ema_26_dev",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos",
    "vol_z",
    "realized_vol_15",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: DataFrame[open, high, low, close, volume] indexed by time.
    Returns a DataFrame of engineered features, same index, with the
    warm-up NaN rows at the start (caller should drop/skip those)."""
    df = bars.copy()
    close = df["close"]

    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ret_15"] = close.pct_change(15)

    sma10 = close.rolling(10).mean()
    sma30 = close.rolling(30).mean()
    df["sma_10_dev"] = (close - sma10) / sma10
    df["sma_30_dev"] = (close - sma30) / sma30

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["ema_12_dev"] = (close - ema12) / ema12
    df["ema_26_dev"] = (close - ema26) / ema26

    df["rsi_14"] = _rsi(close, 14) / 100.0  # scale to ~[0,1]

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd / close
    df["macd_signal"] = macd_signal / close
    df["macd_hist"] = (macd - macd_signal) / close

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    vol = df["volume"]
    vol_mean = vol.rolling(30).mean()
    vol_std = vol.rolling(30).std().replace(0, np.nan)
    df["vol_z"] = (vol - vol_mean) / vol_std

    df["realized_vol_15"] = close.pct_change().rolling(15).std()

    return df[FEATURE_COLUMNS]


@dataclass
class LabelConfig:
    horizon: int = 15          # bars ahead to measure forward return
    deadband_bps: float = 8.0  # +/- threshold (basis points) that counts as "hold"


def make_labels(bars: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """Forward return over `horizon` bars, discretized into
    {-1: sell, 0: hold, 1: buy} using a cost-aware deadband so the model
    isn't trained to chase noise smaller than it could realistically
    capture after spread/slippage."""
    close = bars["close"]
    fwd_ret = close.shift(-cfg.horizon) / close - 1.0
    deadband = cfg.deadband_bps / 10_000.0
    action = pd.Series(0, index=bars.index, dtype=int)
    action[fwd_ret > deadband] = 1
    action[fwd_ret < -deadband] = -1
    return pd.DataFrame({"fwd_ret": fwd_ret, "action": action})


def build_windows(features: pd.DataFrame, labels: pd.DataFrame | None, window: int):
    """Slide a `window`-length lookback over `features`, producing one
    sample per valid end-index. If `labels` is given, pairs each window
    with the label whose index matches the window's last timestamp (i.e.
    "given everything up to and including bar t, what happens next").

    Returns (X, idx) if labels is None  -- X: float32 array [n, window, n_feat]
    Returns (X, y_action, y_ret, idx) if labels is given.
    """
    feat_arr = features.to_numpy(dtype=np.float32)
    n = len(features)
    xs, idxs = [], []
    for end in range(window, n + 1):
        start = end - window
        chunk = feat_arr[start:end]
        if np.isnan(chunk).any():
            continue
        xs.append(chunk)
        idxs.append(features.index[end - 1])

    if not xs:
        empty_x = np.zeros((0, window, len(FEATURE_COLUMNS)), dtype=np.float32)
        if labels is None:
            return empty_x, []
        return empty_x, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32), []

    X = np.stack(xs)
    if labels is None:
        return X, idxs

    y_action, y_ret, keep_idx, keep_X = [], [], [], []
    for x, ts in zip(X, idxs):
        if ts not in labels.index:
            continue
        row = labels.loc[ts]
        if pd.isna(row["fwd_ret"]):
            continue
        keep_X.append(x)
        y_action.append(int(row["action"]) + 1)  # shift {-1,0,1} -> {0,1,2} for CE loss
        y_ret.append(float(row["fwd_ret"]))
        keep_idx.append(ts)

    if not keep_X:
        empty_x = np.zeros((0, window, len(FEATURE_COLUMNS)), dtype=np.float32)
        return empty_x, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32), []

    return np.stack(keep_X), np.array(y_action, dtype=np.int64), np.array(y_ret, dtype=np.float32), keep_idx


def summarize_window(X: np.ndarray) -> np.ndarray:
    """Collapse a sequence window [n, window, n_feat] (the shape
    build_windows() above produces, and what TradingNet's GRU consumes
    directly) into a flat per-sample feature vector [n, n_feat * 6], for
    a model that takes ordinary tabular input instead of a sequence --
    gradient-boosted trees, logistic regression, and similar.

    Why this exists: a tree model has no built-in notion of "step 40
    came before step 41" the way a GRU does -- feeding it the raw
    [window, n_feat] block flattened in time order (window * n_feat =
    900 columns at this project's defaults) would ask it to rediscover
    recency and trend from column position alone, which tree splits
    are a poor fit for. Handing it explicit summary statistics per
    feature column instead -- the window's most recent reading, mean,
    standard deviation, min, max, and a simple end-to-start slope --
    gives it the same "what just happened, and what's the recent
    regime" information the GRU infers on its own, in the tabular form
    it actually works well with. This is deliberately simple (no
    per-lag columns, no interaction terms) so a first tree-model
    comparison isn't accidentally handicapped OR flattered by uneven
    feature engineering effort relative to the GRU path -- see
    scripts/backtest_gbm_challenger.py, which uses this.
    """
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    window = X.shape[1]
    slope = (X[:, -1, :] - X[:, 0, :]) / max(1, window - 1)
    return np.concatenate([last, mean, std, mn, mx, slope], axis=1).astype(np.float32)
