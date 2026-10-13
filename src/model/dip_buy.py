"""
Walk-forward "dip buy" entry model -- a BACKTEST-ONLY alternative to the
GRU's entry signals, for comparing entries on real data. Nothing in the
live bot imports this.

Why it exists: on real 2024-2026 bars the GRU's buy/sell calls carried
almost no information (live: 51.7% directionally right, confidence
unrelated to outcome), and no filter on top of them helped reliably (see
research/entry-research-results.md in the project files). A gradient-
boosted model trained on 12 months of history, taking only its most
confident longs, held up better. Those turned out to be dip buys after
multi-hour sell-offs.

How it works, all causal:
- Features: the bot's own FEATURE_COLUMNS plus longer returns (30/60/130
  bars), time of day, move since the open, the opening gap, the market's
  (SPY's) recent return and a volatility ratio.
- Label: the forward return over the label horizon (clipped to +/-3%).
- Each calendar month of the test range gets a fresh model trained on the
  `train_months` before it, with a one-day gap so no label overlaps the
  test month.
- Entry: long only, when the prediction is in the top `top_frac` of that
  model's own predictions on its training window. Shorts are never taken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data.features import FEATURE_COLUMNS, compute_features

EXTRA_COLUMNS = ["ret_30", "ret_60", "ret_130", "tod", "from_open", "gap",
                 "mkt_ret_15", "mkt_ret_60", "rv_ratio"]
ENTRY_FEATURES = FEATURE_COLUMNS + EXTRA_COLUMNS


def entry_features(series: dict[str, pd.DataFrame], market: str = "SPY",
                   horizon: int = 15) -> dict[str, pd.DataFrame]:
    """Per-ticker frame of ENTRY_FEATURES plus `fwd` (forward return over
    `horizon` bars, NaN where it isn't known yet). Bars must be indexed by
    tz-aware New York timestamps (as CSVFeed/AlpacaHistoricalFeed give)."""
    mkt_close = series[market]["close"] if market in series else None
    out = {}
    for t, bars in series.items():
        df = compute_features(bars)
        close = bars["close"]
        date = pd.Series(bars.index.date, index=bars.index)
        df["ret_30"] = close.pct_change(30)
        df["ret_60"] = close.pct_change(60)
        df["ret_130"] = close.pct_change(130)
        df["tod"] = bars.index.hour * 60 + bars.index.minute
        day_open = bars["open"].groupby(date).transform("first")
        df["from_open"] = close / day_open - 1
        prev_close = close.groupby(date).last().shift(1)
        df["gap"] = day_open / date.map(prev_close) - 1
        if mkt_close is not None:
            df["mkt_ret_15"] = mkt_close.pct_change(15).reindex(bars.index).values
            df["mkt_ret_60"] = mkt_close.pct_change(60).reindex(bars.index).values
        else:
            df["mkt_ret_15"] = np.nan
            df["mkt_ret_60"] = np.nan
        rv = df["realized_vol_15"]
        df["rv_ratio"] = rv / rv.rolling(130).mean()
        df["fwd"] = close.shift(-horizon) / close - 1
        out[t] = df[ENTRY_FEATURES + ["fwd"]]
    return out


def _stack(feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for t, df in feats.items():
        df = df.copy()
        df["ticker"] = t
        frames.append(df)
    return pd.concat(frames).sort_index()


def walk_forward_signals(train_series: dict[str, pd.DataFrame],
                         test_series: dict[str, pd.DataFrame],
                         start: str, end: str, train_months: int = 12,
                         top_frac: float = 0.10, horizon: int = 15,
                         seed: int = 0) -> pd.DataFrame:
    """Predictions for every test bar in [start, end), one model per month.

    `train_series` supplies the training history (it may be the same data
    as `test_series`, or a longer history from another source, e.g. Alpaca
    history for a Yahoo test window). Returns a frame indexed by
    (ticker, timestamp) with columns `pred` and `enter` (bool)."""
    tz = "America/New_York"
    train = _stack(entry_features(train_series, horizon=horizon))
    test = _stack(entry_features(test_series, horizon=horizon))
    train = train.dropna(subset=ENTRY_FEATURES[:1] + ["fwd"])
    rows = []
    for month in pd.period_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="M"):
        m_start = max(month.start_time.tz_localize(tz), pd.Timestamp(start, tz=tz))
        m_end = min((month + 1).start_time.tz_localize(tz), pd.Timestamp(end, tz=tz))
        t_start = month.start_time.tz_localize(tz) - pd.DateOffset(months=train_months)
        fit = train[(train.index >= t_start) & (train.index < m_start - pd.Timedelta(days=1))]
        tst = test[(test.index >= m_start) & (test.index < m_end)]
        if len(tst) == 0:
            continue
        if len(fit) < 1000:
            raise ValueError(f"only {len(fit)} training rows before {m_start:%Y-%m-%d}; "
                             f"the training data needs ~{train_months} months of history before the test range")
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              random_state=seed)
        model.fit(fit[ENTRY_FEATURES], fit["fwd"].clip(-0.03, 0.03))
        threshold = np.quantile(model.predict(fit[ENTRY_FEATURES]), 1 - top_frac)
        pred = model.predict(tst[ENTRY_FEATURES])
        rows.append(pd.DataFrame({"ticker": tst["ticker"].values, "timestamp": tst.index,
                                  "pred": pred, "enter": (pred >= threshold) & (pred > 0)}))
    if not rows:
        return pd.DataFrame(columns=["pred", "enter"])
    return pd.concat(rows).set_index(["ticker", "timestamp"])


class DipBuySignals:
    """Looks up precomputed walk-forward signals for the orchestrator's
    `signal_override` hook: BUY when the model says enter, HOLD otherwise."""

    def __init__(self, signals: pd.DataFrame, confidence: float = 0.75):
        self._enter = {k: (bool(e), float(p)) for k, e, p in
                       zip(signals.index, signals["enter"], signals["pred"])}
        self.confidence = confidence

    def __call__(self, ticker: str, timestamp) -> dict:
        enter, pred = self._enter.get((ticker, pd.Timestamp(timestamp)), (False, 0.0))
        if not enter:
            return {"action": 1, "confidence": 0.0, "expected_return": pred}
        return {"action": 2, "confidence": self.confidence, "expected_return": pred}
