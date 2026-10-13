import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.model.dip_buy import DipBuySignals, ENTRY_FEATURES, entry_features, walk_forward_signals


def _bars(n_days=330, seed=0, tickers=("SPY", "AAA")):
    """Random-walk 15m regular-session bars, New York time."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n_days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30 + 15 * k) for d in days for k in range(26)],
                           tz="America/New_York")
    out = {}
    for t in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
        out[t] = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                               "close": close, "volume": rng.integers(1000, 5000, len(idx)).astype(float)},
                              index=idx)
    return out


def test_entry_features_are_causal():
    bars = _bars(n_days=40)
    full = entry_features(bars)["AAA"]
    cut = {t: b.iloc[:600] for t, b in bars.items()}
    part = entry_features(cut)["AAA"]
    # Features up to bar 599 don't change when later bars are added.
    pd.testing.assert_frame_equal(full[ENTRY_FEATURES].iloc[:600], part[ENTRY_FEATURES], check_freq=False)


def test_walk_forward_only_enters_long_and_covers_test_range():
    bars = _bars()
    sig = walk_forward_signals(bars, bars, "2025-01-01", "2025-03-01", top_frac=0.1)
    ts = sig.index.get_level_values("timestamp")
    assert ts.min() >= pd.Timestamp("2025-01-01", tz="America/New_York")
    assert ts.max() < pd.Timestamp("2025-03-01", tz="America/New_York")
    assert sig["enter"].any()
    assert (sig.loc[sig["enter"], "pred"] > 0).all()


def test_walk_forward_needs_training_history():
    bars = _bars(n_days=30)
    with pytest.raises(ValueError, match="training rows"):
        walk_forward_signals(bars, bars, "2024-01-20", "2024-02-01")


def test_signals_map_to_buy_or_hold():
    ts = pd.Timestamp("2025-01-02 10:00", tz="America/New_York")
    sig = pd.DataFrame({"pred": [0.004, 0.001], "enter": [True, False]},
                       index=pd.MultiIndex.from_tuples([("AAA", ts), ("BBB", ts)], names=["ticker", "timestamp"]))
    lookup = DipBuySignals(sig)
    assert lookup("AAA", ts)["action"] == 2
    assert lookup("BBB", ts)["action"] == 1
    assert lookup("CCC", ts)["action"] == 1


def test_no_training_label_reaches_into_the_test_month(monkeypatch):
    """2024-07-01 is a Monday: the Friday before's late bars have 15-bar
    labels that end on that Monday, so they must not be trained on."""
    import src.model.dip_buy as dip_buy
    fitted = []
    real = dip_buy.HistGradientBoostingRegressor

    class Spy(real):
        def fit(self, X, y):
            fitted.append(X.index.max())
            return super().fit(X, y)

    monkeypatch.setattr(dip_buy, "HistGradientBoostingRegressor", Spy)
    bars = _bars(n_days=200)
    m_start = pd.Timestamp("2024-07-01", tz="America/New_York")
    feats = entry_features(bars)["AAA"]
    walk_forward_signals(bars, bars, "2024-07-01", "2024-07-10", train_months=6)
    last = fitted[0]
    assert feats.loc[last, "fwd_end"] < m_start
    # ...and it's the last such bar: the next one's label would reach the test month.
    nxt = feats.index[feats.index.get_loc(last) + 1]
    assert feats.loc[nxt, "fwd_end"] >= m_start
