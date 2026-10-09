import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.data.features import FEATURE_COLUMNS, summarize_window


def _window(values_per_feature):
    """Build a [1, window, n_feat] array where every feature column
    ramps linearly through the given (start, end) pair across the
    window, so mean/min/max/slope are easy to hand-compute."""
    window = 4
    n_feat = len(values_per_feature)
    X = np.zeros((1, window, n_feat), dtype=np.float32)
    for f, (start, end) in enumerate(values_per_feature):
        X[0, :, f] = np.linspace(start, end, window)
    return X


class TestSummarizeWindow:
    def test_output_shape_is_six_stats_per_feature_column(self):
        n, window, n_feat = 3, 20, len(FEATURE_COLUMNS)
        X = np.random.default_rng(0).normal(size=(n, window, n_feat)).astype(np.float32)
        out = summarize_window(X)
        assert out.shape == (n, n_feat * 6)
        assert out.dtype == np.float32

    def test_last_mean_min_max_match_a_hand_computed_ramp(self):
        # One feature column ramping 0 -> 3 over 4 steps: [0, 1, 2, 3].
        X = _window([(0.0, 3.0)])
        out = summarize_window(X)
        last, mean, std, mn, mx, slope = out[0]
        assert last == pytest.approx(3.0)       # most recent value
        assert mean == pytest.approx(1.5)        # mean of [0,1,2,3]
        assert mn == pytest.approx(0.0)
        assert mx == pytest.approx(3.0)
        assert std == pytest.approx(np.std([0.0, 1.0, 2.0, 3.0]))
        assert slope == pytest.approx(1.0)       # (3 - 0) / (4 - 1)

    def test_slope_is_negative_for_a_declining_feature(self):
        X = _window([(10.0, -2.0)])
        out = summarize_window(X)
        slope = out[0, 5]
        assert slope < 0

    def test_a_flat_constant_feature_has_zero_std_and_zero_slope(self):
        X = _window([(5.0, 5.0)])
        out = summarize_window(X)
        last, mean, std, mn, mx, slope = out[0]
        assert (last, mean, mn, mx) == (5.0, 5.0, 5.0, 5.0)
        assert std == pytest.approx(0.0)
        assert slope == pytest.approx(0.0)

    def test_summarizes_every_sample_independently(self):
        X = np.concatenate([_window([(0.0, 3.0)]), _window([(10.0, -2.0)])], axis=0)
        out = summarize_window(X)
        assert out.shape == (2, 6)
        assert out[0, 0] == pytest.approx(3.0)
        assert out[1, 0] == pytest.approx(-2.0)
