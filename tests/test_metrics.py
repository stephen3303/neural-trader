import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.analysis.metrics import (
    max_drawdown,
    period_returns,
    profit_factor,
    sharpe_ratio,
    summarize,
    turnover,
    win_rate,
)


class TestPeriodReturns:
    def test_simple_returns_hand_computed(self):
        # 100 -> 110 is +10%, 110 -> 99 is -10% (99/110 == 0.9 exactly)
        rets = period_returns([100, 110, 99])
        assert rets.tolist() == pytest.approx([0.10, -0.10])

    def test_fewer_than_two_points_is_empty(self):
        assert period_returns([]).tolist() == []
        assert period_returns([100]).tolist() == []

    def test_skips_a_non_positive_prior_equity_without_crashing(self):
        # Equity hitting 0 would make the NEXT return undefined (division
        # by zero) -- that single return should be dropped, not raise or
        # produce inf/nan, while the surrounding valid returns survive.
        rets = period_returns([100, 0, 50])
        assert rets.tolist() == pytest.approx([-1.0])  # only the 100->0 leg


class TestSharpeRatio:
    def test_zero_variance_returns_is_zero_not_inf(self):
        # A perfectly flat equity curve (e.g. no trades ever sized) has
        # returns = [0.0]*4 exactly in floating point -- zero variance,
        # which must be defined as 0.0, not inf/nan from a 0/0 division.
        assert sharpe_ratio([100, 100, 100, 100, 100]) == 0.0

    def test_hand_computed_with_varying_returns(self):
        # Equity: 100, 110, 99, 108.9 -> returns: +0.10, -0.10, +0.10
        # mean = 0.033333..., sample stdev (ddof=1) of [0.1,-0.1,0.1]:
        #   deviations from mean: [0.06667, -0.13333, 0.06667]
        #   sum of squares = 0.0044444 + 0.0177778 + 0.0044444 = 0.0266667
        #   variance = 0.0266667 / 2 = 0.0133333 -> stdev = 0.115470
        # sharpe (periods_per_year=1, no annualization) = 0.033333/0.115470
        equity = [100, 110, 99, 108.9]
        got = sharpe_ratio(equity, periods_per_year=1.0)
        expected_mean = (0.10 + -0.10 + 0.10) / 3
        expected_std = math.sqrt(((0.10 - expected_mean) ** 2 + (-0.10 - expected_mean) ** 2 +
                                   (0.10 - expected_mean) ** 2) / 2)
        expected = expected_mean / expected_std
        assert got == pytest.approx(expected, rel=1e-6)

    def test_annualization_scales_by_sqrt_of_periods(self):
        equity = [100, 110, 99, 108.9]
        daily = sharpe_ratio(equity, periods_per_year=1.0)
        annualized = sharpe_ratio(equity, periods_per_year=252.0)
        assert annualized == pytest.approx(daily * math.sqrt(252.0), rel=1e-6)

    def test_too_few_points_is_zero_not_an_error(self):
        assert sharpe_ratio([]) == 0.0
        assert sharpe_ratio([100]) == 0.0
        assert sharpe_ratio([100, 110]) == 0.0  # only 1 return -- stdev undefined


class TestMaxDrawdown:
    def test_hand_computed_peak_to_trough(self):
        # peak 150 -> trough 75 is the largest decline = 50%
        assert max_drawdown([100, 120, 90, 150, 75]) == pytest.approx(0.5)

    def test_monotonically_rising_curve_has_zero_drawdown(self):
        assert max_drawdown([100, 110, 120, 130]) == pytest.approx(0.0)

    def test_empty_curve_is_zero(self):
        assert max_drawdown([]) == 0.0

    def test_wipeout_is_capped_at_one(self):
        assert max_drawdown([100, 50, 0]) == pytest.approx(1.0)

    def test_single_point_is_zero(self):
        assert max_drawdown([100]) == 0.0


class TestProfitFactor:
    def test_hand_computed_mixed_trades(self):
        # gains = 1+1 = 2, losses = |-1| = 1 -> profit factor 2.0
        assert profit_factor([1.0, 1.0, -1.0]) == pytest.approx(2.0)

    def test_no_losses_with_a_winner_is_infinite(self):
        assert profit_factor([1.0, 2.0]) == float("inf")

    def test_no_trades_is_zero(self):
        assert profit_factor([]) == 0.0

    def test_all_breakeven_is_zero(self):
        assert profit_factor([0.0, 0.0]) == 0.0

    def test_all_losses_is_zero(self):
        assert profit_factor([-1.0, -2.0]) == pytest.approx(0.0)


class TestWinRate:
    def test_hand_computed(self):
        assert win_rate([1.0, -1.0, 1.0, 0.0]) == pytest.approx(0.5)

    def test_empty_is_zero(self):
        assert win_rate([]) == 0.0

    def test_all_wins_is_one(self):
        assert win_rate([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_breakeven_does_not_count_as_a_win(self):
        assert win_rate([0.0]) == 0.0


class TestTurnover:
    def test_hand_computed(self):
        # total traded = 5+5+10 = 20, over 100 bars -> 0.2 per bar
        assert turnover([5.0, 5.0, 10.0], 100) == pytest.approx(0.2)

    def test_uses_absolute_value_of_each_trade_size(self):
        assert turnover([-5.0, 5.0], 10) == pytest.approx(1.0)

    def test_zero_bars_is_zero_not_a_division_error(self):
        assert turnover([5.0], 0) == 0.0

    def test_no_trades_is_zero(self):
        assert turnover([], 100) == 0.0


class TestSummarize:
    def test_bundles_every_metric_with_matching_keys(self):
        out = summarize(
            equity_curve=[100, 110, 99, 108.9],
            trade_pnls=[1.0, 1.0, -1.0],
            trade_sizes_pct_equity=[5.0, 5.0, 10.0],
            n_bars=100,
            periods_per_year=1.0,
        )
        assert out["sharpe_ratio"] == pytest.approx(sharpe_ratio([100, 110, 99, 108.9], periods_per_year=1.0))
        assert out["max_drawdown"] == pytest.approx(max_drawdown([100, 110, 99, 108.9]))
        assert out["profit_factor"] == pytest.approx(2.0)
        assert out["win_rate"] == pytest.approx(2 / 3)
        assert out["turnover"] == pytest.approx(0.2)
        assert out["n_trades"] == 3
