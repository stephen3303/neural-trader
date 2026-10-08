"""
Performance metrics for backtests and live/paper trading runs.

Every function here is a small, pure, independently-testable calculation
over plain sequences of numbers -- no dependency on Orchestrator, the
decision log, or any other part of the system. That is deliberate: the
whole point of this module is that `scripts/backtest_portfolio.py` and a
future dashboard summary panel can both compute the exact same numbers
from whatever equity curve / trade list they have on hand, and both can
be checked against hand-computed values in tests/test_metrics.py.

Two input shapes recur throughout:

- `equity_curve`: a time-ordered sequence of total account equity
  (cash + marked positions), one value per bar. Used for Sharpe ratio and
  max drawdown -- both path-dependent (they care about the sequence, not
  just the set of per-trade outcomes).
- `trade_pnls`: a sequence of REALIZED per-trade P&L values, each as a
  fraction of account equity at the time the trade was opened (this
  matches Orchestrator._resolve_matured's `pnl_pct_of_equity`, i.e. the
  same number already fed into RiskManager.update_after_trade_result).
  Used for profit factor and win rate -- both care only about the
  distribution of outcomes, not the order they happened in.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def period_returns(equity_curve: Sequence[float]) -> np.ndarray:
    """Simple (not log) bar-over-bar returns from an equity curve.
    `period_returns([100, 110, 99]) == [0.10, -0.10]`. Needs at least 2
    equity points to produce one return; returns an empty array otherwise.
    A non-positive equity value would make the following return
    undefined (division by <= 0), so any such entries are skipped rather
    than producing inf/nan -- a backtest that actually blows the account
    to zero or negative is already reported as a 100% max_drawdown by
    that function, so this function failing loudly isn't needed too."""
    arr = np.asarray(equity_curve, dtype=np.float64)
    if len(arr) < 2:
        return np.array([], dtype=np.float64)
    prev = arr[:-1]
    curr = arr[1:]
    valid = prev > 0
    out = np.full(len(prev), np.nan)
    out[valid] = (curr[valid] - prev[valid]) / prev[valid]
    return out[~np.isnan(out)]


def sharpe_ratio(equity_curve: Sequence[float], periods_per_year: float = 252.0,
                  risk_free_rate_per_period: float = 0.0) -> float:
    """Annualized Sharpe ratio computed from an equity curve's per-bar
    returns: mean(excess_return) / std(excess_return) * sqrt(periods_per_year).

    `periods_per_year` must match the bar frequency the equity curve was
    sampled at -- 252 for daily bars, ~252*26 for 15-minute bars during a
    6.5h trading day, etc. (see scripts/backtest_portfolio.py's
    --periods-per-year flag). Returns 0.0 if there isn't enough data (< 2
    returns) or the returns have zero variance (e.g. a flat equity
    curve -- no trades were ever sized), rather than raising or returning
    nan/inf, since "not enough signal yet" is a far more common case here
    than an actual division-by-zero bug.

    Uses sample standard deviation (ddof=1, Bessel's correction), the
    conventional choice for a Sharpe ratio estimated from a finite sample
    rather than a known population.
    """
    rets = period_returns(equity_curve)
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free_rate_per_period
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough decline in the equity curve, as a fraction
    of the peak (0.0 = never drew down at all, 1.0 = wiped out entirely).
    `max_drawdown([100, 120, 90, 150, 75]) == 0.5` (peak 150 -> trough 75).
    Returns 0.0 for an empty or single-point curve. If equity ever
    reaches zero or below, that bar's drawdown is capped at 1.0 (can't
    lose more than 100% of a peak) instead of dividing by a non-positive
    peak."""
    arr = np.asarray(equity_curve, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_peak > 0, (running_peak - arr) / running_peak, 1.0)
    dd = np.clip(dd, 0.0, 1.0)
    return float(np.max(dd))


def profit_factor(trade_pnls: Sequence[float]) -> float:
    """sum(winning trade P&L) / abs(sum(losing trade P&L)). > 1.0 means
    winners outweighed losers in aggregate dollar (or %-equity) terms,
    regardless of how many of each there were. `profit_factor([1, 1, -1])
    == 2.0`. If there are no losing trades: `inf` when there's at least
    one winner (no losses to weigh against), `0.0` if there are no
    trades at all or every trade was exactly breakeven (nothing to call
    a "factor" of)."""
    arr = np.asarray(trade_pnls, dtype=np.float64)
    gains = float(arr[arr > 0].sum()) if len(arr) else 0.0
    losses = float(-arr[arr < 0].sum()) if len(arr) else 0.0
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def win_rate(trade_pnls: Sequence[float]) -> float:
    """Fraction of trades with strictly positive P&L (an exact-breakeven
    trade counts as a non-win, same convention as profit_factor treating
    it as neither a gain nor a loss). `win_rate([1, -1, 1, 0]) == 0.5`.
    Returns 0.0 for an empty sequence rather than nan."""
    arr = np.asarray(trade_pnls, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr > 0))


def turnover(trade_sizes_pct_equity: Sequence[float], n_bars: int) -> float:
    """Average per-bar turnover: total traded notional (as a % of
    account equity, summed across all trades' absolute size) divided by
    the number of bars the backtest ran for. `turnover([5, 5, 10], 100)
    == 0.2` -- i.e. on average 0.2% of equity's worth of notional was
    traded per bar over that run. Returns 0.0 if n_bars <= 0 rather than
    dividing by zero."""
    if n_bars <= 0:
        return 0.0
    total = float(np.sum(np.abs(np.asarray(trade_sizes_pct_equity, dtype=np.float64)))) if len(trade_sizes_pct_equity) else 0.0
    return total / n_bars


def summarize(equity_curve: Sequence[float], trade_pnls: Sequence[float],
              trade_sizes_pct_equity: Sequence[float], n_bars: int,
              periods_per_year: float = 252.0) -> dict:
    """Convenience bundle of every metric above, in the shape
    scripts/backtest_portfolio.py prints and a future dashboard summary
    panel could consume directly."""
    return {
        "sharpe_ratio": sharpe_ratio(equity_curve, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "profit_factor": profit_factor(trade_pnls),
        "win_rate": win_rate(trade_pnls),
        "turnover": turnover(trade_sizes_pct_equity, n_bars),
        "n_trades": int(len(trade_pnls)),
    }
