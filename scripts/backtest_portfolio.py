#!/usr/bin/env python
"""
Walk-forward, multi-ticker backtest with real performance metrics
(Sharpe ratio, max drawdown, profit factor, win rate, turnover).

Unlike scripts/backtest.py (a quick single-ticker train/val sanity check
with a naive, cost-free simulated P&L), this script runs the ACTUAL
production code path -- the same Orchestrator, risk manager, continual
trainer, and PaperBroker (with its slippage/commission cost model) that
scripts/run_paper_trading.py uses -- across every ticker in config.yaml
simultaneously, sharing one capital pool and one replay buffer exactly
like live trading would. "Walk-forward" here isn't a separate mode to
opt into: the online/continual-learning loop IS walk-forward by
construction (predict on each bar using only data up to that bar, then
learn from it once its outcome matures), so running the real pipeline
over historical data already is the walk-forward backtest -- there is no
separate, parallel backtest-only code path to drift out of sync with
what actually trades.

    python scripts/backtest_portfolio.py                      # synthetic, all 12 tickers
    python scripts/backtest_portfolio.py --feed yfinance       # real recent history
    python scripts/backtest_portfolio.py --max-bars 2000       # shorter run
    python scripts/backtest_portfolio.py --periods-per-year 252  # for daily-bar annualization

Metrics are computed by src/analysis/metrics.py (see that module and
tests/test_metrics.py for the exact formulas and hand-computed checks).
Costs are whatever config.yaml's `broker:` section says (PaperBroker's
slippage_bps/commission_bps) -- this is NOT a costless backtest.

Writes its own decision log (default: logs/backtest_decisions.jsonl) so
repeated runs don't interleave with or overwrite a live/paper-trading
session's logs/decisions.jsonl. Does not persist risk/drift state across
runs (risk_state_path/drift_state_path are left at their default of
None) -- every invocation starts from a clean kill-switch/drift state,
which is what you want when comparing backtest runs against each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.metrics import summarize
from src.config import load_config
from src.data.feed import SyntheticFeed, YFinanceFeed
from src.execution.broker import PaperBroker
from src.model.network import TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskManager
from src.training.drift import DriftMonitor
from src.training.trainer import ContinualTrainer
from src.utils import seed_everything


def install_kill_switch_auto_reset(orch: Orchestrator) -> list[dict]:
    """BACKTEST-ANALYSIS ONLY -- never call this from run_paper_trading.py
    or run_live_alpaca.py. RiskManager.reset_kill_switch() deliberately
    requires human_confirmed=True and is never invoked automatically by
    the trading loop (see its docstring) -- that is a deliberate,
    safety-critical design choice for live/paper trading, and this
    function does not change that code at all.

    What it's for: DriftMonitor's hit-rate/brier/drawdown halt can
    legitimately trip during a model's early cold-start phase (before it
    has had a chance to learn anything -- see the "real bug" sections of
    the README for how often an undertrained model's retrains get
    vetoed) -- and once tripped, size_order() always returns "kill switch
    engaged" for the rest of the run, since nothing resets it. In a
    multi-thousand-bar backtest with no human present to click reset,
    that means a single bad early patch can make the rest of a long
    backtest report "0 trades" and tell you nothing about how the
    strategy performs once the model is actually trained. This monkey-
    patches `orch.risk_manager.trip_kill_switch` (objects this script
    constructs itself -- it never reaches into Orchestrator or
    RiskManager's source) so that every trip is recorded (for the
    report, with the bar it happened on -- read from orch.equity_curve,
    which Orchestrator.run() always appends to BEFORE checking
    should_halt() on the same bar) and then immediately reversed with
    `reset_kill_switch(human_confirmed=True)`, so the backtest keeps
    running and keeps accumulating real trade/equity history past the
    trip. Returns the list this records trips into, each as
    {"bar": int, "reason": str}.

    Only wired up behind --ignore-kill-switch, which defaults to False --
    the default backtest run reports the kill switch exactly as it would
    behave live (fails closed, stays closed), which is itself useful
    information (see the README)."""
    trips: list[dict] = []
    risk_manager = orch.risk_manager
    original_trip = risk_manager.trip_kill_switch

    def _trip_and_auto_reset(reason: str) -> None:
        trips.append({"bar": len(orch.equity_curve), "reason": reason})
        original_trip(reason)
        risk_manager.reset_kill_switch(human_confirmed=True)

    risk_manager.trip_kill_switch = _trip_and_auto_reset
    return trips


def collapse_trip_ranges(trips: list[dict]) -> list[dict]:
    """Group consecutive-bar kill-switch trips into contiguous ranges for
    reporting. A "trip storm" (the halt condition persisting bar after
    bar once it fires -- see install_kill_switch_auto_reset's docstring)
    can be 1000+ individual trips long; printing one line per trip would
    make the report unreadable. `trips` must be in the order they
    occurred (bar numbers non-decreasing) -- exactly what
    install_kill_switch_auto_reset's returned list already is.
    `collapse_trip_ranges([{"bar": 5, "reason": "a"}, {"bar": 6,
    "reason": "b"}, {"bar": 9, "reason": "c"}])` ==
    `[{"start_bar": 5, "end_bar": 6, "start_reason": "a", "end_reason":
    "b", "n_trips": 2}, {"start_bar": 9, "end_bar": 9, "start_reason":
    "c", "end_reason": "c", "n_trips": 1}]`."""
    ranges: list[dict] = []
    for t in trips:
        if ranges and t["bar"] == ranges[-1]["end_bar"] + 1:
            ranges[-1]["end_bar"] = t["bar"]
            ranges[-1]["end_reason"] = t["reason"]
            ranges[-1]["n_trips"] += 1
        else:
            ranges.append({"start_bar": t["bar"], "end_bar": t["bar"],
                            "start_reason": t["reason"], "end_reason": t["reason"],
                            "n_trips": 1})
    return ranges


def _per_ticker_breakdown(orch: Orchestrator) -> dict[str, dict]:
    """Slice orch.trade_log by ticker and run the same metrics on each
    slice. Turnover is omitted per-ticker (it's a portfolio-level
    capital-utilization figure; n_bars is shared across all tickers, not
    per-ticker, so a per-ticker turnover number computed the same way
    would double-count the shared denominator)."""
    out: dict[str, dict] = {}
    for ticker in orch.tickers:
        pnls = [t["pnl_pct_of_equity"] for t in orch.trade_log if t["ticker"] == ticker]
        from src.analysis.metrics import profit_factor, win_rate
        out[ticker] = {
            "n_trades": len(pnls),
            "win_rate": win_rate(pnls),
            "profit_factor": profit_factor(pnls),
            "total_pnl_pct_equity": sum(pnls),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--feed", default=None, choices=["synthetic", "yfinance"])
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--periods-per-year", type=float, default=252.0,
                         help="Annualization factor for Sharpe ratio -- must match the bar "
                              "frequency. 252 for daily bars (yfinance_interval: 1d). For "
                              "intraday bars, scale by bars-per-trading-day too, e.g. "
                              "15-minute bars over a 6.5h session: 252 * 26 =~ 6552.")
    parser.add_argument("--log-path", default="logs/backtest_decisions.jsonl")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seeds Python/numpy/torch global RNG (model weight init, dropout) "
                              "and the continual trainer's own retrain RNG, so re-running the "
                              "same backtest twice gives IDENTICAL results -- see "
                              "src/utils.py:seed_everything's docstring for why this was needed "
                              "(before it, results varied run-to-run for reasons having nothing "
                              "to do with the strategy). Pass a different value to compare "
                              "multiple random initializations instead of trusting just one.")
    parser.add_argument("--ignore-kill-switch", action="store_true",
                         help="BACKTEST ANALYSIS ONLY -- never use this flag's underlying "
                              "mechanism outside this script. Auto-reverses every kill-switch "
                              "trip (human_confirmed=True) so the backtest keeps running/trading "
                              "past it instead of reporting near-zero trades for the rest of the "
                              "run because of one early cold-start rough patch. Off by default: "
                              "without this flag, the kill switch behaves exactly as it would "
                              "live (fails closed, stays closed until a human resets it). See "
                              "install_kill_switch_auto_reset()'s docstring and the README.")
    args = parser.parse_args()

    seed_everything(args.seed)

    cfg = load_config(args.config)
    tickers = cfg["tickers"]
    feed_kind = args.feed or cfg["data"]["feed"]

    if feed_kind == "synthetic":
        feed = SyntheticFeed(tickers, n_bars=cfg["data"]["n_bars"])
    else:
        feed = YFinanceFeed(tickers, period=cfg["data"]["yfinance_period"],
                             interval=cfg["data"]["yfinance_interval"])

    cfg["_trainer_cfg"].seed = args.seed
    model = TradingNet(cfg["_model_cfg"])
    trainer = ContinualTrainer(model, cfg["_trainer_cfg"])
    risk_manager = RiskManager(cfg["_risk_cfg"])
    broker = PaperBroker(starting_cash=cfg["broker"]["starting_cash"],
                          slippage_bps=cfg["broker"]["slippage_bps"],
                          commission_bps=cfg["broker"]["commission_bps"])
    logger = DecisionLogger(args.log_path)
    drift_monitor = DriftMonitor(cfg["_drift_cfg"])

    orch = Orchestrator(
        tickers=tickers, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=cfg["_model_cfg"].window,
        label_cfg=cfg["_label_cfg"],
    )

    kill_switch_trips: list[dict] = []
    if args.ignore_kill_switch:
        kill_switch_trips = install_kill_switch_auto_reset(orch)
        print("--ignore-kill-switch is ON: kill-switch trips will be auto-reversed for this "
              "backtest run only (never happens in run_paper_trading.py/run_live_alpaca.py). "
              "See install_kill_switch_auto_reset()'s docstring.")

    print(f"Running walk-forward backtest: feed={feed_kind} tickers={tickers} "
          f"max_bars={args.max_bars or 'ALL'} seed={args.seed} "
          f"costs(slippage_bps={cfg['broker']['slippage_bps']}, "
          f"commission_bps={cfg['broker']['commission_bps']})")

    orch.run(max_bars=args.max_bars)

    n_bars = len(orch.equity_curve)
    trade_pnls = [t["pnl_pct_of_equity"] for t in orch.trade_log]
    trade_sizes = [t["size_pct_equity"] for t in orch.trade_log]
    metrics = summarize(orch.equity_curve, trade_pnls, trade_sizes, n_bars,
                         periods_per_year=args.periods_per_year)
    per_ticker = _per_ticker_breakdown(orch)

    final_equity = orch.equity_curve[-1] if orch.equity_curve else broker.starting_cash
    total_return_pct = (final_equity / broker.starting_cash - 1.0) * 100.0

    print("\n=== Portfolio summary ===")
    print(f"Bars processed:        {n_bars}")
    print(f"Starting equity:       {broker.starting_cash:,.2f}")
    print(f"Final equity:          {final_equity:,.2f}")
    print(f"Total return:          {total_return_pct:+.2f}%")
    print(f"Sharpe ratio:          {metrics['sharpe_ratio']:.3f}  (annualized, periods_per_year={args.periods_per_year:.0f})")
    print(f"Max drawdown:          {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Profit factor:         {metrics['profit_factor']:.3f}" if metrics['profit_factor'] != float('inf')
          else "Profit factor:         inf (no losing trades)")
    print(f"Win rate:              {metrics['win_rate'] * 100:.1f}%")
    print(f"Turnover:              {metrics['turnover']:.3f}% of equity/bar (avg)")
    print(f"Total realized trades: {metrics['n_trades']}")
    print(f"Retrain events:        {len(orch.retrain_events)} "
          f"(promoted: {sum(1 for r in orch.retrain_events if r['promoted'])})")
    print(f"Kill switch engaged:   {risk_manager.kill_switch_engaged()} "
          f"{risk_manager.state.halt_reasons if risk_manager.kill_switch_engaged() else ''}")
    if kill_switch_trips:
        ranges = collapse_trip_ranges(kill_switch_trips)
        print(f"\nKill switch tripped {len(kill_switch_trips)}x across {len(ranges)} episode(s) "
              f"during this run (auto-reversed because --ignore-kill-switch was passed -- "
              f"without it, the FIRST episode below would have ended real trading for the rest "
              f"of the run):")
        MAX_SHOWN = 10
        for r in ranges[:MAX_SHOWN]:
            span = f"bar {r['start_bar']}" if r["start_bar"] == r["end_bar"] else f"bars {r['start_bar']}-{r['end_bar']}"
            print(f"  {span} ({r['n_trips']}x): {r['start_reason']}"
                  + (f"  ->  {r['end_reason']}" if r["end_reason"] != r["start_reason"] else ""))
        if len(ranges) > MAX_SHOWN:
            print(f"  ... and {len(ranges) - MAX_SHOWN} more episode(s)")
    elif risk_manager.kill_switch_engaged():
        print("\nNote: the kill switch is engaged at the end of this run, which means "
              "size_order() returned 0 for every bar after it tripped -- trade counts/metrics "
              "above may reflect far fewer bars of real trading than --max-bars suggests. "
              "Re-run with --ignore-kill-switch to see full-period metrics net of this (backtest-"
              "analysis only -- see that flag's help text for why this is never done live).")

    print("\n=== Per-ticker breakdown ===")
    header = f"{'ticker':<8}{'trades':>8}{'win_rate':>12}{'profit_factor':>16}{'total_pnl_%eq':>16}"
    print(header)
    print("-" * len(header))
    for ticker, m in per_ticker.items():
        pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.3f}"
        print(f"{ticker:<8}{m['n_trades']:>8}{m['win_rate'] * 100:>11.1f}%{pf:>16}{m['total_pnl_pct_equity']:>+16.3f}")

    print(f"\nDecision log written to: {args.log_path}")
    print("\nThis is a backtest against historical/synthetic data run through the "
          "real strategy code -- it still cannot account for real-world slippage "
          "beyond PaperBroker's simple model, latency, partial fills, or a real "
          "broker's rejections. A good number here is necessary but not "
          "sufficient before considering real capital; see the README.")


if __name__ == "__main__":
    main()
