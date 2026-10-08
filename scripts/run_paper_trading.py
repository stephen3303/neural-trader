#!/usr/bin/env python
"""
Run the full pipeline end-to-end in paper-trading mode against either the
synthetic feed (default, no network needed) or real historical data via
yfinance.

    python scripts/run_paper_trading.py                  # synthetic, default config
    python scripts/run_paper_trading.py --feed yfinance   # real recent history
    python scripts/run_paper_trading.py --max-bars 500    # shorter smoke run

This is PAPER trading -- a PaperBroker is always used here. See
src/execution/broker.py and the README before ever pointing any of this at
a funded brokerage account.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.feed import SyntheticFeed, YFinanceFeed
from src.execution.broker import PaperBroker
from src.model.network import TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskManager
from src.training.drift import DriftMonitor
from src.training.trainer import ContinualTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--feed", default=None, choices=["synthetic", "yfinance"])
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--status-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true",
                         help="Load the latest checkpoint from the configured checkpoint_dir "
                              "before running, instead of starting from an untrained model. "
                              "Off by default here so repeated smoke-test runs stay "
                              "reproducible; run_live_alpaca.py always resumes.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tickers = cfg["tickers"]
    feed_kind = args.feed or cfg["data"]["feed"]

    if feed_kind == "synthetic":
        feed = SyntheticFeed(tickers, n_bars=cfg["data"]["n_bars"])
    else:
        feed = YFinanceFeed(tickers, period=cfg["data"]["yfinance_period"],
                             interval=cfg["data"]["yfinance_interval"])

    model = TradingNet(cfg["_model_cfg"])
    trainer = ContinualTrainer(model, cfg["_trainer_cfg"])
    risk_manager = RiskManager(cfg["_risk_cfg"])
    broker = PaperBroker(starting_cash=cfg["broker"]["starting_cash"],
                          slippage_bps=cfg["broker"]["slippage_bps"],
                          commission_bps=cfg["broker"]["commission_bps"])
    logger = DecisionLogger(cfg["logging"]["log_path"])
    drift_monitor = DriftMonitor(cfg["_drift_cfg"])

    if args.resume:
        resumed_version = trainer.load_latest_checkpoint()
        if resumed_version is not None:
            print(f"Resumed model weights from checkpoint: v{resumed_version} "
                  f"(from {cfg['_trainer_cfg'].checkpoint_dir}/)")
        else:
            print(f"--resume given but no checkpoint found in "
                  f"{cfg['_trainer_cfg'].checkpoint_dir}/ -- starting from an untrained model.")

    orch = Orchestrator(
        tickers=tickers, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=cfg["_model_cfg"].window,
        label_cfg=cfg["_label_cfg"],
    )

    print(f"Running paper trading: feed={feed_kind} tickers={tickers} "
          f"max_bars={args.max_bars or 'ALL'}")

    orch.run(max_bars=args.max_bars)

    print("\n=== Final status ===")
    print("Equity:", round(broker.get_equity({t: 0 for t in tickers}), 2),
          " (started at", broker.starting_cash, ")")
    print("Fills executed:", len(broker.fills))
    print("Retrain events:", len(orch.retrain_events),
          "| promoted:", sum(1 for r in orch.retrain_events if r["promoted"]))
    print("Drift snapshot:", drift_monitor.snapshot())
    print("Kill switch engaged:", risk_manager.kill_switch_engaged(),
          risk_manager.state.halt_reasons if risk_manager.kill_switch_engaged() else "")
    print(f"Decision log written to: {cfg['logging']['log_path']}")


if __name__ == "__main__":
    main()
