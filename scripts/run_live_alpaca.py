#!/usr/bin/env python
"""
Run the continual-learning loop against REAL-TIME Alpaca market data, with
ALL ORDERS STILL GOING THROUGH ALPACA'S PAPER ACCOUNT. No real money moves.
This is the "live data, still paper trading" stage described in the
README -- the step before ever considering a funded account.

Setup:
    pip install alpaca-py python-dotenv   # python-dotenv is optional

    export ALPACA_API_KEY="your-paper-key-id"
    export ALPACA_SECRET_KEY="your-paper-secret-key"
    # (or put them in a .env file next to this project -- see .env.example)

Run:
    python scripts/run_live_alpaca.py

What this does differently from run_paper_trading.py:
    - Market data is AlpacaLiveFeed: a one-time historical warm-up via REST,
      then real-time minute bars over Alpaca's websocket, for as long as
      this process runs.
    - Orders go through AlpacaBroker, which refuses to construct unless
      paper=True -- see src/execution/alpaca_broker.py.
    - The loop pauses outside market hours (checked via Alpaca's clock)
      instead of spinning or erroring when the stream goes quiet overnight.

Known limitation, read before leaving this running unattended: the replay
buffer still lives in memory only. If this process restarts mid-session
(crash, reboot, manual stop), the model's learned weights and the risk/
drift state below all resume correctly, but the buffer of raw samples
used for the *next* retrain starts empty again. Fine for now; the replay
buffer already supports save()/load() (src/data/buffer.py) if you want to
close that gap too.

The risk manager's daily-loss/consecutive-loss counters and the drift
monitor's rolling hit-rate/calibration/equity windows ARE now persisted
(checkpoints/risk_state.json, checkpoints/drift_state.json, saved after
every state-changing event) and resumed automatically on startup -- a
restart no longer forgets today's drawdown or silently re-arms a kill
switch that had already tripped.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be exported directly instead

from src.config import load_config
from src.data.alpaca_feed import AlpacaLiveFeed
from src.execution.alpaca_broker import AlpacaBroker
from src.model.network import TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskManager
from src.training.drift import DriftMonitor
from src.training.trainer import ContinualTrainer


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"Missing required environment variable {name}. Set it (or add it to a "
            f".env file -- see .env.example) before running this script."
        )
    return val


def wait_for_market_open(trading_client, poll_seconds: float = 60.0) -> None:
    while True:
        clock = trading_client.get_clock()
        if clock.is_open:
            return
        print(f"Market closed. Next open: {clock.next_open}. Checking again in "
              f"{poll_seconds:.0f}s...")
        time.sleep(poll_seconds)


def main():
    api_key = require_env("ALPACA_API_KEY")
    secret_key = require_env("ALPACA_SECRET_KEY")

    cfg = load_config("config.yaml")
    tickers = cfg["tickers"]

    from alpaca.trading.client import TradingClient
    trading_client = TradingClient(api_key, secret_key, paper=True)

    alpaca_cfg = cfg.get("alpaca", {})
    feed = AlpacaLiveFeed(tickers, api_key, secret_key, feed=alpaca_cfg.get("data_feed", "iex"))
    broker = AlpacaBroker(api_key, secret_key, paper=True)

    model = TradingNet(cfg["_model_cfg"])
    trainer = ContinualTrainer(model, cfg["_trainer_cfg"])
    risk_manager = RiskManager(cfg["_risk_cfg"])
    logger = DecisionLogger(cfg["logging"]["log_path"])
    drift_monitor = DriftMonitor(cfg["_drift_cfg"])

    # Resume from the most recently promoted checkpoint, if one exists, so a
    # restart (crash, reboot, manual stop/start) doesn't throw away every
    # promoted retrain and fall back to an untrained v0 model. This does NOT
    # restore the replay buffer -- that still resets on restart (see the
    # "Known limitation" note at the top of this file) -- but the risk
    # manager's daily-loss counters and drift monitor's rolling windows
    # (below) now do.
    resumed_version = trainer.load_latest_checkpoint()
    if resumed_version is not None:
        print(f"Resumed model weights from checkpoint: v{resumed_version} "
              f"(from {cfg['_trainer_cfg'].checkpoint_dir}/)")
    else:
        print(f"No existing checkpoint found in {cfg['_trainer_cfg'].checkpoint_dir}/ "
              f"-- starting from an untrained model.")

    # Same idea for the risk manager's daily-loss/consecutive-loss counters
    # and the drift monitor's rolling hit-rate/calibration/equity windows --
    # restored in place (mutating risk_manager.state / drift_monitor's
    # internal deques directly) BEFORE Orchestrator is constructed, so
    # Orchestrator starts already holding whatever state existed when this
    # process last stopped, instead of a kill switch that's forgotten
    # today's drawdown. Saved automatically by Orchestrator after every
    # state-changing event for the rest of this run (risk_state_path /
    # drift_state_path below) -- see RiskManager.save_state() /
    # DriftMonitor.save_state()'s docstrings for why "after every event"
    # rather than just on clean shutdown.
    state_dir = Path(cfg["_trainer_cfg"].checkpoint_dir)
    risk_state_path = state_dir / "risk_state.json"
    drift_state_path = state_dir / "drift_state.json"
    if risk_manager.load_state(risk_state_path):
        print(f"Resumed risk manager state: daily_pnl={risk_manager.state.daily_pnl_pct:.2f}%, "
              f"consecutive_losses={risk_manager.state.consecutive_losses}, "
              f"kill_switch_engaged={risk_manager.kill_switch_engaged()}")
    if drift_monitor.load_state(drift_state_path):
        snap = drift_monitor.snapshot()
        print(f"Resumed drift monitor state: {snap['n_outcomes']} outcomes, "
              f"hit_rate={snap['hit_rate']}, drawdown_pct={snap['drawdown_pct']}")

    orch = Orchestrator(
        tickers=tickers, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=cfg["_model_cfg"].window,
        label_cfg=cfg["_label_cfg"],
        risk_state_path=risk_state_path, drift_state_path=drift_state_path,
    )

    print(f"Alpaca paper trading (live data): tickers={tickers}")
    print("Waiting for market open if needed (Ctrl+C to stop)...")
    wait_for_market_open(trading_client)

    def stop_when_market_closes() -> bool:
        return not trading_client.get_clock().is_open

    try:
        orch.run(stop_check=stop_when_market_closes)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\n=== Session summary ===")
    print("Equity:", round(broker.get_equity(), 2))
    print("Retrain events:", len(orch.retrain_events),
          "| promoted:", sum(1 for r in orch.retrain_events if r["promoted"]))
    print("Drift snapshot:", drift_monitor.snapshot())
    print("Kill switch engaged:", risk_manager.kill_switch_engaged(),
          risk_manager.state.halt_reasons if risk_manager.kill_switch_engaged() else "")
    print(f"Decision log: {cfg['logging']['log_path']}")


if __name__ == "__main__":
    main()
