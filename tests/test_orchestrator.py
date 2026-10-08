import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.feed import SyntheticFeed
from src.data.features import LabelConfig
from src.execution.broker import PaperBroker
from src.model.network import ModelConfig, TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskConfig, RiskManager
from src.training.drift import DriftConfig, DriftMonitor
from src.training.trainer import ContinualTrainer, TrainerConfig

N_FEATURES = 15


def _make_orchestrator(tmp_path, window=20):
    tickers = ["T1"]
    feed = SyntheticFeed(tickers, n_bars=500, seed=1)
    model = TradingNet(ModelConfig(n_features=N_FEATURES, window=window, hidden_size=8, trunk_size=8))
    trainer = ContinualTrainer(model, TrainerConfig(checkpoint_dir=str(tmp_path / "checkpoints")))
    risk_manager = RiskManager(RiskConfig())
    broker = PaperBroker()
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    drift_monitor = DriftMonitor(DriftConfig())
    orch = Orchestrator(
        tickers=tickers, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=window, label_cfg=LabelConfig(),
        # features.py's rolling indicators (sma_30_dev, vol_z, etc.) need up
        # to 30 bars of their own warm-up before they stop being NaN, on top
        # of the `window`-length lookback build_windows() needs -- too little
        # history here means build_windows() filters every candidate window
        # out for containing NaNs and _try_predict() returns None.
        warmup_bars=window + 35,
    )
    return orch


def test_predictions_use_the_most_recently_promoted_model(tmp_path):
    """Regression test for a real bug found while reviewing the continual-
    retraining path: ContinualTrainer.maybe_retrain() promotes a challenger
    by REBINDING its own `self.model` attribute to a new object
    (`self.model = challenger`) -- it never mutates the original model
    in place. Orchestrator previously called `self.model.predict(...)`,
    where `self.model` was a separate reference captured once at
    construction time. After the very first promoted retrain, those two
    references point at two different objects: `trainer.model` is the
    improved challenger, but `orchestrator.model` is still the stale,
    pre-promotion model -- so every prediction for the rest of the run
    silently ignores all future retraining, while the logs claim a newer
    `model_version` the whole time. This test forces a promotion directly
    (bypassing the stochastic retrain gate, for a fast deterministic test)
    and asserts a prediction right after it actually reflects the new
    model, not the stale one."""
    orch = _make_orchestrator(tmp_path)

    # Prime enough history to make a real prediction.
    for bar in orch.feed.stream(orch.tickers):
        orch._update_history(bar)
        orch._bar_count[bar.ticker] += 1
        if len(orch._history[bar.ticker]) >= orch.warmup_bars:
            break

    # Simulate a promoted retrain exactly like ContinualTrainer.maybe_retrain
    # does on promotion: swap in a NEW model object, don't mutate the old one.
    old_model = orch.trainer.model
    new_model = TradingNet(old_model.cfg)
    # Make the new model's output unambiguously distinguishable from the old
    # one's, so "which model actually produced this prediction" is observable.
    with torch.no_grad():
        new_model.action_head.bias[:] = torch.tensor([10.0, -10.0, -10.0])  # forces "sell" (index 0)
        old_model.action_head.bias[:] = torch.tensor([-10.0, -10.0, 10.0])  # forces "buy" (index 2)
    orch.trainer.model = new_model
    orch.trainer._version += 1

    pred = orch._try_predict(orch.tickers[0], pd_timestamp_stub())
    assert pred is not None, "expected a prediction once warmup_bars of history exist"
    assert pred["action"] == 0, (
        "prediction used the stale pre-promotion model (predicted 'buy', action=2) "
        "instead of the newly promoted one (should predict 'sell', action=0) -- "
        "Orchestrator is not reading orch.trainer.model for inference"
    )


def pd_timestamp_stub():
    import pandas as pd
    return pd.Timestamp("2024-01-01")
