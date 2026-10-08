"""
A permanent, CI-enforced "gross error" smoke test for the real strategy
pipeline -- not a reimplementation of any one component, but an actual
end-to-end run of Orchestrator + RiskManager + ContinualTrainer +
PaperBroker, exactly like scripts/run_paper_trading.py /
scripts/backtest_portfolio.py, asserting on sanity bounds a healthy
system should never violate.

Why this exists: every bug fixed in this project tonight (model
collapse, the dead kill-switch, stale account equity, non-reproducible
runs) was found by actually RUNNING the real pipeline on real data, not
by reasoning about the code in isolation -- unit tests on individual
functions passed the whole time each of those bugs was live. This file
turns that same practice into something that runs automatically on
every push (see .github/workflows/tests.yml), instead of depending on
someone remembering to run a backtest by hand before trusting a change.
It intentionally does NOT assert on trading performance (a losing run
is not a bug -- see the README's "before you even think about live
trading" section) -- only on things that indicate the pipeline itself
is broken: crashes, NaN/inf propagating into state that's supposed to
be a plain float, equity collapsing far faster than any bounded
position-sizing config should allow, and position/exposure caps being
silently violated.

All tests here share ONE real pipeline run (a module-scoped fixture)
rather than each re-running it -- this is a slow, real integration run
(not a mocked unit test), so re-running it per-assertion would make this
file dominate the suite's runtime for no extra coverage. Scale (bars,
tickers, retrain frequency) is deliberately small -- enough to exercise
warm-up, real predictions, sizing, fills, outcome resolution, and a
handful of retrains, while keeping the whole file's runtime well under
what the rest of the (fast, unit-level) suite costs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.data.feed import SyntheticFeed
from src.data.features import LabelConfig
from src.execution.broker import PaperBroker
from src.model.network import ModelConfig, TradingNet
from src.monitor.logger import DecisionLogger
from src.orchestrator import Orchestrator
from src.risk.manager import RiskConfig, RiskManager
from src.training.drift import DriftConfig, DriftMonitor
from src.training.trainer import ContinualTrainer, TrainerConfig
from src.utils import seed_everything

N_FEATURES = 15
TICKERS = ["T1", "T2"]


def _run_real_pipeline(tmp_path, n_bars=250, seed=42):
    """Builds and runs the actual production code path (not a
    reimplementation). max_history=100/warmup_bars=55 (rather than the
    defaults of 400/120) deliberately shrinks how much of the synthetic
    series gets consumed as warm-up, so a modest n_bars still leaves
    plenty of bars for real streaming predictions -- see the sibling
    tests' use of this function for why the proportions matter."""
    seed_everything(seed)
    feed = SyntheticFeed(TICKERS, n_bars=n_bars, seed=seed)
    model_cfg = ModelConfig(n_features=N_FEATURES, window=20, hidden_size=8, trunk_size=8)
    model = TradingNet(model_cfg)
    trainer_cfg = TrainerConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        min_buffer_size=60, retrain_every_n_new=60, epochs_per_retrain=1,
        batch_size=32, seed=seed,
    )
    trainer = ContinualTrainer(model, trainer_cfg)
    risk_manager = RiskManager(RiskConfig())  # defaults: 10% max position, 60% max gross exposure
    broker = PaperBroker(starting_cash=100_000.0, slippage_bps=2.0, commission_bps=1.0)
    logger = DecisionLogger(tmp_path / "decisions.jsonl")
    drift_monitor = DriftMonitor(DriftConfig())

    orch = Orchestrator(
        tickers=TICKERS, feed=feed, model=model, trainer=trainer,
        risk_manager=risk_manager, broker=broker, logger=logger,
        drift_monitor=drift_monitor, window=20, label_cfg=LabelConfig(),
        max_history=100, warmup_bars=55,
    )
    orch.run()  # exhausts the feed -- bounded by n_bars above, no max_bars needed
    return orch


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    return _run_real_pipeline(tmp_path_factory.mktemp("gross_errors"))


class TestGrossErrors:
    def test_a_real_run_completes_without_raising(self, real_run):
        """The most basic possible gross-error check: does the real
        pipeline run start-to-finish on real (synthetic) multi-ticker
        data without an unhandled exception, and actually process bars
        (not silently do nothing, e.g. from a warm-up/feed-length
        mismatch)."""
        assert len(real_run.equity_curve) > 0

    def test_equity_curve_is_always_finite_and_positive(self, real_run):
        """A NaN/inf/negative equity value anywhere in the curve means
        something in the sizing/fill/P&L math produced garbage -- this
        is exactly the kind of error class math.isfinite() guards
        exist for elsewhere in this codebase (see
        RiskManager.update_account_equity)."""
        for e in real_run.equity_curve:
            assert math.isfinite(e), f"non-finite equity value: {e}"
            assert e > 0, f"non-positive equity value: {e}"

    def test_equity_never_collapses_catastrophically(self, real_run):
        """Not a performance assertion (a genuinely bad model losing
        money slowly is expected and fine) -- a guard against a sizing/
        leverage bug blowing up the account far faster than the
        configured risk caps (max_position_pct=10%, max_gross_exposure_pct=60%,
        hard_stop_loss_pct=3%) should ever allow in a few hundred bars.
        If this fires, suspect a sizing or stop-loss bug, not an unlucky
        model."""
        starting = real_run.broker.starting_cash
        worst = min(real_run.equity_curve)
        assert worst > starting * 0.5, (
            f"equity dropped below 50% of starting cash ({worst:.2f} of {starting:.2f}) "
            "-- this is far more than the configured per-trade/exposure caps should allow "
            "over this short a run; treat as a probable sizing/leverage bug, not bad luck."
        )

    def test_every_realized_trade_has_a_finite_pnl(self, real_run):
        for t in real_run.trade_log:
            assert math.isfinite(t["pnl_pct_of_equity"]), t

    def test_no_single_trade_exceeds_the_configured_position_cap(self, real_run):
        """size_pct_equity on any realized trade must never exceed
        RiskConfig.max_position_pct -- a violation here means
        size_order()'s caps are being bypassed somewhere in the loop,
        not that the model made an aggressive call (aggressive calls
        are exactly what the cap exists to bound)."""
        cap = real_run.risk_manager.cfg.max_position_pct
        for t in real_run.trade_log:
            assert t["size_pct_equity"] <= cap + 1e-6, (
                f"trade sized at {t['size_pct_equity']}% exceeds the configured "
                f"max_position_pct of {cap}%: {t}"
            )

    def test_kill_switch_state_is_internally_consistent(self, real_run):
        rm = real_run.risk_manager
        if rm.kill_switch_engaged():
            assert len(rm.state.halt_reasons) > 0, "kill switch engaged with no recorded reason"

    def test_retrain_records_are_well_formed_when_any_occurred(self, real_run):
        """Loose structural check (not a promotion-rate assertion --
        see the model-collapse/degeneracy-veto README sections for why
        most early retrains being vetoed is itself expected, not a
        bug) that every retrain record has the shape the dashboard and
        backtest harness depend on."""
        for r in real_run.retrain_events:
            assert isinstance(r["promoted"], bool)
            assert r["degeneracy_veto_reason"] is None or isinstance(r["degeneracy_veto_reason"], str)
            assert math.isfinite(r["champion_val"]["ce"] + r["champion_val"]["mse"])
            assert math.isfinite(r["challenger_val"]["ce"] + r["challenger_val"]["mse"])


def test_same_seed_reproduces_the_entire_run(tmp_path):
    """End-to-end reproducibility check for the fix in the backtest-
    harness commit (seed_everything + TrainerConfig.seed): two full
    pipeline runs with the same seed must produce IDENTICAL equity
    curves and trade logs, not just similar ones. Deliberately its own
    pair of fresh runs (not `real_run`) since it needs two independent
    executions to compare."""
    orch_a = _run_real_pipeline(tmp_path / "a", seed=123)
    orch_b = _run_real_pipeline(tmp_path / "b", seed=123)
    assert orch_a.equity_curve == orch_b.equity_curve
    assert len(orch_a.trade_log) == len(orch_b.trade_log)
    for ta, tb in zip(orch_a.trade_log, orch_b.trade_log):
        assert ta == tb
