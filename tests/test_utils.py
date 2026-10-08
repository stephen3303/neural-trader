import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.utils import seed_everything


class TestSeedEverything:
    """Regression coverage for the reproducibility bug found while
    building the backtest harness: torch's global RNG (which governs
    TradingNet's weight init and dropout) was never seeded anywhere, so
    two runs against the identical, deterministic SyntheticFeed data
    could diverge completely in their trading behavior."""

    def test_same_seed_reproduces_python_random(self):
        seed_everything(123)
        a = [random.random() for _ in range(5)]
        seed_everything(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_same_seed_reproduces_numpy_legacy_global_rng(self):
        seed_everything(123)
        a = np.random.rand(5)
        seed_everything(123)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_same_seed_reproduces_torch_global_rng(self):
        seed_everything(123)
        a = torch.rand(5)
        seed_everything(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_diverge(self):
        seed_everything(1)
        a = torch.rand(5)
        seed_everything(2)
        b = torch.rand(5)
        assert not torch.equal(a, b)

    def test_seeds_a_freshly_constructed_models_weights_reproducibly(self):
        """The actual real-world symptom: TradingNet() picks its initial
        weights from torch's global RNG at construction time."""
        from src.model.network import ModelConfig, TradingNet

        cfg = ModelConfig(n_features=10, window=20, hidden_size=8, trunk_size=8)
        seed_everything(42)
        m1 = TradingNet(cfg)
        seed_everything(42)
        m2 = TradingNet(cfg)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)
