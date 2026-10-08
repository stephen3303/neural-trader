"""
Small shared utilities that don't belong to any one module.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every source of randomness this codebase actually uses, so a
    script that calls this once at startup gets fully reproducible runs.

    Found while building the backtest harness (task #16): re-running
    `scripts/backtest_portfolio.py` twice against the exact same
    historical/synthetic data produced wildly different results (one run
    tripped the kill switch almost immediately and never traded again;
    another ran the whole way with "hold or low confidence" on every
    single bar) -- NOT because of a bug in the strategy, but because
    nothing in the codebase ever seeded torch's global RNG. TradingNet's
    weight initialization (nn.Linear/nn.GRU's default init) and dropout
    both draw from it, so every run started from a different random
    model and trained differently from there, even though
    SyntheticFeed's market data is itself already deterministic (it
    takes its own explicit `seed` argument, default 7). Without this,
    there is no way to tell "did my change actually help" apart from
    "I got a different random initialization this time" -- which
    defeats the point of a backtest meant to measure long-term
    profitability.

    This seeds:
    - Python's `random` module (not currently used for anything that
      affects results, but cheap and standard to cover anyway).
    - numpy's legacy global RNG (`np.random.seed`) -- note this does
      NOT affect `np.random.default_rng(...)` generators, which draw
      their own independent entropy; ContinualTrainer/ReplayBuffer use
      `TrainerConfig.seed` -> `ContinualTrainer._rng` for that instead
      (see trainer.py), not this function.
    - torch's global RNG (`torch.manual_seed`), which governs
      `TradingNet`'s weight initialization and dropout -- this is the
      one that actually explained the wildly divergent backtest runs
      above.
    - torch's CUDA RNG, if a GPU is available, for the same reason.

    Does not make PyTorch's GPU convolution/cuDNN kernels bit-for-bit
    deterministic (this codebase doesn't use any), and does not seed
    SyntheticFeed/YFinanceFeed -- those already take their own explicit
    seed/period arguments and are deterministic on their own."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
