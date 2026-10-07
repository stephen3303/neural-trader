import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.model.network import ModelConfig, TradingNet, compute_loss


def test_forward_shapes():
    cfg = ModelConfig(n_features=15, window=60, hidden_size=16, gru_layers=1, trunk_size=16)
    model = TradingNet(cfg)
    x = torch.randn(4, 60, 15)
    logits, ret = model(x)
    assert logits.shape == (4, 3)
    assert ret.shape == (4,)


def test_predict_outputs_valid_probs():
    cfg = ModelConfig(n_features=15, window=60, hidden_size=16, gru_layers=1, trunk_size=16)
    model = TradingNet(cfg)
    x = torch.randn(3, 60, 15)
    out = model.predict(x)
    probs = out["probs"].numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert set(out["action"].numpy().tolist()).issubset({0, 1, 2})


def test_loss_is_finite_and_backprops():
    cfg = ModelConfig(n_features=15, window=20, hidden_size=8, gru_layers=1, trunk_size=8)
    model = TradingNet(cfg)
    x = torch.randn(8, 20, 15)
    y_action = torch.randint(0, 3, (8,))
    y_ret = torch.randn(8) * 0.01
    logits, pred_ret = model(x)
    loss, parts = compute_loss(logits, pred_ret, y_action, y_ret)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0
