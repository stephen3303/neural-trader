"""
The neural network itself.

Architecture (deliberately small and fast to retrain, since it has to be
fine-tuned on a schedule, not just trained once):

    [batch, window, n_features]
            |
        GRU encoder (2 layers, bidirectional=False -- causal, no lookahead)
            |
      temporal attention pooling (learned weights over the window instead
      of just taking the last hidden state, so the network can weigh e.g.
      "the breakout 5 bars ago" more than "the flat bar 1 bar ago")
            |
       shared trunk (LayerNorm + MLP)
          /        \
   action head    return head
  (3-way softmax   (scalar regression:
   sell/hold/buy)   expected forward return)

Two heads instead of one because they regularize each other: the
classification head gives a calibrated, risk-manager-friendly decision
signal (with a confidence/probability the risk layer can threshold on),
while the regression head keeps the network honest about *magnitude*, not
just direction -- without it, a classifier can "win" by being barely right
about direction on huge numbers of near-zero moves.

Why a GRU instead of a Transformer: at the window lengths used here
(60-120 bars) a 2-layer GRU with attention pooling gets comparable
sequence-modeling quality at a fraction of the parameters and retraining
cost, which matters a lot for a model that's being incrementally
fine-tuned continuously rather than trained once offline. If you later
want to scale up (longer context, cross-ticker attention), swap
`TemporalEncoder` for a Transformer encoder -- the rest of the file
(heads, loss, inference wrapper) doesn't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_features: int
    window: int = 60
    hidden_size: int = 64
    gru_layers: int = 2
    dropout: float = 0.2
    trunk_size: int = 64


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, n_features]
        h_seq, _ = self.gru(x)                      # [batch, window, hidden]
        attn_logits = self.attn(h_seq).squeeze(-1)   # [batch, window]
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [batch, window, 1]
        pooled = (h_seq * attn_weights).sum(dim=1)   # [batch, hidden]
        return pooled


class TradingNet(nn.Module):
    """Full network: encoder -> trunk -> (action_logits, expected_return)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TemporalEncoder(cfg)
        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.trunk_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.action_head = nn.Linear(cfg.trunk_size, 3)   # sell=0, hold=1, buy=2
        self.return_head = nn.Linear(cfg.trunk_size, 1)

    def forward(self, x: torch.Tensor):
        pooled = self.encoder(x)
        h = self.trunk(pooled)
        action_logits = self.action_head(h)
        expected_return = self.return_head(h).squeeze(-1)
        return action_logits, expected_return

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """x: [batch, window, n_features] -> dict of numpy-friendly outputs."""
        self.eval()
        action_logits, expected_return = self.forward(x)
        probs = F.softmax(action_logits, dim=-1)
        action = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        return {
            "action": action,                 # 0=sell,1=hold,2=buy
            "probs": probs,                   # [batch, 3]
            "confidence": confidence,         # [batch]
            "expected_return": expected_return,  # [batch]
        }


def compute_loss(action_logits, expected_return, y_action, y_ret,
                  return_loss_weight: float = 1.0, class_weights: torch.Tensor | None = None):
    """Combined classification + regression loss (see module docstring for
    why both heads exist)."""
    ce = F.cross_entropy(action_logits, y_action, weight=class_weights)
    mse = F.mse_loss(expected_return, y_ret)
    total = ce + return_loss_weight * mse
    return total, {"ce": ce.item(), "mse": mse.item(), "total": total.item()}
