#!/usr/bin/env python
"""
Offline backtest / sanity check: train the network on an initial chunk of
history with plain supervised learning (time-ordered train/val split, no
shuffling), then report validation accuracy and a naive simulated P&L on
the held-out period. Run this BEFORE `run_paper_trading.py` to sanity
check that the architecture + features can learn anything at all on your
data before handing it the (much more complex) online loop.

    python scripts/backtest.py --ticker SPY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import load_config
from src.data.feed import SyntheticFeed, YFinanceFeed
from src.data.features import build_windows, compute_features, make_labels
from src.model.network import TradingNet, compute_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--feed", default=None, choices=["synthetic", "yfinance"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ticker = args.ticker or cfg["tickers"][0]
    feed_kind = args.feed or cfg["data"]["feed"]

    if feed_kind == "synthetic":
        feed = SyntheticFeed([ticker], n_bars=cfg["data"]["n_bars"])
        bars = feed.get_history(ticker, cfg["data"]["n_bars"])
    else:
        feed = YFinanceFeed([ticker], period=cfg["data"]["yfinance_period"],
                             interval=cfg["data"]["yfinance_interval"])
        bars = feed._series[ticker]

    feats = compute_features(bars)
    labels = make_labels(bars, cfg["_label_cfg"])
    X, y_action, y_ret, idxs = build_windows(feats, labels, cfg["_model_cfg"].window)
    print(f"Built {len(X)} windowed samples for {ticker}.")

    n_val = int(len(X) * args.val_fraction)
    n_train = len(X) - n_val
    Xtr, ytr_a, ytr_r = X[:n_train], y_action[:n_train], y_ret[:n_train]
    Xva, yva_a, yva_r = X[n_train:], y_action[n_train:], y_ret[n_train:]
    print(f"Train: {len(Xtr)}  Val (held-out, later in time): {len(Xva)}")

    model = TradingNet(cfg["_model_cfg"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["_trainer_cfg"].lr)
    batch_size = cfg["_trainer_cfg"].batch_size

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        total_loss = 0.0
        for start in range(0, len(Xtr), batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = torch.from_numpy(Xtr[batch_idx])
            ya = torch.from_numpy(ytr_a[batch_idx])
            yr = torch.from_numpy(ytr_r[batch_idx])
            optimizer.zero_grad()
            logits, pred_ret = model(xb)
            loss, _ = compute_loss(logits, pred_ret, ya, yr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        avg_loss = total_loss / max(1, len(Xtr))

        model.eval()
        with torch.no_grad():
            logits, pred_ret = model(torch.from_numpy(Xva))
            val_acc = (logits.argmax(dim=-1) == torch.from_numpy(yva_a)).float().mean().item()
        print(f"epoch {epoch + 1:>2}/{args.epochs}  train_loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

    # Naive simulated P&L on the held-out period: go long/short per the
    # model's call, flat on "hold", ignoring costs (illustrative only --
    # the real risk-managed P&L comes from run_paper_trading.py).
    model.eval()
    with torch.no_grad():
        out = model.predict(torch.from_numpy(Xva))
    actions = out["action"].numpy() - 1  # {0,1,2} -> {-1,0,1}
    pnl = actions * yva_r
    cum_pnl = float(np.sum(pnl))
    hit_rate_nonhold = float(np.mean((actions[actions != 0] ==
                                       (yva_a[actions != 0] - 1))) if np.any(actions != 0) else 0.0)
    print(f"\nHeld-out naive cumulative return (no costs): {cum_pnl:.4f}")
    print(f"Directional hit rate on non-hold calls: {hit_rate_nonhold:.3f}  "
          f"(n={int(np.sum(actions != 0))} of {len(actions)})")
    print("\nThis is a quick sanity check, not a validated trading result -- "
          "see the README for why a positive number here is necessary but far "
          "from sufficient before considering real capital.")


if __name__ == "__main__":
    main()
