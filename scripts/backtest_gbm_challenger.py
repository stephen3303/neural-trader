#!/usr/bin/env python
"""
Offline, apples-to-apples comparison: does a gradient-boosted tree model
(scikit-learn's HistGradientBoosting*, trained on the SAME engineered
features and labels this project already computes) do any better than
the live system's GRU, on this project's own data?

This does NOT touch Orchestrator, RiskManager, the live champion/
challenger promotion gate, or anything that could affect a real/paper
order. It is a standalone research script, same spirit as
scripts/backtest.py (which this borrows its train/val methodology from)
-- just run twice, once per model family, on identical walk-forward
splits, so any difference in the numbers is actually about the model,
not about getting different data or a different split by accident.

Why walk-forward with multiple folds, instead of one 80/20 split (what
scripts/backtest.py and the live ContinualTrainer's own promotion gate
both do): a single held-out slice only tells you "did it work on this
one period" -- it says nothing about whether a model that looked good
on one stretch of history also holds up on a different one. Each fold
here trains on an expanding prefix of history and validates on the next
contiguous block, so a model that only works in calm regimes (for
example) should show up as inconsistent across folds, not just get
judged on whichever period happened to get sampled into the single
val slice.

Why a tree model doesn't get the GRU's raw [window, n_feat] sequence:
trees have no inherent notion of step order, so instead it gets
src.data.features.summarize_window()'s per-feature summary statistics
(last/mean/std/min/max/slope across the window) -- see that function's
docstring for the reasoning. Two separate models per fold, mirroring
TradingNet's own two-head design: a classifier for the 3-way action and
a regressor for the forward return.

Metrics reported, identical methodology for both models so they're
directly comparable:
  - val accuracy (3-way action classification)
  - directional hit rate on non-hold calls only (the metric that
    actually matters for trading -- calling "hold" correctly is cheap
    and uninformative under this project's label imbalance, see
    ContinualTrainer._class_weights()'s docstring)
  - naive, cost-free cumulative return on the held-out fold (action *
    realized forward return, summed) -- illustrative only, exactly as
    scripts/backtest.py already caveats: this ignores slippage/
    commission/sizing/risk caps entirely, so it is NOT a substitute for
    running the real strategy through scripts/backtest_portfolio.py.

    python scripts/backtest_gbm_challenger.py                    # synthetic, all configured tickers
    python scripts/backtest_gbm_challenger.py --tickers SPY,QQQ  # a subset
    python scripts/backtest_gbm_challenger.py --folds 6 --epochs 12
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
from src.data.features import build_windows, compute_features, make_labels, summarize_window
from src.model.network import TradingNet, compute_loss
from src.utils import seed_everything


def _train_gru(cfg, Xtr, ytr_a, ytr_r, epochs: int) -> TradingNet:
    model = TradingNet(cfg["_model_cfg"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["_trainer_cfg"].lr)
    batch_size = cfg["_trainer_cfg"].batch_size
    for _epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        for start in range(0, len(Xtr), batch_size):
            batch_idx = perm[start:start + batch_size]
            if len(batch_idx) == 0:
                continue
            xb = torch.from_numpy(Xtr[batch_idx])
            ya = torch.from_numpy(ytr_a[batch_idx])
            yr = torch.from_numpy(ytr_r[batch_idx])
            optimizer.zero_grad()
            logits, pred_ret = model(xb)
            loss, _ = compute_loss(logits, pred_ret, ya, yr)
            loss.backward()
            optimizer.step()
    return model


def _eval_gru(model: TradingNet, Xva, yva_a, yva_r) -> dict:
    model.eval()
    with torch.no_grad():
        out = model.predict(torch.from_numpy(Xva))
    actions = out["action"].numpy() - 1  # {0,1,2} -> {-1,0,1}
    return _score(actions, yva_a - 1, yva_r)


def _train_gbm(Xtr, ytr_a, ytr_r, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    Str = summarize_window(Xtr)
    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(Str, ytr_a)
    reg = HistGradientBoostingRegressor(random_state=seed)
    reg.fit(Str, ytr_r)
    return clf, reg


def _eval_gbm(clf, reg, Xva, yva_a, yva_r) -> dict:
    Sva = summarize_window(Xva)
    actions = clf.predict(Sva) - 1  # {0,1,2} -> {-1,0,1}
    return _score(actions, yva_a - 1, yva_r)


def _score(actions_signed: np.ndarray, true_action_signed: np.ndarray, y_ret: np.ndarray) -> dict:
    """actions_signed / true_action_signed are both in {-1, 0, 1}. Shared
    scoring so the GRU and GBM paths can never silently diverge in how a
    number gets computed -- see the module docstring for what each one
    means."""
    acc = float(np.mean(actions_signed == true_action_signed))
    non_hold = actions_signed != 0
    n_trades = int(np.sum(non_hold))
    hit_rate = float(np.mean(actions_signed[non_hold] == true_action_signed[non_hold])) if n_trades else 0.0
    cum_pnl = float(np.sum(actions_signed * y_ret))
    return {"accuracy": acc, "hit_rate_nonhold": hit_rate, "n_trades": n_trades, "cum_pnl_nocost": cum_pnl}


def _avg(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


def _fmt(m: dict) -> str:
    return (f"acc={m.get('accuracy', 0):.3f}  hit_rate(non-hold)={m.get('hit_rate_nonhold', 0):.3f}  "
            f"n_trades={m.get('n_trades', 0):.0f}  cum_pnl(no cost)={m.get('cum_pnl_nocost', 0):+.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--feed", default=None, choices=["synthetic", "yfinance"])
    parser.add_argument("--tickers", default=None, help="comma-separated; default: every ticker in config.yaml")
    parser.add_argument("--folds", type=int, default=4, help="walk-forward folds (expanding window)")
    parser.add_argument("--epochs", type=int, default=8, help="GRU training epochs per fold")
    parser.add_argument("--seed", type=int, default=42,
                         help="Also passed to SyntheticFeed as its data-generation seed -- "
                              "run with a different --seed if you split --tickers across "
                              "multiple invocations, since SyntheticFeed seeds one shared RNG "
                              "once and draws each ticker's series from it IN LIST ORDER (see "
                              "src/data/feed.py): two separate invocations using the same seed "
                              "hand their Nth ticker the exact same underlying synthetic series "
                              "regardless of that ticker's actual symbol, which is NOT a second "
                              "independent regime to compare against (found while building this "
                              "script: splitting all 12 configured tickers into two 6-ticker "
                              "runs at the default seed silently gave the second run's six "
                              "tickers identical numbers, position-for-position, to the first).")
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.config)
    feed_kind = args.feed or cfg["data"]["feed"]
    tickers = args.tickers.split(",") if args.tickers else list(cfg["tickers"])

    gru_folds, gbm_folds = [], []  # flat across all tickers, for the grand summary

    # Construct ONE feed covering every ticker, exactly like
    # scripts/backtest_portfolio.py does -- NOT a fresh SyntheticFeed per
    # ticker inside the loop below. SyntheticFeed seeds one shared RNG
    # once at construction and draws each ticker's series from it in
    # sequence (see src/data/feed.py), so a single-ticker SyntheticFeed
    # call always replays the exact same draws from the same default
    # seed=7 -- constructing one per ticker here would have silently
    # handed every ticker byte-for-byte IDENTICAL synthetic data (caught
    # by this script's own first real run: SPY, GOOGL, and AMZN posted
    # the GBM numbers, fold for fold).
    if feed_kind == "synthetic":
        feed = SyntheticFeed(tickers, n_bars=cfg["data"]["n_bars"], seed=args.seed)
    else:
        feed = YFinanceFeed(tickers, period=cfg["data"]["yfinance_period"],
                             interval=cfg["data"]["yfinance_interval"])

    for ticker in tickers:
        bars = feed.get_history(ticker, cfg["data"]["n_bars"])
        feats = compute_features(bars)
        labels = make_labels(bars, cfg["_label_cfg"])
        X, y_action, y_ret, idxs = build_windows(feats, labels, cfg["_model_cfg"].window)

        n_blocks = args.folds + 1
        block_size = len(X) // n_blocks
        if block_size < cfg["_trainer_cfg"].batch_size:
            print(f"{ticker}: only {len(X)} windowed samples -- too few for {args.folds} "
                  f"walk-forward folds, skipping.")
            continue

        print(f"\n=== {ticker}: {len(X)} windowed samples, {args.folds} walk-forward folds ===")
        ticker_gru, ticker_gbm = [], []
        for fold in range(1, n_blocks):
            train_end = fold * block_size
            val_end = train_end + block_size if fold < n_blocks - 1 else len(X)
            Xtr, ya_tr, yr_tr = X[:train_end], y_action[:train_end], y_ret[:train_end]
            Xva, ya_va, yr_va = X[train_end:val_end], y_action[train_end:val_end], y_ret[train_end:val_end]

            gru_model = _train_gru(cfg, Xtr, ya_tr, yr_tr, args.epochs)
            gru_metrics = _eval_gru(gru_model, Xva, ya_va, yr_va)

            clf, reg = _train_gbm(Xtr, ya_tr, yr_tr, args.seed)
            gbm_metrics = _eval_gbm(clf, reg, Xva, ya_va, yr_va)

            print(f"  fold {fold} (train n={train_end:>5}, val n={val_end - train_end:>4}): "
                  f"GRU {_fmt(gru_metrics)}")
            print(f"  fold {fold} (train n={train_end:>5}, val n={val_end - train_end:>4}): "
                  f"GBM {_fmt(gbm_metrics)}")

            ticker_gru.append(gru_metrics)
            ticker_gbm.append(gbm_metrics)

        if ticker_gru:
            print(f"  {ticker} avg across folds -- GRU {_fmt(_avg(ticker_gru))}")
            print(f"  {ticker} avg across folds -- GBM {_fmt(_avg(ticker_gbm))}")
            gru_folds.extend(ticker_gru)
            gbm_folds.extend(ticker_gbm)

    print("\n=== Overall (every fold, every ticker) ===")
    print("GRU:", _fmt(_avg(gru_folds)))
    print("GBM:", _fmt(_avg(gbm_folds)))
    print("\nThis is a research comparison on naive, cost-free simulated P&L -- it does "
          "NOT run the risk manager, broker cost model, or kill switch, and a win here is "
          "necessary but far from sufficient before considering wiring a different model "
          "family into the live champion/challenger loop. See scripts/backtest_portfolio.py "
          "for the real, cost-aware, risk-managed comparison once a model family looks "
          "promising here.")


if __name__ == "__main__":
    main()
