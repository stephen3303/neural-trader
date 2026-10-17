# Label-horizon paper experiment

Backtests (2025Q1-2026Q3, 2 seeds) showed a 30-bar label horizon beating the
current 15 by about 25 points summed over 14 runs. This puts both on paper
side by side before changing the live trader.

| Arm | Config | Differs from config.yaml |
|---|---|---|
| `paper-h15` | `config-h15.yaml` | `drift.hit_rate_floor: 0.0` |
| `paper-h30` | `config-h30.yaml` | `label.horizon: 30`, `drift.hit_rate_floor: 0.0` |

The hit-rate floor is off in both arms because at 0.34 it halts within the
first days (it sits below the model's own chance rate). The drawdown,
Brier, daily-loss and consecutive-loss limits still apply. Both arms start
from an untrained model, so they learn on the same live bars from the same
starting point. The live trader (`live-trader`) is untouched.

## Setup (once)
1. In the Alpaca dashboard, create two extra **paper** accounts and generate
   API keys for each.
2. Next to `.env`, create `.env.paper-h15` and `.env.paper-h30`, each with
   that account's `ALPACA_API_KEY=` and `ALPACA_SECRET_KEY=`. Never reuse the
   live trader's keys: an arm would adopt its positions on startup.
3. `docker compose build` then
   `docker compose --profile experiment up -d paper-h15 paper-h30`

Logs and checkpoints go to `experiments/label-horizon/h15/` and `h30/`
(git-ignored). Compare the two arms' equity and trades after a few weeks.
