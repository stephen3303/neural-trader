# Label-horizon paper experiment

Backtests (2025Q1-2026Q3, 2 seeds) showed a 30-bar label horizon beating the
current 15 by about 25 points summed over 14 runs. This puts the 30-bar
setting on paper before changing the live trader.

| Arm | Config | Differs from config.yaml |
|---|---|---|
| `paper-h15` | `config-h15.yaml` | `drift.hit_rate_floor: 0.0`, `risk.max_consecutive_losses: 100000` |
| `paper-h30` | `config-h30.yaml` | `label.horizon: 30`, `drift.hit_rate_floor: 0.0`, `risk.max_consecutive_losses: 100000` |

The hit-rate floor is off in both arms because at 0.34 it halts within the
first days (it sits below the model's own chance rate). The consecutive-loss
halt is off too: backtests of these configs showed 6 losses in a row latching
the halt by trading day 5 to 13 of every quarter, because the bot makes about
70 trades a day at a ~45% win rate. The drawdown, Brier and daily-loss limits
still apply. Both arms start
from an untrained model, so they learn on the same live bars from the same
starting point. The live trader (`live-trader`) is untouched.

## Setup (once)
Each arm needs its own Alpaca **paper** account, and the live trader already uses one.
Never reuse the live trader's keys: an arm would adopt its positions on startup.

1. Next to `.env`, create `.paper-h30.env` (and `.paper-h15.env` if you have a second
   extra account), each with that account's `ALPACA_API_KEY=` and `ALPACA_SECRET_KEY=`.
   The names end in `.env` so Windows doesn't treat the arm name as the extension.
   If you save from Notepad, pick "All files" so it doesn't add `.txt`.
2. `mkdir -p experiments/label-horizon/h30/logs experiments/label-horizon/h30/checkpoints`
   (same with `h15` for the second arm).
3. `docker compose build` then `docker compose up -d paper-h30`
4. The arms don't open their own data stream. The free Alpaca data plan allows one
   real-time websocket per login, and paper accounts under one login share it (in the
   week of 2026-10-19 the arm and the live trader took turns being locked out with
   `connection limit exceeded`). So `live-trader` owns the stream and appends every bar
   to `bars.jsonl` in the `shared-bars` Docker volume, and the arms read that file. The arms get bars only
   while `live-trader` is running, and both trade on identical bars.

**With only one extra account**, run `paper-h30`. The 15-bar side then comes from the
live trader (if its hit-rate floor is off, so it keeps trading) or from the backtests.

Logs and checkpoints go to `experiments/label-horizon/h15/` and `h30/`
(git-ignored, and kept out of the image). The dashboard only reads `./logs`, so
the arms don't show there: compare their `decisions.jsonl` files directly.
`docker compose down` leaves the arms running; stop one with `docker compose stop paper-h30`.
Compare the two arms' equity and trades after a few weeks.
