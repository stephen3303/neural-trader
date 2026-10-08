# neural-trader

A scaffold for a neural network that monitors the market continuously and
retrains itself on realized outcomes ("online"/continual learning), wired
to a risk layer and a paper-trading execution layer. It runs end-to-end
today on synthetic or real historical data in **paper trading mode only**.

This is a starting point for your own system, not a finished trading
product — see **Before you even think about live trading** at the bottom.

## Why this architecture

**The core loop.** The network doesn't train once and then just run
inference forever. Every prediction has a built-in expiration date (the
label horizon, default 15 bars ahead): once that many bars pass, the
realized return becomes ground truth, gets logged, and gets added to a
replay buffer. A `ContinualTrainer` periodically fine-tunes a *copy* of the
live model on that buffer and only promotes the copy to "live" if it beats
the current model on a held-out, time-ordered validation slice. That
champion/challenger gate is the single most important safety property in
the whole system: it means a bad batch of data can leave the model
unchanged, but it's much harder for it to actively make the model worse.

**The model.** A 2-layer GRU with learned attention pooling over the
lookback window, feeding two heads: a 3-way sell/hold/buy classifier and a
scalar expected-return regressor. See `src/model/network.py` for the full
reasoning — short version: at these window lengths (60-120 bars) a small
GRU gets Transformer-competitive sequence modeling for a fraction of the
compute, which matters because this model gets retrained continuously, not
once.

**The risk layer is separate from the model on purpose.** `src/risk/manager.py`
never trusts the model's confidence at face value — it volatility-targets
position size, caps it with a fractional-Kelly ceiling (not full Kelly,
because a continually retrained model's self-estimated edge is exactly the
kind of number you shouldn't fully trust), and enforces hard per-position,
per-day, and consecutive-loss limits that trip a kill switch. The kill
switch can only be reset with an explicit `human_confirmed=True` — it will
never silently resume trading on its own.

**A separate drift monitor watches the model from outside.** `src/training/drift.py`
tracks rolling hit-rate, calibration (Brier score), and realized drawdown,
and can recommend a halt faster than a retrain cycle could react — this is
what catches a regime shift or a quietly-degrading model between retrains.

**Everything is logged with full provenance.** `src/monitor/logger.py`
writes one JSON line per prediction, risk decision, fill, outcome, and
retrain event, each linked by a `decision_id` — so any order can be traced
back to exactly what the model saw, how confident it was, what the risk
layer decided, and what actually happened.

## Project layout

```
src/
  data/
    feed.py         # abstract MarketDataFeed + SyntheticFeed + YFinanceFeed
    alpaca_feed.py  # AlpacaLiveFeed: real-time bars via Alpaca's websocket
    features.py     # technical-indicator feature engineering + forward labels
    orderbook.py           # L2 order book: OrderBookFeed + Synthetic/AlpacaCrypto/EquityL2Stub
    orderbook_features.py  # spread/imbalance/microprice/depth features from order book snapshots
    buffer.py       # ring-buffer replay buffer for continual learning
  model/
    network.py    # TradingNet (GRU encoder + attention + dual heads)
    signals.py    # raw model output -> Signal (ticker, action, confidence, ...)
  training/
    trainer.py    # ContinualTrainer: champion/challenger retraining loop
    drift.py       # DriftMonitor: live hit-rate / calibration / drawdown
  risk/
    manager.py    # RiskManager: position sizing + kill switch
  execution/
    broker.py         # Broker interface + PaperBroker + LiveBrokerStub (unimplemented)
    alpaca_broker.py  # AlpacaBroker: real paper-trading fills via Alpaca (refuses paper=False)
  monitor/
    logger.py      # DecisionLogger: structured JSONL decision trail
  orchestrator.py  # the main loop tying all of the above together
  config.py        # loads config.yaml into the dataclasses above
scripts/
  backtest.py           # offline supervised sanity check before running the online loop
  run_paper_trading.py  # continual-learning loop against synthetic/historical data
  run_live_alpaca.py    # continual-learning loop against REAL-TIME Alpaca data, paper orders only
  serve_dashboard.py    # local server: auto-refreshing dashboard while a loop above is running
  premarket_check.py    # verifies today's a trading day + (re)starts the two scripts above if needed
  premarket_check.bat   # Task Scheduler entry point for premarket_check.py
  stop_trading.py        # stops run_live_alpaca.py / serve_dashboard.py if they're running
  stop_trading.bat       # double-clickable entry point for stop_trading.py
tests/                  # unit tests for the model, risk layer, trainer checkpointing,
                        # orchestrator wiring, Alpaca broker safety guard, and order book
config.yaml             # all tunable parameters in one place
.env.example            # template for ALPACA_API_KEY / ALPACA_SECRET_KEY
```

**A correctness bug fixed in `orchestrator.py` worth knowing about if you
ran this before this fix:** `Orchestrator` used to call `self.model.predict(...)`
on a reference to the model captured once at construction time. But
`ContinualTrainer.maybe_retrain()` promotes a challenger by *rebinding* its
own `self.model` attribute (`self.model = challenger`), not by mutating the
existing model object in place. The effect: after the very first promoted
retrain in any run, `Orchestrator`'s predictions silently kept coming from
the stale, pre-promotion model forever — while the decision log correctly
reported an incrementing `model_version` the whole time, making it look
like retraining was working. It's now fixed to always read
`self.trainer.model` at prediction time (see the comment in
`_try_predict()`), with a regression test
(`tests/test_orchestrator.py::test_predictions_use_the_most_recently_promoted_model`)
that forces a promotion and asserts the next prediction reflects the new
model. If you have historical logs from before this fix, treat any
model-version-vs-behavior comparisons in them with that in mind — the
model backing a prediction may not match what the log's `model_version`
field implied.

## Running it

```bash
pip install -r requirements.txt

# 1. Sanity check: can the architecture learn anything at all on this data?
python scripts/backtest.py --ticker SPY

# 2. Run the full continual-learning + paper-trading loop (synthetic data,
#    no network required):
python scripts/run_paper_trading.py --max-bars 2000

# 3. Same, but against real recent history via yfinance:
python scripts/run_paper_trading.py --feed yfinance --max-bars 2000

# 4. Real-time Alpaca data, still paper trading -- see the section below
#    before running this one:
python scripts/run_live_alpaca.py

# Unit tests (requirements-dev.txt adds pytest on top of requirements.txt)
pip install -r requirements-dev.txt
pytest tests/
```

The full suite also runs automatically on every push/PR via GitHub Actions
(`.github/workflows/tests.yml`) -- CPU-only, no Alpaca credentials or network
access needed, since the tests use `SyntheticFeed`/`PaperBroker` throughout.
That suite includes `tests/test_gross_errors.py`: an actual end-to-end run
of `Orchestrator`+`RiskManager`+`ContinualTrainer`+`PaperBroker` (small
scale, ~10s), asserting sanity bounds a healthy system should never
violate -- finite/positive equity throughout, no catastrophic equity
collapse, every realized trade's P&L finite and within the configured
position cap, consistent kill-switch state, well-formed retrain records,
and full run-to-run reproducibility with the same seed. Every real bug
fixed in this project was found by actually running the pipeline, not by
reasoning about functions in isolation -- this makes that check run on
every future push automatically instead of depending on someone
remembering to do it by hand.

`run_paper_trading.py` prints final equity, number of fills, how many
retrain attempts were promoted, the drift monitor's final snapshot, and
whether the kill switch tripped. The full decision trail is in
`logs/decisions.jsonl`.

## Watching it live: the dashboard

`dashboard.html` is a self-contained monitoring UI (equity curve, a
**performance summary** panel, rolling hit-rate/calibration, retrain
history, per-ticker signal traces, a searchable decision feed). You can
always open it and drag `decisions.jsonl` onto the page by hand, but
while a loop above is actively running, it's more useful with
auto-refresh:

```bash
# In a second terminal, alongside run_paper_trading.py / run_live_alpaca.py:
python scripts/serve_dashboard.py
```

This opens the dashboard in your browser already watching `logs/decisions.jsonl`
and polls it every few seconds, so fills, retrains, and halts show up on
their own as the trading script writes them -- no manual re-drop needed. A
"LIVE" badge appears in the header when it's working. It's read-only in
both directions: this server only ever reads the log file, never writes to
it, and has no connection to the trading process -- stopping it (Ctrl+C)
doesn't affect `run_live_alpaca.py` or `run_paper_trading.py` at all, and
vice versa. If you ever open `dashboard.html` by itself (double-click, or
as a published artifact) instead of through this server, it just falls back
to the drag-and-drop flow, same as before.

Options: `--log <path>` to point at a different log file, `--port <n>` if
8787 is taken, `--no-open` to skip auto-opening a browser tab.

**Performance summary panel (task #17).** Right below the equity curve,
the dashboard now shows Sharpe ratio, max drawdown, profit factor, win
rate, and turnover, plus a per-ticker breakdown table (trades, win rate,
profit factor, total P&L) — now that 12 tickers are tracked instead of
3, scanning per-ticker charts one at a time stopped being a practical way
to see which symbols were actually carrying the strategy. These numbers
are computed **live, in the browser, directly from whatever log is
loaded** — no new files, no server round-trip, no new dependencies — using
the exact same formulas as `src/analysis/metrics.py` (see
`realizedTrades`/`sharpeRatio`/`profitFactor`/etc. in `dashboard.html`'s
script, deliberately written to mirror that module line-for-line). That
means a live/paper-trading log opened in the dashboard and an offline
`scripts/backtest_portfolio.py` report of the same data always agree —
verified directly: running the backtest harness and then loading its own
`--log-path` output into the dashboard in a headless browser produced
**identical** numbers (Sharpe 0.257, max drawdown 0.0%, profit factor
2.452, win rate 66.7%, same 3 realized META trades) to the script's
printed report for that same run. Sharpe's annualization factor
(`periods/yr`, default 252 for daily bars) is a dropdown next to the
panel, matching `backtest_portfolio.py`'s `--periods-per-year` flag.

**Risk & exposure panel (task #23).** Below the performance summary,
a new panel shows gross portfolio exposure over time against
`max_gross_exposure_pct` (a line chart with the cap drawn as a
reference line), current daily P&L against `max_daily_loss_pct`, and a
per-ticker table of peak exposure reached this run against
`max_position_pct`, with a status pill marking any ticker that reached
its cap. This only became worth building once `open_notional_pct`/
`per_ticker_notional_pct`/`daily_pnl_pct` were actually real numbers
(see the exposure-cap and daily-reset bug fixes above) — before those
fixes there was nothing meaningful to show here, since every one of
those values was either permanently zero or an uncapped running total.
Fed by a new `DecisionLogger.log_risk_state()` row written once per
bar. Verified the same way as the performance panel: loaded a real
backtest log into the dashboard in a headless browser and diffed every
displayed number against the log file's own `risk_state` rows directly
— exact match, including the per-ticker peaks and which tickers showed
a "reached cap" pill.

Building this panel is also what surfaced the mark-to-market-drift
correction documented above (see "A sixth real bug..."): the panel
reads live, per-bar, mark-to-market exposure, which is a more faithful
picture than a one-off ad hoc verification script happens to be.

## Live data, paper trading (Alpaca)

`run_live_alpaca.py` runs the exact same continual-learning loop, but
`AlpacaLiveFeed` (`src/data/alpaca_feed.py`) replaces the synthetic/historical
feed with real-time minute bars from Alpaca's websocket, and `AlpacaBroker`
(`src/execution/alpaca_broker.py`) places real orders against your **Alpaca
paper account** — simulated fills from a real exchange, not simulated prices.
No real money is at risk; `AlpacaBroker` refuses to construct at all unless
`paper=True`, and that isn't a config flag, it's a guard in the code.

**Setup:**

1. `pip install alpaca-py python-dotenv` (both are in `requirements.txt`).
2. Get your **paper** API key/secret from the Alpaca dashboard — it shows
   separate keys for paper and live accounts, so double-check which you're
   copying.
3. Copy `.env.example` to `.env` and fill in `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY` (or just `export` them in your shell).
4. `python scripts/run_live_alpaca.py`. It waits for market open if the
   market's closed, then runs until market close or you Ctrl+C it.

**What's different from synthetic/historical paper trading:**

- Data is real-time IEX data (`DataFeed.IEX`, Alpaca's free tier — one
  exchange's tape, not the full consolidated SIP feed, so prices can lag or
  differ slightly from what a broker's own chart shows; fine for testing
  the pipeline, worth knowing before reading too much into small
  discrepancies). Switch `alpaca.data_feed` in `config.yaml` to `sip` if
  you have a paid plan.
- Orders really do round-trip through Alpaca's matching engine; fills come
  back as real `filled_avg_price`/`filled_qty`, not a slippage model.
  Alpaca is commission-free for US equities, so `commission` on every fill
  is `0.0` here (unlike `PaperBroker`'s configurable bps estimate).
- The loop pauses outside market hours instead of erroring when the
  stream goes quiet overnight, and stops cleanly at market close.
- Tickers and the data feed tier are set in `config.yaml` under `tickers:`
  and `alpaca:` — `risk.account_equity` is just the *starting* assumption
  now (see below for why it no longer stays frozen there).

**Position sizing now tracks your real account balance, not a frozen
config value.** `risk.account_equity` in `config.yaml` used to be a
static number position sizing converted `size_pct_equity` into dollars
with — and it never changed again after startup, no matter how much the
real Alpaca balance moved. `AlpacaBroker.get_equity()` was already
querying Alpaca's real, current equity every bar (for the dashboard/drift
monitor), that value just never made it back into the risk manager.
Fixed: `Orchestrator` now calls `RiskManager.update_account_equity()`
with the broker's real equity at the start of every bar, before that
bar's sizing decision — so a `size_pct_equity` of, say, 5% is always 5%
of your *actual* current balance, compounding gains/losses correctly
instead of drifting from a number frozen at whatever it was when the
process started. A single bad/transient equity reading (NaN, zero,
negative, a dropped API call) is ignored rather than corrupting sizing —
it just keeps using the last known-good value.

**Model weights now survive a restart.** On startup, `run_live_alpaca.py`
automatically loads the most recently *promoted* checkpoint from
`checkpoints/` (whatever `trainer.checkpoint_dir` in `config.yaml` points
at) via `ContinualTrainer.load_latest_checkpoint()`, instead of always
starting from an untrained `v0` model — it prints which version it resumed
(or that it found none, on a genuinely first run). This closes a real bug:
`ContinualTrainer._save_checkpoint()` has always written a `.pt` file on
every promotion, but until now nothing ever loaded one back, so every
restart silently threw away all continual learning progress while the logs
kept reporting an incrementing `model_version` from a fresh session.
`run_paper_trading.py` does *not* resume by default (repeated smoke-test
runs should stay reproducible) — pass `--resume` to opt in there.

**Risk/drift state now survives a restart too.** `run_live_alpaca.py`
loads `checkpoints/risk_state.json` and `checkpoints/drift_state.json` on
startup (printing what it resumed) via `RiskManager.load_state()` /
`DriftMonitor.load_state()`, and `Orchestrator` re-saves them after every
state-changing event for the rest of the run — not just on clean
shutdown, so a crash loses at most one event's worth of state, not the
whole day. Concretely: today's daily P&L, the current consecutive-loss
streak, and — most importantly — **an already-tripped kill switch** all
survive a restart now, along with the drift monitor's rolling hit-rate/
calibration/equity windows. Verified with a real run: tripped the kill
switch via a bad hit-rate, restarted with fresh in-memory objects, and
confirmed `trading_enabled` loaded back as `False` with the original halt
reason intact rather than silently re-enabling trading.

**Still a known limitation — read before leaving this running unattended:**
the replay buffer itself still lives in memory only (it already has
`.save()`/`.load()` in `src/data/buffer.py` if you want to wire that up
too). If the process restarts mid-session, the model's weights and the
risk/drift state above all resume correctly, but the buffer of raw
samples waiting for the next retrain starts empty again.

## Automated pre-market check (Windows)

`scripts/premarket_check.py` + `scripts/premarket_check.bat` exist so you
don't have to remember to start `run_live_alpaca.py` and
`serve_dashboard.py` by hand every trading morning. The check:

1. Asks Alpaca's own market calendar whether today is actually a trading
   day — this correctly skips weekends *and* market holidays (Thanksgiving,
   July 4th, etc.) with no hardcoded holiday list to maintain. On a
   non-trading day it logs that and exits; nothing else happens.
2. On a real trading day, checks whether `run_live_alpaca.py` and
   `serve_dashboard.py` are already running (by command line, via WMI —
   `python.exe` alone isn't enough to tell them apart) and starts whichever
   one isn't, detached so it keeps running after the check script exits.

It does **not** place trades or touch any model/risk state itself —
starting `run_live_alpaca.py` 30 minutes early just means it sits there
printing "market closed..." until 9:30 like always.

**Set it up to run automatically ~30 minutes before the 9:30am ET open:**

1. `pip install tzdata` (in `requirements.txt` — Windows has no built-in
   timezone database, which the trading-day check needs).
2. Test it by hand first: `python scripts\premarket_check.py` from the
   project folder. Confirm it logs correctly and (outside market hours)
   doesn't start anything it shouldn't.
3. Open Task Scheduler → Create Task, or from a **PowerShell** terminal
   (not `cmd.exe` — use PowerShell's own `ScheduledTasks` module, not the
   older `schtasks.exe` command-line tool. `schtasks.exe` has a
   long-standing quirk where it splits a path at the first space into
   "program" + "arguments" unless you wrap it in a second, nested pair of
   literal quote characters — and a path like this one, with a space in
   "Neural Trader", hits that every time. The error looks like `Task
   Scheduler failed to launch action "C:\Users\steph\Documents\Neural"`
   (truncated at the space) with error value `2147942402` (file not
   found). The `ScheduledTasks` cmdlets below take the path as a normal
   parameter instead, so this doesn't come up):
   ```powershell
   $Action  = New-ScheduledTaskAction -Execute "C:\Users\steph\Documents\Neural Trader\scripts\premarket_check.bat"
   $Trigger = New-ScheduledTaskTrigger -Daily -At 9:00AM
   Register-ScheduledTask -TaskName "NeuralTrader PremarketCheck" -Action $Action -Trigger $Trigger -Description "Starts/checks the neural-trader scripts ~30 min before market open."
   ```
   `-At 9:00AM` assumes your Windows clock is set to Eastern time — if it
   isn't, use 9:00am Eastern converted to your local time instead.
   `-Daily` (every day, not just weekdays) is intentional: the
   trading-day check above is what actually filters out weekends/holidays,
   so the schedule itself can stay simple.
4. If you want it to run even when you're not logged in, open the task's
   Properties → General and check "Run whether user is logged on or not"
   (you'll be prompted for your password once, to store it).
5. If a scheduled run fails with `'python' is not recognized`, open
   `premarket_check.bat` and hardcode the full path to your `python.exe`
   in the `PYTHON_EXE` variable at the top (find it with `where python`) —
   a task that runs while logged out doesn't load your normal PATH.

**If the task's Action ends up wrong** (truncated path, error `2147942402`
in the task's History/"Last Run Result"): fix just the Action in place,
same idea as above, no need to delete and recreate:
```powershell
$Action = New-ScheduledTaskAction -Execute "C:\Users\steph\Documents\Neural Trader\scripts\premarket_check.bat"
Set-ScheduledTask -TaskName "NeuralTrader PremarketCheck" -Action $Action
```

Every run appends to `logs\premarket_check.log`, and anything it starts
logs to its own `logs\run_live_alpaca_stdout.log` / `logs\serve_dashboard_stdout.log`
— check those first if a morning run doesn't look right.

**Shutting it down.** If you're looking at the terminal window either
script is running in, Ctrl+C there is the cleanest stop — `run_live_alpaca.py`
catches it and prints a session summary (final equity, fills, drift
snapshot, kill switch state) before exiting. If a script was started
detached (by `premarket_check.py`, which has no visible window) or you just
don't want to hunt down the window, double-click `scripts\stop_trading.bat`
(or run `python scripts\stop_trading.py`) — it finds and stops both
`run_live_alpaca.py` and `serve_dashboard.py` by process command line. That's
a hard stop, not a graceful one: no session summary, and whatever was only
in memory (replay buffer, today's risk counters, drift windows) is gone,
same as a crash — expected for paper trading, see the known limitation
above. It never touches your Alpaca paper account itself.

To stop the scheduled task from firing again tomorrow (without deleting its
history): `Disable-ScheduledTask -TaskName "NeuralTrader PremarketCheck"` —
`Enable-ScheduledTask` to turn it back on later, or
`Unregister-ScheduledTask -TaskName "NeuralTrader PremarketCheck" -Confirm:$false`
to remove it entirely.

## Order book (Level 2) features

`src/data/orderbook.py` and `src/data/orderbook_features.py` add an L2
order-book pipeline, independent of the per-bar OHLCV features in
`features.py`. Read this before wiring it into anything, because what's
real here is not what you might assume:

**Alpaca does not provide Level 2 depth for equities.** Checked directly
against the installed `alpaca-py` SDK while building this: `StockDataStream`
only has `subscribe_quotes` (top-of-book NBBO, i.e. Level 1), not an order
book subscription. So for SPY/QQQ/AAPL — what this project actually
trades — there is no real L2 feed available through Alpaca at all, at any
tier. That's a market-data-vendor limitation, not something more code
here can work around.

What IS real: Alpaca's **crypto** data API genuinely streams full L2 books
(`CryptoDataStream.subscribe_orderbooks`), so `AlpacaCryptoOrderBookFeed`
is a working integration — useful if this project ever trades crypto
pairs, not useful for the equities it trades today.

So, three classes in `orderbook.py`, three different levels of "real":
- `SyntheticOrderBookFeed` — deterministic synthetic L2 book generator
  (geometric mid-price walk, spread that widens with volatility, depth
  that decays away from the touch, a random bid/ask size imbalance). This
  is what lets you build and test the feature pipeline today, same spirit
  as `SyntheticFeed` in `feed.py`.
- `AlpacaCryptoOrderBookFeed` — real, for crypto pairs only.
- `EquityL2FeedStub` — `NotImplementedError` placeholder, same pattern as
  `LiveBrokerStub`. Its docstring lists actual vendor options for equities
  L2 (Polygon.io, Databento, IEX Cloud DEEP, direct exchange feeds) and
  what a real integration has to add that the synthetic feed skips
  entirely: binary protocol parsing, book reconstruction with
  sequence-gap detection, snapshot/delta reconciliation on reconnect, and
  per-symbol data costs that aren't small at any real scale.

`orderbook_features.py` turns a stream of snapshots (one ticker at a time,
same per-ticker contract as `compute_features()`) into six causal,
stationary features: `spread_bps`; `obi_l1` and `obi_l5` (order book
imbalance — `(bid_size - ask_size) / (bid_size + ask_size)` at the top
level and across the top 5); `microprice_dev_bps` (the classic
size-weighted microprice vs. the simple mid, in basis points — a
short-term price-pressure signal); and `depth_bid_z` / `depth_ask_z`
(rolling z-scored depth, so the network sees "thinner/thicker liquidity
than usual" rather than a raw, cross-ticker-incomparable share count).

This is **not wired into the model or the live trading loop** — it's a
self-contained, tested module, not an automatic change to `n_features` or
`FEATURE_COLUMNS`. To actually use it: call `compute_orderbook_features()`
per ticker alongside `compute_features()`, `pd.concat` the two feature
frames, bump `n_features` in `config.yaml`'s `model:` section to match,
and retrain — same mechanism `features.py`'s own docstring already
describes for adding any new column. Doing that for real equities trading
still needs a real L2 vendor first (see above); for crypto, or for
testing the pipeline today, the synthetic feed is enough.

## A real model-collapse bug, found and fixed

While working on long-term profitability, I ran the actual online loop
(not a demo/sample log — `ContinualTrainer.maybe_retrain()` against real
synthetic data) for 3000 bars and found the continual-retraining path was
silently promoting a broken model almost every cycle:

- **"Hold" was never learned.** The forward-return deadband (`label.deadband_bps`)
  makes "hold" a small minority of labels (under ~3% on the default
  config) — unweighted cross-entropy gave the model essentially no reason
  to ever predict it. Across 13 promoted retrains, "hold" was predicted
  exactly once (by the untrained v0 model) and zero times afterward.
- **Within any one model version, predictions frequently collapsed to a
  single action** — 100% sell for an entire version, then 100% buy the
  next, etc. — and the champion/challenger loss-regression gate didn't
  catch it: a collapsed challenger can still post a loss that's "not
  meaningfully worse" than the champion's by chance, especially under
  label imbalance, so **13 out of 13 retrains got promoted** with no
  resistance at all.

Two independent fixes, both in `src/training/trainer.py` (`TrainerConfig`
/ `ContinualTrainer`), both covered by unit tests in `tests/test_trainer.py`:

1. **Inverse-frequency class weighting** (`_class_weights`, config:
   `trainer.use_class_weights` / `trainer.max_class_weight`) — computed
   fresh from the buffer on every retrain, normalized to average 1.0 so
   the loss scale doesn't drift, and clipped so a near-empty class can't
   be dominated by a handful of noisy samples.
2. **A second, independent promotion gate** (`_degenerate_prediction_reason`,
   config: `trainer.max_degenerate_action_frac`, default 0.97) — vetoes
   promotion outright if the challenger predicts one action on almost the
   entire validation set, regardless of how its loss compares to the
   champion's. This is deliberately a *different kind* of check than the
   loss gate (checks the shape of the predictions, not their loss) so the
   two don't share the same blind spot.

Re-running the identical 3000-bar scenario after the fix: **1 of 13**
retrains got promoted, and the other 12 were correctly vetoed with a
logged reason (e.g. `predicted action 2 on 100.0% of the validation set`).
That's the fix working as intended — it does **not** mean the model now
has a real trading edge, only that the system stopped rubber-stamping
collapsed models as improvements. Whether this architecture can learn a
genuine edge at all is a separate, open question (see the deadband/
feature-richness discussion above and the disclaimer at the bottom of
this README) — the responsibility of these two gates is narrower and
more load-bearing than that: never silently make the live model *worse*
in a way nothing else would catch.

## A second real bug: the daily-loss / consecutive-loss kill switch was dead code

Found immediately after the collapse-bug fix above, while starting work
on persisting risk state across restarts: `RiskManager.update_after_trade_result()`
— the function `cfg.max_daily_loss_pct` and `cfg.max_consecutive_losses`
depend on — was **never called anywhere in `Orchestrator`'s main loop**.
It was only ever exercised by `tests/test_risk.py`'s direct unit tests of
`RiskManager` in isolation. In real trading this meant those two
kill-switch conditions could never trip, no matter how badly a session
went — only `DriftMonitor`'s separate hit-rate/Brier/equity-drawdown halt
(which *is* wired into the loop) was ever actually live.

Fixed in `src/orchestrator.py`: each pending prediction now carries the
`size_pct_equity` it was actually sized and filled at (`0.0` if the
signal was hold/low-confidence/risk-capped/rejected by the broker — never
counted as a realized trade), and when that prediction matures,
`_resolve_matured()` feeds its realized P&L (`size_pct_equity * direction
* realized_return`) into `risk_manager.update_after_trade_result()`. 6 new
regression tests in `tests/test_orchestrator.py`
(`TestRiskManagerReceivesRealizedPnl`) cover: a losing filled trade moving
`daily_pnl_pct`, a winning trade resetting the consecutive-loss streak, an
unsized/unfilled prediction correctly *not* touching risk state, repeated
losses actually tripping the kill switch through the real loop path (not
just the isolated unit test), and the `run()`-level plumbing that decides
`size_pct_equity` based on whether the broker actually returned a `Fill`.

## A rigorous multi-ticker backtest harness, and two more real findings

`scripts/backtest.py` (single ticker, naive train/val split, no costs) was
useful as a quick "can the architecture learn anything at all" sanity
check, but it's not a backtest of the actual strategy: it doesn't use
the risk manager, the kill switch, position sizing, broker costs, or the
continual-retraining loop. `scripts/backtest_portfolio.py` runs the
**real** production code path — the same `Orchestrator`, `RiskManager`,
`ContinualTrainer`, and `PaperBroker` (with its slippage/commission cost
model) that `run_paper_trading.py` uses — across every ticker in
`config.yaml` simultaneously, sharing one capital pool and one replay
buffer exactly like live trading would. "Walk-forward" isn't a separate
mode here: the online/continual-learning loop *is* walk-forward by
construction (predict using only data up to the current bar, learn from
it once its outcome matures later), so running the real pipeline over
historical data already is the walk-forward backtest.

It reports Sharpe ratio, max drawdown, profit factor, win rate, and
turnover (`src/analysis/metrics.py`, every formula checked against
hand-computed values in `tests/test_metrics.py`), plus a per-ticker
breakdown. `Orchestrator` grew two new attributes to support this:
`equity_curve` (full-resolution, unbounded — `DriftMonitor`'s own equity
window is deliberately bounded for live monitoring and can't answer "what
was the Sharpe over this whole multi-thousand-bar run") and `trade_log`
(one entry per *realized, actually-filled* trade, carrying the exact same
`pnl_pct_of_equity` value already fed into `risk_manager.update_after_trade_result()`,
so the backtest report and the risk manager's own bookkeeping can never
disagree about which trades counted).

Building it surfaced two more real issues, neither of them in the metrics
math itself:

**1. Results weren't reproducible run-to-run.** Running the exact same
backtest twice against the identical, deterministic `SyntheticFeed` data
gave wildly different outcomes — one run tripped the kill switch almost
immediately and traded zero times; another ran "hold or low confidence"
on literally every single bar. The cause: nothing in the codebase ever
seeded `torch`'s global RNG, which governs `TradingNet`'s weight
initialization and dropout — so every run started from a different
random model. Separately, `ContinualTrainer.maybe_retrain()` created a
*brand-new* `np.random.default_rng()` (seeded from OS entropy, immune to
any global seeding) on every single retrain call, so even a seeded model
would still retrain on a different random minibatch sequence each time.
Fixed with `src/utils.py:seed_everything()` (seeds Python/numpy/torch
global RNG, called once at startup) and `TrainerConfig.seed` (a
persistent `ContinualTrainer._rng`, reused across every retrain instead
of recreated per-call). `scripts/backtest_portfolio.py --seed 42`
(the default) now produces byte-for-byte identical reports across runs —
verified directly: two consecutive runs' full printed summaries diffed
as identical. See `tests/test_utils.py` and
`TestSeededReproducibility` in `tests/test_trainer.py`.

**2. The hit-rate kill switch can trip during cold start and never
recover — by design, which is exactly the problem for an unattended
backtest.** `RiskManager.reset_kill_switch()` deliberately requires
`human_confirmed=True` and is never called automatically by the trading
loop (correct and intentional for live trading — see its docstring). But
`DriftMonitor`'s hit-rate/Brier/drawdown halt can legitimately fire while
a model is still cold-starting (before it's had a real chance to learn —
most of its early retrains get vetoed as collapsed, see the bug above),
and once it trips `RiskManager`'s kill switch, nothing in a backtest
(there's no human present to click reset) ever un-trips it. A single bad
early patch therefore makes the *entire rest* of a multi-thousand-bar
backtest report "0 trades," which says nothing about how the strategy
performs once the model is actually trained. `scripts/backtest_portfolio.py
--ignore-kill-switch` is a narrow, clearly-labeled escape hatch for this:
it monkeypatches *this script's own* `risk_manager.trip_kill_switch` to
record each trip and immediately reverse it
(`reset_kill_switch(human_confirmed=True)`), so the backtest keeps
accumulating real trade/equity history past it. It touches no file in
`src/` and is off by default — without it, the report reflects exactly
what live/paper trading would actually do (fails closed, stays closed).
See `install_kill_switch_auto_reset()`'s docstring in the script and
`TestInstallKillSwitchAutoReset` in `tests/test_backtest_portfolio.py`.
This is a real operational risk worth knowing about before trusting this
system unattended: **a freshly-deployed model may need a human to
manually reset the kill switch once, early in its life**, after its
initial cold-start learning period — it will not recover on its own.

```
python scripts/backtest_portfolio.py                         # synthetic, all 12 tickers, seed=42
python scripts/backtest_portfolio.py --feed yfinance         # real recent history
python scripts/backtest_portfolio.py --ignore-kill-switch     # see full-period metrics (backtest only)
```

### An open question, deliberately not acted on: class weights vs. what's actually trained on

While investigating why action confidence tends to cluster tightly near
`min_confidence` early in a run, I noticed that
`ContinualTrainer._class_weights()` computes its inverse-frequency class
weights from the **entire** replay buffer, but `ReplayBuffer.sample_batch()`
actually trains each retrain on a **blend** (`recent_fraction=0.6` by
default) of the most-recent `recent_window_frac=0.15` slice and a
uniform draw from the whole buffer. If the recent slice's class balance
differs meaningfully from the full buffer's (plausible after a regime
shift, or once the buffer wraps), the weights meant to counteract
whatever imbalance the model is actually training on are measured
against the wrong distribution.

Measuring this on a real 1500-bar/12-ticker run, the divergence turned
out to be modest (weights differed by roughly 20-30% on the minority
classes, not an order of magnitude) — not clearly the dominant cause of
anything currently broken, and the existing two-gate promotion check
(loss regression + the degeneracy veto from the model-collapse fix
above) already catches a badly-collapsed challenger regardless of how
well-calibrated its training weights were. I prototyped a fix (blending
`_class_weights()`'s counts the same way `sample_batch()` blends its
draws) but it changes which labels "rare" means relative to in a way
that flips the ordering in at least one existing hand-computed test
(`TestClassWeights::test_rare_class_gets_a_higher_weight_than_common_classes`),
and I don't have strong enough evidence this is actually a net
improvement to training quality rather than just a different (not
better) distributional assumption — that needs a real before/after
backtest comparison with `scripts/backtest_portfolio.py`, across more
than one seed, which is more validation than a judgment call like this
should get made without. Flagging it here rather than shipping it
unilaterally: this is a real design question worth deciding on
deliberately, not a clear bug like the others in this section.

## A third and fourth real bug: two more dead-code safety checks, found by auditing every public `RiskManager` method against what `Orchestrator` actually calls

The kill-switch bug above was found by noticing one specific function was
never called. That suggested a general audit was worth doing: for every
public method on `RiskManager` (the layer that matters most for eventual
live trading), is it actually invoked anywhere in `Orchestrator`/
`scripts/*.py`, or only ever exercised by a unit test calling it directly
in isolation? Running that check (independently, twice) turned up two
more real instances of exactly this bug class.

**1. `reset_daily_counters()` was implemented, unit-tested, and never
called.** `state.daily_pnl_pct` — the number `cfg.max_daily_loss_pct`'s
kill-switch check compares against — was therefore an **all-time
cumulative total that never reset**, not a daily one, despite its name
and despite being persisted/resumed across restarts via
`risk_state_path`. Two concrete ways that breaks the "daily" loss limit:
a genuinely bad day could fail to trip the kill switch at all if prior
days were net positive enough to keep the cumulative total above
`-max_daily_loss_pct`; or, the other direction, a few bad days already
baked into the cumulative total could make the kill switch permanently
untrippable-as-designed or trip on an otherwise fine day once the running
total creeps past the threshold on its own. Either way, `max_daily_loss_pct`
silently stopped meaning what its name says.

Fixed with `Orchestrator._maybe_reset_daily_counters(timestamp)`: detects
a calendar-day boundary crossing in the bar timestamp stream
(`pd.Timestamp(timestamp).date()`, which `SyntheticFeed`/`YFinanceFeed`/
`AlpacaLiveFeed` all produce consistently) and calls
`risk_manager.reset_daily_counters()` — then persists the reset via
`save_state()` when a `risk_state_path` is configured, so a restart right
after midnight can't un-reset it. Wired into `run()`'s main loop right
after `_resolve_matured()`, once per bar. 5 new regression tests in
`tests/test_orchestrator.py` (`TestDailyCounterReset`) cover: the very
first bar ever seen correctly not triggering a reset (nothing to compare
against yet), same-day bars leaving the counter untouched, a day-boundary
crossing resetting it to exactly zero, the reset surviving a save/load
round trip, and the real `run()` loop actually calling it.

**2. `RiskState.open_notional_pct` was never written anywhere.**
`size_order()`'s portfolio-wide gross-exposure cap
(`cfg.max_gross_exposure_pct`, default 60%, meant to bound aggregate open
position notional across *every* ticker combined) checks this value —
but it was initialized to `0.0` and nothing in the codebase ever updated
it, not even a test. That makes the check `open_notional_pct >=
max_gross_exposure_pct` permanently unable to fire (`0.0` is never `>=` a
positive cap), and the headroom calculation
(`max_gross_exposure_pct - open_notional_pct`) always evaluates to the
full, uncapped limit. With enough tickers each independently sized up to
`max_position_pct` (the *per-ticker* cap, which this bug never affected
and which still worked correctly on its own), aggregate exposure across
the whole portfolio could exceed the configured limit with nothing to
stop it — a hard leverage limit providing zero actual protection.
Verified directly on the real pipeline (8 tickers, `max_position_pct=20%`,
`max_gross_exposure_pct=25%`): before the fix, `open_notional_pct` stayed
exactly `0.0` for the entire run regardless of how many of 271 trades
filled; after it, the value is real and tracked (final 21.9%, max
observed 26.45% against the 25% cap — the brief overshoot above the cap
is expected mark-to-market drift of *already-open* positions between
bars, not a sizing bug: `size_order()`'s headroom formula correctly
bounds every *new* trade at the moment it's sized, but can't retroactively
shrink a position that's already open and has since moved).

Fixed with `RiskManager.update_open_exposure(open_notional_pct)` (same
defensive non-finite/negative-value handling as `update_account_equity`,
but — unlike that method — `0.0` is a perfectly normal reading here, no
open positions, and must not be treated as a bad one) and
`Orchestrator._gross_exposure_pct(mark_prices, equity)`, which sums
`abs(broker.get_position(t).quantity) * mark_price` across every ticker
and expresses it as a percent of current equity. Wired into `run()`
right after the existing `update_account_equity(equity)` call, so sizing
for *this* bar sees exposure computed from *this* bar's mark prices. For
`AlpacaBroker` this does mean one additional real API call per ticker per
bar (on top of the `get_equity()` call already made every bar) — judged
acceptable given Alpaca's ~200 req/min rate limit comfortably covers a
double-digit-ticker portfolio on a ≥1-minute bar cadence, but worth
knowing about before scaling ticker count up substantially further. 11
new regression tests total: `TestUpdateOpenExposure` in
`tests/test_risk.py` (6 — a valid reading is stored; the cap now actually
blocks a new order once breached, which was impossible to exercise before
this fix; sizing is capped to *remaining* headroom, not the full cap;
negative/non-finite readings are ignored; zero is accepted, not treated
as a bad reading) and `TestGrossExposureTracking` in
`tests/test_orchestrator.py` (5 — zero with no open positions; a long
position's notional marked to the current price, not the stale entry
price; falling back to entry price when no mark price is known yet;
a short position contributing its absolute notional; zero/negative
equity returning zero instead of dividing by it; and the real `run()`
loop actually feeding a nonzero value to the risk manager).

Both findings came from the same methodology, applied twice independently
(once by a delegated review pass, once by directly grepping every
`RiskManager` public method against `src/orchestrator.py`/`scripts/*.py`)
and cross-checked against each other before fixing anything — the kind
of systematic "is this safety check actually wired in, or just tested in
isolation" pass worth repeating any time a new one gets added here.

## A fifth real bug, found by the same technique applied to `size_order()`'s own return value: the hard stop-loss was never enforced

Running that same "is this safety check actually wired in" audit one
level deeper — not just on every `RiskManager` *method*, but on every
*field* `size_order()` returns — turned up one more: `stop_loss_pct`.
`RiskConfig.hard_stop_loss_pct` (default 3.0%, documented right in the
dataclass as "per-position stop loss") gets computed and attached to
every sized order, but grep confirmed nothing downstream ever read it.
A filled position was only ever closed when its prediction *matured* —
a full `label_cfg.horizon` bars later (15 by default) — at whatever
price the bar happened to close at, however far that price had moved
against the position in the meantime. So "hard stop loss" wasn't a
bound on a position's loss at all; it was an unused number sitting in a
dict.

Fixed with `Orchestrator._check_stop_losses(ticker, bar)`, called once
per bar for every ticker, right before `_resolve_matured()` (so a
position breaching its stop and naturally maturing on the same bar
resolves exactly once, via the stop, not twice). For every still-open,
*actually filled* pending position (`size_pct_equity > 0` — an unfilled
hold/low-confidence/risk-capped/rejected prediction has no real position
to stop out of, and must be left alone so its training label still
reflects what actually happened over its full horizon, not a synthetic
early exit), it checks the bar's `low` (long) or `high` (short) — not
just `close` — against the stop level, so a breach-and-recover within
one bar is still caught, matching how a real stop order would have
filled intrabar. A breach exits immediately at the exact stop price via
a new shared `_resolve_one()` helper (factored out of `_resolve_matured`
so the two exit paths — natural maturity and early stop — can never
quietly drift out of sync in how they update the drift monitor, replay
buffer, risk manager, and trade log).

Verified empirically before writing tests: on a 3,000-bar/12-ticker run
with the kill switch forced off (to get a rich trade sample — 230
realized trades), the worst per-position loss, as a fraction of that
trade's own sized notional, was exactly −3.00% — not one trade exceeded
the configured stop, where before this fix a position could and did
ride out losses far larger than that over its full 15-bar horizon. One
direct, visible consequence on the standard 1,500-bar backtest
(`scripts/backtest_portfolio.py --ignore-kill-switch`): realized trades
went from 3 to 6 and the reported win rate/profit factor dropped (two
positions that would previously have ridden out a dip and recovered by
maturity now get cut early at −3% instead) — this is the stop working
as designed, not a regression; a hard per-position loss limit is a
safety property, not a performance guarantee, and this project has
already explicitly chosen "bounded losses" over "maximize backtest P&L"
everywhere else in `RiskManager`. 6 new regression tests in
`tests/test_orchestrator.py` (`TestStopLossEnforcement`) cover: a long
position stopped out by an intrabar low that recovers by the close; the
short-side mirror (intrabar high); a position within the stop band
surviving untouched; an unfilled prediction never being stopped out;
the exactly-once resolution when a breach and a natural maturity land
on the same bar; and `run()`'s loop actually calling the check every
bar.

*Correction, found by a delegated dead-code audit near the end of the
overnight session that first wrote this section, independently
verified before acting on it:* "not one trade exceeded the configured
stop" above is accurate only for the SYNTHETIC bookkeeping this fix
updates (`trade_log`, the risk manager's daily-P&L/consecutive-loss
counters, the training label) -- it is not, as written, a bound on what
actually happens to the real broker position. Both exit paths
(`_check_stop_losses` and natural maturity) computed a realized return
and graded the prediction, but neither one ever submitted a real
closing order to `broker` -- confirmed directly by grep: before the fix
in the section below, `submit_order` had exactly one call site in this
file, and it only ever fired for *new* signals. The real position just
kept sitting open, continuing to drift with the market, past whatever
"stop" the logs said had triggered, until some unrelated future signal
happened to net it down -- which could be never. See "A seventh real
bug" below for the fix and what it actually changes.

## A sixth real bug, one level deeper still: `max_position_pct` capped each order, never a ticker's accumulated position

The stop-loss audit above worked by checking whether a *field*
`size_order()` returns was actually read downstream. Applying the same
question to a field `size_order()` itself *reads* — `max_position_pct`
— turned up one more: it was checked against each new order's own size,
in isolation, every time, but nothing ever checked it against how much
of that ticker was *already held*. A sustained run of same-direction
signals on one ticker — a real trend, which is exactly the condition
this strategy is built to ride — could keep adding to that ticker's
position bar after bar, with each individual order correctly ≤
`max_position_pct` on its own, while the ticker's total accumulated
exposure drifted well past the documented "single position cap, % of
equity." Verified directly on a real 3,000-bar/12-ticker run (kill
switch forced off to get a rich trade sample): before this fix, QQQ's
position reached 17.9% of equity and NVDA's 13.9%, both well past the
configured 10% cap.

Fixed the same way as the portfolio-wide gross-exposure cap:
`RiskState.per_ticker_notional_pct` (a new `dict[ticker, float]`) and
`RiskManager.update_per_ticker_exposure(ticker, pct)`, fed every bar by
`Orchestrator._ticker_notional_pct()` (reads `broker.get_position()` for
just the one ticker about to be sized this bar — cheap, and the only one
that needs a fresh reading right now). `size_order()` now refuses a new
order outright once a ticker is already at its cap ("max position pct
reached for this ticker") and otherwise sizes any new order to that
ticker's *remaining* headroom, not the raw configured cap — identical in
shape to how the gross-exposure fix changed the portfolio-wide check.
Re-running the same 3,000-bar measurement after the fix: QQQ's worst
observed exposure dropped to 10.1% and NVDA's to exactly 10.0%.

*Correction, found while building the dashboard panel below:* that
re-measurement used `Position.avg_price` (cost basis) to compute
notional, not a live mark price — understating the true mark-to-market
drift a real risk system would show, since a position's market value
moves with the price, not with what it was bought at. Re-checked via
the dashboard's "Risk & exposure" panel, which reads the exact same
`per_ticker_notional_pct` values `size_order()` itself enforces (marked
to each ticker's own latest close): on the same run, MSFT's peak was
actually 12.8% and NVDA's 11.3% against the 10% cap — confirmed not a
sizing bug by checking the decision log directly: MSFT's last fill was
over an hour (71 bars) before its peak-exposure reading, so that entire
climb from ~10% to 12.8% was ordinary price appreciation on a position
that hadn't traded at all in the meantime (MSFT ranged \$81–\$106 over
the run). This is the same "a sizing-time cap can't retroactively
shrink an already-open position" limitation as the portfolio-wide cap
above, just a larger real-world magnitude than the avg_price-based
check happened to show — not a new bug, and not something this fix
claims to prevent, but worth stating accurately rather than leaving the
smaller, understated number as the record. Whether to add continuous
position trimming to hold the mark-to-market value itself under the cap
(as opposed to the current entry-time-only enforcement) is a real,
separate design question — a meaningfully different, more complex
behavior change than either exposure-cap fix in this section — and is
intentionally left here as a flagged question rather than something
decided unilaterally overnight, same spirit as the class-weights
question above. `RiskManager.load_state()` reads this new
field with `.get(..., {})` rather than a bare key lookup, so resuming
from a `risk_state.json` saved before this fix existed doesn't raise
`KeyError`. 15 new regression tests total: `TestUpdatePerTickerExposure`
in `tests/test_risk.py` (10 — a valid reading is stored; multiple
tickers tracked independently; the cap now actually blocks a new order
once a ticker is at it; sizing is capped to that ticker's *remaining*
headroom; a different ticker's exposure doesn't affect this one's
sizing; negative/non-finite readings are ignored; zero is accepted; a
save/load round trip; and loading an old state file with no such key at
all doesn't raise) and `TestPerTickerExposureTracking` in
`tests/test_orchestrator.py` (5 — zero with no position; notional marked
to the current price, not the stale entry price; falling back to entry
price with no mark price yet; zero/negative equity returning zero
instead of dividing by it; and the real `run()` loop feeding a nonzero
value to the risk manager).

## A seventh real bug, found by a delegated audit and independently verified before being fixed: stop-loss and maturity resolution never closed the real broker position

Both exit paths in `Orchestrator._resolve_one` -- natural maturity
(`_resolve_matured`) and the hard stop-loss (`_check_stop_losses`,
fixed above) -- updated the drift monitor, the replay buffer, the risk
manager's daily-P&L/consecutive-loss counters, and `trade_log`
identically, every time. What none of that ever did was touch
`broker.positions` or `broker.cash`. Confirmed directly by grep before
writing any fix: `submit_order` had exactly one call site anywhere in
this file, in `run()`'s new-signal path. So a "closed" prediction, by
either exit, was closed only in the system's own bookkeeping -- the
real (paper, and eventually live) position it opened just kept sitting
there, continuing to mark-to-market with the price, until some
unrelated future signal on that same ticker happened to trade in the
opposite direction and net it down. For a ticker that never got another
opposite-direction signal, that position would simply never close.

This was found by a subagent dead-code audit delegated near the end of
the overnight session that wrote the stop-loss fix above, specifically
because that same "grep every call site" technique had already found
two real bugs that session. Per this project's own standing practice
(the system prompt note at the top of this file on verifying delegated
work), the finding was independently re-confirmed via the same greps
before any code changed: one `submit_order` call site, firing only for
new signals; `entry_price` stored as the raw `bar.close`, not the
slippage-adjusted real fill price; and -- an important piece of
context, not itself part of the bug -- `DriftMonitor`'s equity-based
drawdown halt (`record_equity(broker.get_equity(mark_prices))`, wired
correctly since earlier in this session) *is* a real, broker-equity-
based circuit breaker, entirely unaffected by this gap. The per-position
stop-loss and the daily-loss/consecutive-loss kill switch were the
layers actually compromised; the portfolio-wide drawdown halt was not.

Fixed by giving `Broker` a second entry point alongside `submit_order`:
`close_quantity(ticker, action, quantity, ref_price, timestamp)`.
`submit_order` sizes a *new* position by dollar notional (the natural
unit for "put 5% of equity into this ticker") and lets the broker
derive a share count from its own fill price; closing an *existing*
position needs the opposite shape -- an exact, already-known quantity
(whatever the original entry's `Fill` reported), handed to the broker
directly rather than re-derived from a dollar amount that would, after
slippage, only approximately net back out to the right number of
shares. `PaperBroker` and `AlpacaBroker` both implement it (sharing
their existing fill math via a small internal helper each, so
`submit_order` and `close_quantity` can never compute a fill price
differently from each other); `LiveBrokerStub` gets the matching
`NotImplementedError` stub. `_resolve_one` now stores each pending
entry's real filled quantity (`fill.quantity`, 0.0 if the order never
filled) and, whenever a real fill actually happened
(`size_pct_equity > 0`, the same existing gate that decides whether a
trade is "real" everywhere else in this file), submits a closing order
in the opposite direction for exactly that quantity at the exit price
-- the stop level for an early exit, the bar's close for a natural
maturity. `DecisionLogger.log_close_order` records every attempt,
filled or not, so a close that doesn't confirm (the realistic failure
mode for `AlpacaBroker`, whose order can time out waiting for a
terminal status -- see its docstring) is visible in the decision log
instead of silently indistinguishable from one that worked; the
synthetic bookkeeping and training label still proceed either way,
since a prediction's correctness doesn't depend on whether the real
order executed.

Because the FIFO `_pending` queue can hold more than one still-open
entry per ticker at once (a new signal can fire every bar against a
15-bar label horizon), each entry's close nets out only its *own*
contribution -- confirmed with a dedicated regression test that opens
two overlapping entries on the same ticker, resolves only the first,
and checks the second's share of the position is untouched afterward.

Verified empirically, before and after, on the same 3,000-bar/12-ticker
synthetic run (seed 7, kill switch forced off for a rich trade sample --
the same `--ignore-kill-switch` backtest harness used throughout this
file), by running the walk-forward loop directly and reading
`broker.positions` at the end rather than relying only on `trade_log`:

| | before this fix | after this fix |
|---|---|---|
| realized trades | 80 | 88 |
| tickers still holding a real position at the end of the run | 10 of 12 | 1 of 12 |
| gross real exposure at the end of the run | 60.22% of equity -- pinned at the configured `max_gross_exposure_pct` cap | 10.00% of equity -- just one ticker's not-yet-matured most recent entry |

Before the fix, 10 of 12 tickers were still sitting on real positions
when the run ended, with aggregate exposure pinned against the
portfolio-wide cap -- exactly the silent, unbounded accumulation this
whole section exists to describe, and the reason the dashboard's "Risk
& exposure" panel kept showing gross exposure hugging 60% for so much
of a run. After the fix, every position whose owning prediction had
actually resolved was flat, with only the single most-recent (and
therefore still genuinely pending, not yet matured) entry left open --
exactly the shape a correctly-closing continuous-exposure system should
have. (Trade count and final equity differ slightly between the two
runs, 80 vs. 88 trades: expected, not a discrepancy to chase down --
closing real positions changes the broker's real equity on every
subsequent bar, which feeds back into `update_account_equity()` and
therefore every later sizing decision, so the two runs' trading paths
diverge after the first real close. That the paths diverge at all is
itself a small piece of independent confirmation that real closes are
now actually happening.) The decision log from the "after" run was also
checked directly for gross errors: all 88 real trades produced exactly
88 `close_order` log rows, every one `filled: true` (PaperBroker always
fills a valid close), and zero NaN/Inf values anywhere across all
20,250 log rows the run wrote.

23 new regression tests: `TestRealBrokerCloseOnResolution` in
`tests/test_orchestrator.py` (8 -- maturity resolution actually
flattens the real position; the hard stop-loss does too; closing a
short buys back the exact quantity; two overlapping entries on the same
ticker each net out only their own share; an unfilled prediction never
submits a closing order at all; a successful close is logged; a close
that doesn't confirm is logged as unfilled rather than silently dropped
-- and the trade is still graded either way; and a full end-to-end
`run()` → real fill → forced maturity → flat position check, not a
reimplementation of the loop); a new `tests/test_broker.py` giving
`PaperBroker` its first direct unit tests at all (10 -- `submit_order`'s
own cash/position/commission math, previously only ever exercised
indirectly through `Orchestrator` tests, plus `close_quantity`: closing
a long or a short flat, a partial close leaving the correct residual,
cash/commission matching `submit_order`'s own model exactly, HOLD/
non-positive quantity rejected without touching the position, closing a
ticker with no existing position still executing cleanly, and the
resulting `Fill` carrying the closing action, not the original entry's);
and 5 more in `tests/test_alpaca_broker.py` (a filled close returns a
`Fill` with the closing action and the real fill price/quantity;
`close_quantity` submits the exact requested quantity, never one
re-derived from a notional -- checked with a deliberately absurd
`ref_price` that would expose the bug if it reappeared; a rejected order
returns `None`; an order that never confirms within the poll window
returns `None`; and HOLD/non-positive quantity never submits anything
to the client at all).

What this fix does *not* change: the continuous net-exposure design
itself (a new order still sizes against the ticker's *remaining*
headroom under the per-ticker/gross caps, not against "open a discrete
trade and close it later" bookkeeping), and the mark-to-market drift
question flagged in the sixth bug's correction above (a sizing-time cap
still can't retroactively shrink an *already-open* position that drifts
via ordinary price movement between signals -- this fix makes exits
actually real, it doesn't add continuous trimming). Both remain
accurately described, not newly introduced, by this change.

## Volatility-scaled stop-loss, replacing one flat percentage for every ticker

The hard stop-loss added above (and the seventh bug's fix, which made it
actually close a real position) used one flat `hard_stop_loss_pct`
(3.0%) for every ticker, every time -- the same number whether the
ticker was SPY on a quiet day or NVDA on a wild one. That's a real
problem, not just an aesthetic one: a flat percentage stop is
effectively a bet on how volatile the underlying is, and this strategy
holds twelve tickers with very different volatility profiles.

Measured directly from `realized_vol_15` (the same 15-bar close-to-close
return stdev `size_order()` already uses for vol-targeted sizing) over a
3,000-bar synthetic run: per-bar realized vol ranges from roughly 0.3%
at a calm ticker's 10th percentile up to over 4% at a choppy one's
wilder moments -- more than an order of magnitude apart, on the same
flat 3% stop. In practice that meant a calm ticker's stop was far looser
than it needed to be, while a choppy one's was tight enough to be routinely
triggered by ordinary noise rather than any real adverse move -- exactly
the "3% strict will likely cause too many exits" problem on the names
this strategy most wants to be able to ride a real trend on.

Fixed with `RiskManager.stop_loss_pct_for(realized_vol)`:
`stop_loss_vol_multiplier` (default 3.0) standard deviations of that
ticker's own recent realized volatility, clamped to
[`min_stop_loss_pct`, `max_stop_loss_pct`] (default 1.5% / 8.0%) so an
ultra-calm ticker's stop never gets razor-thin and a genuinely wild
ticker's stop never gets so wide it stops meaning anything.
`size_order()` computes this once per sized order (from that bar's
`realized_vol`, the same input already driving position size) and
returns it as `stop_loss_pct`; `Orchestrator.run()` now stores that
value on the pending entry itself (a new `stop_loss_pct` field,
alongside `filled_quantity` from the seventh bug's fix), and
`_check_stop_losses()` reads each entry's *own* stop level instead of
one value shared by every position. `hard_stop_loss_pct` is gone from
`RiskConfig` entirely -- it was a config *value*, not persisted
`RiskState`, so there's no state-file migration concern, and
`config.yaml`'s `risk:` section now carries the three new keys instead.

Verified directly against the same 3,000-bar realized-vol distribution
above, computed deterministically from `stop_loss_pct_for()` itself
(not a single noisy backtest run, which can't isolate this cleanly --
see the caveat below): at each ticker's own *median* vol, the new stop
is actually slightly *tighter* than the old flat 3% (anywhere from
−1.8% for the choppiest ticker, MSFT, to −30.6% for the calmest, SPY) --
appropriate, since most of the time a tighter stop costs little and
protects more. But at each ticker's own *90th-percentile* vol -- the
choppy moments a flat stop was actually getting triggered by -- the
stop widens past the old flat 3% for all 12 of 12 tickers, from +22%
(JPM, the calmest) up to +75% (NVDA, the choppiest: 5.24% vs the old
3.00%). That's the mechanism working exactly as intended: tighter when
it costs little, wider exactly when the old flat stop was most likely
to be noise, not signal.

A secondary, honestly-caveated observation from an actual backtest (same
3,000-bar/12-ticker run, seed 7, kill switch forced off): total
stop-loss-triggered exits went from 26 to 38, and the *distribution*
shifted as the mechanism above predicts -- NVDA (the choppiest ticker
measured above) dropped from 4 stop-outs to 1, while several calmer
tickers that had zero stop-outs under the flat 3% (SPY, QQQ, GOOGL, UNH)
now have a few, consistent with their vol-scaled stop being slightly
tighter than 3% most of the time. This is **not** a clean controlled
comparison, though, for the same reason noted in the seventh bug's
section above: closing real positions (that fix) changes equity on every
later bar, which changes every later sizing decision, so the "before"
and "after" trading paths diverge after the very first difference --
total trade count differs too (88 vs. 91). The deterministic vol-to-stop
table above is the real evidence this fix does what it's supposed to;
this backtest is only a sanity check that it doesn't look obviously
wrong in a real run (no NaN/blowup, no runaway trade count, direction of
the shift matches the mechanism), not proof of the exact trade-by-trade
effect in isolation.

13 new regression tests: `TestStopLossVolScaling` in `tests/test_risk.py`
(7 -- scales linearly with `realized_vol`; a calmer ticker gets a
tighter stop than a choppier one; clamped to the floor/ceiling at the
extremes; falls back to the floor on non-finite/non-positive vol;
`size_order()` returns the computed value, not a flat constant; a
no-trade decision still reports `stop_loss_pct` as `None`) and
`TestStopLossVolScaling` in `tests/test_orchestrator.py` (4 -- two
entries with different stops breach independently on the same adverse
move, proving `_check_stop_losses` reads each entry's own value, not one
shared by the whole ticker; a position survives a move that would
breach a flat 3% but not its own real (wider) stop, then is correctly
stopped out once its own stop level IS breached; `run()`'s real,
computed `stop_loss_pct_for()` output ends up on the pending entry
end-to-end; and an unfilled prediction's `stop_loss_pct` stays `None`
and is never read).

## An eighth finding, from a live run: 9 of 12 tickers never streamed a single live bar, because two Alpaca websocket clients were fighting over one connection slot

Reported while this was running live (paper account, real-time data):
`localhost:8787`'s dashboard looked empty, and once the live decision log
was actually read, the real symptom was narrower than "no data" -- `SPY`,
`QQQ`, and `AAPL` (the project's *original* three-ticker list) had bars
and were trading normally; the other nine tickers added later
(`MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, UNH, XOM`) had logged exactly
zero bars since the config was expanded to 12. The Performance Summary
panel was accurately reporting an empty result for those nine -- it
wasn't a dashboard bug.

What the evidence showed, from `logs/run_live_alpaca_stdout.log` and
`logs/premarket_check.log`:

- `run_live_alpaca.py` had been restarted that morning (pid change,
  confirmed by its own startup banner printing the full, correct 12-ticker
  list) right as the market opened.
- From the moment that new process started, every single attempt to open
  its Alpaca market-data websocket failed immediately at the auth step
  with `ValueError: connection limit exceeded` -- thousands of consecutive
  failures, with no successful connection ever logged, for hours, still
  ongoing at the time this was investigated.
- Alpaca's free/IEX data plan allows exactly **one concurrent live
  websocket connection per API key**. `connection limit exceeded` at auth
  means some *other* connection was already holding that slot.
- `scripts/premarket_check.py` (the script that makes sure
  `run_live_alpaca.py` / `serve_dashboard.py` are running, scheduled
  before market open) detects an already-running instance by matching
  `run_live_alpaca.py` against the command line of every `python.exe`
  process via WMI. That check is accurate for anything started the way
  the project's own scripts start it -- but it had a blind spot: a
  process started by hand with the Windows `py` launcher (`py.exe`) or as
  `pythonw.exe` wouldn't match `Name='python.exe'` and would be invisible
  to it, letting a duplicate run alongside an automatically-started
  instance without either one ever knowing about the other.
- That lines up exactly with what was observed: an older process (still
  holding the original three-ticker subscription) kept the one available
  connection slot occupied, while the newer, correctly-configured process
  could never get past auth to subscribe its other nine tickers at all.

**This entry documents a diagnosis and an observability/detection fix, not
a confirmed root-cause fix** -- I can't inspect or kill Windows processes
on the machine this runs on from here, so resolving the actual duplicate
(if that's what it is) needs a one-time manual check: look for more than
one process with `run_live_alpaca.py` in its command line (Task Manager,
or `Get-CimInstance Win32_Process -Filter "Name='python.exe' OR
Name='pythonw.exe' OR Name='py.exe'" | Select ProcessId,CommandLine` in
PowerShell) and stop the extra one(s), then let `premarket_check.py`
start a single clean instance.

Two changes land now regardless of that manual step, so this class of
problem is fast to diagnose (or avoid) next time:

- **`src/data/alpaca_feed.py`**: `AlpacaLiveFeed.stream()` now prints one
  line the *first* time each subscribed ticker's bar arrives (e.g. `first
  live bar received for AAPL (1/12 tickers streaming so far)`), purely
  additive -- it doesn't change what's yielded or when, only what's
  logged. The previous log had plenty of evidence of *failures*
  (`log.exception(...)`/`log.warning(...)` are above the default
  `WARNING` threshold) but zero evidence of *successes*
  (`log.info("connected to ...")` / `log.info("starting ... websocket
  connection")` are below it and were silently dropped with no handler
  configured) -- so there was no way to tell, from the log alone, which
  tickers were actually getting through. The module now also raises the
  `alpaca` logger to `INFO` with a handler, surfacing those previously
  -invisible connect/subscribe/reconnect lines too.
- **`scripts/premarket_check.py`**: `is_running()`'s WMI filter now also
  matches `pythonw.exe` and `py.exe`, not just `python.exe`, closing the
  blind spot described above.

6 new regression tests: `TestStreamDoesNotChangeBarDelivery` (3) and
`TestStreamFirstBarTracking` (3) in `tests/test_alpaca_feed.py` (the new
logging is additive and never changes bar delivery order/content, prints
exactly once per ticker on its first bar, never mentions a ticker that
never streamed, and reports the progress fraction against the full
subscribed list) and `TestIsRunning` in `tests/test_premarket_check.py`
(7 -- matches `python.exe`, `py.exe`, and `pythonw.exe` command lines;
correctly reports not-running on a non-match or a subprocess error; and
the constructed PowerShell filter string contains all three process
names).

## A ninth finding: `_prime_history()` swallowed every warm-up failure completely silently

Asked how often to expect a retrain, the honest answer turned out to be
"it hasn't predicted at all yet today" -- `logs/decisions.jsonl` had zero
`prediction` events despite tickers having streamed live bars for a
while. That's explained on its own by the earlier duplicate-process
incident (every restart resets `_history` to empty, so the 120-live-bar
warm-up clock keeps getting reset too) -- but digging into *why*
predictions start at all surfaced something worth fixing regardless:
`Orchestrator._prime_history()` is supposed to backfill each ticker's
history via `feed.get_history()` before the live loop starts, so
prediction can begin almost immediately instead of waiting out
`warmup_bars` (120) live bars from scratch. If that REST call fails or
comes back empty for a ticker, it was handled like this:

```python
try:
    hist = self.feed.get_history(t, self.max_history)
except Exception:
    hist = None
if hist is None or len(hist) == 0:
    continue
```

No print, no log, nothing -- a feed/subscription permission problem, a
transient REST error, or simply an unrecognized ticker all looked
identical to "warm-up worked fine, just wait" from the log. The only way
to tell the two apart was to wait ~2 hours and see whether `prediction`
events ever showed up; there was no way to tell *why* if they didn't.

`_prime_history()` now prints exactly one line per ticker, every time,
covering all four outcomes: the exception and its message when
`get_history()` raises, an explicit "returned no bars" note when it
succeeds but is empty, "ready to predict immediately" when the backfill
clears `warmup_bars`, or how many more live bars are still needed when it
doesn't. Purely additive -- it reports what already happens, it doesn't
change which ticker falls back to live-only warm-up or how long that
takes.

5 new tests in `TestPrimeHistoryReporting` (`tests/test_orchestrator.py`,
via a minimal `_StubPrimeFeed`): the exception path reports the message
and never raises out of `_prime_history()` itself; an empty-but-no-
exception result is reported distinctly from a raised exception; a
backfill that clears `warmup_bars` reports "ready to predict
immediately"; one that doesn't reports exactly how many more bars are
needed; and multiple tickers in the same call are each reported on their
own merits (one can succeed while another fails in the same pass).

## A "Logs" page on the dashboard

Added directly off the back of this session's own workflow: diagnosing
the duplicate-process and silent-warm-up issues above meant grepping
`run_live_alpaca_stdout.log` by hand through a shell, every time. The
dashboard now has a second page for exactly that, reachable via a
Monitor / Logs toggle next to the brand mark in the topbar.

**Process logs.** `scripts/serve_dashboard.py` now also serves:

- `GET /logs` -- a JSON listing of every `*.log` file sitting next to
  `decisions.jsonl` (today: `run_live_alpaca_stdout.log`,
  `serve_dashboard_stdout.log`, `premarket_check.log`), discovered by
  glob rather than a hardcoded filename list, so a future script's
  `*_stdout.log` shows up with no code change.
- `GET /log/<name>` -- the tail of one of those files, plain text.
  `<name>` only ever resolves by exact-matching a filename `/logs` just
  discovered on disk, so there's no path-traversal surface via this
  route. Capped at `--log-tail-bytes` (default 200,000 bytes, roughly
  2-3k lines) read via a seek-from-the-end, not a full read -- these
  files can and do run past 100k lines / several MB during a reconnect
  storm (the `run_live_alpaca_stdout.log` from the incident above was
  7.7MB by the time this was tested against it), and loading the whole
  thing on every 4-second poll was never going to be an option.

The dashboard's Logs page polls both endpoints the same way the Monitor
page already polls `/live-log`: a chip row picks which discovered log to
view, a search box filters (and highlights) matching lines, and an
"errors/warnings only" checkbox narrows to lines matching
`error|warn|exceeded|traceback|exception`. None of it works when the
page was opened by dropping a file in or from the bundled sample --
there's no server behind either of those to ask -- and that's handled
as an explicit message in the panel, not a silent blank space.

**Raw decision-log lines.** A second new card shows the unprocessed
NDJSON lines behind whatever log is currently loaded (works in both live
and drop-in/sample mode, unlike the process-log panel), with its own
search box -- for spotting a malformed line or an event type the charts
above don't otherwise surface. `loadText()`'s parser has always silently
dropped a line it can't `JSON.parse()` (`catch (e) { /* skip malformed
line */ }`); this view is the one place that silently-dropped line is
still visible.

Verified three ways, since this project has no JS test runner to lean
on: the three new pure Python functions backing the server routes
(`discover_text_logs`, `tail_bytes`, `log_listing`) have 17 unit +
HTTP-routing tests in `tests/test_serve_dashboard.py` (glob discovery
and sorting, the exclude guard, tail-boundary correctness on an oversized
file, 404s for an undiscovered name, a path-traversal attempt, the
configured byte cap actually being respected); a throwaway jsdom
harness (not part of the repo) drove the real `dashboard.html` through a
scripted `fetch` mock end-to-end -- tab switching, log selection,
search/highlight filtering on both panels, and switching back to Monitor
without leaking any Logs-page state -- 27/27 checks, plus 4/4 more for
the no-server (drop-in/sample) fallback path; and the real
`serve_dashboard.py` was run against this project's actual `logs/`
directory, confirming `/logs` and `/log/run_live_alpaca_stdout.log`
against that real 7.7MB file in practice, not just in a unit test.

## The Logs page gets the pill/chip treatment

Feedback on the page above, right after it shipped: it was still just a
flat dump of raw text and raw JSON wearing the dashboard's dark theme --
not actually in its visual language, and not any easier to read than
`grep`. Two changes:

**Raw decision-log lines is now a table, not a text dump.** Every line
is parsed and rendered as one row -- a type pill (reusing the existing
`.action-pill`/`.status-pill` classes from the Decision Feed: `buy`/
`sell`/`hold` for predictions, `good`/`critical` for outcomes,
`good`/`muted` for retrain promotions, and new `warning`/`info` variants
for everything else) plus a plain-English summary (`"AAPL — BUY call,
72% confidence (model v3)"`, `"Correct — realized +0.42%"`, `"Retrain v4
— rejected (loss gate failed)"`, `"Gross exposure 12.4% (cap 60.0%) ·
daily P&L +0.30%"`, and so on for every event type `DecisionLogger`
emits). The exact JSON is one click away via the same expandable
`<details class="row-detail">` row the Decision Feed table already uses
-- nothing is hidden, just not shown by default. A line that still
fails `JSON.parse()` is kept as its own "Unparsed" type instead of being
silently dropped, same as before, just now filterable like everything
else. Filter pills are built from whatever types are actually present in
the loaded log (with live counts), plus the search box narrows against
the summary text too, not just the raw line.

**Process logs now classify and group, instead of dumping every stdout
line flat.** Each line gets a level -- error / warning / connect / info
-- shown as a small colored dot (same dot idiom as the action pills),
and a `Traceback (most recent call last): ... ExceptionType: message`
block collapses into a single collapsible entry (summary = the exception
message, full frames one click away) instead of taking up 5-10 lines in
the view. The old "errors/warnings only" checkbox is gone in favor of
filter pills (All / Errors / Warnings / Connects / Info, each with a
live count) matching the rest of the dashboard's chip row convention.
Run against the real `run_live_alpaca_stdout.log` from the
connection-limit incident, the Errors filter immediately isolates the
~600 repeated `ValueError: connection limit exceeded` retries from the
~35 lines that actually mattered -- which was the entire point of
building this page in the first place.

Verified against the real server again: the existing 222 Python tests
are untouched by this (it's a client-only change -- `/logs` and
`/log/<name>` already returned exactly what the new rendering needed),
a rewritten throwaway jsdom harness checks the type/level pills, the
human-readable summaries per event type, the unparsed-line fallback, and
both filter-pill interactions end-to-end (23/23), and a live
`serve_dashboard.py` run against this project's actual `logs/` directory
confirmed the table and traceback-collapsing against the real 638-line
`run_live_alpaca_stdout.log` and the real 2,449-line `decisions.jsonl`,
not just synthetic fixtures.

## A tenth finding: inconsistent per-ticker warm-up bar counts (118 vs 65), from a rolling multi-day lookback window

Spotted on the dashboard's Monitor page after a live run: some tickers'
small-multiples warmed up with 118 bars, others with only 65, at the exact
same moment -- despite every ticker going through the identical
`_prime_history()` call with the identical `max_history` lookback.

The cause was in `AlpacaLiveFeed.get_history()`: it asked Alpaca for up to
`lookback` (400) one-minute bars over a rolling `history_days`-day window
(10 days back from now), via `StockBarsRequest(..., limit=lookback)`.
`limit` caps the request, it doesn't guarantee you get the *most recent*
`lookback` bars -- for a less liquid ticker with sparse IEX coverage,
bars from several days back could still be part of what's returned, at
the expense of some of today's bars never being reached at all. A
heavily-traded ticker, by contrast, fills the limit from today's bars
alone. Net effect: two tickers starting at the same instant could end up
with very different "today, so far" coverage, purely as an artifact of
how far back each one's request happened to dig, not real data
availability.

Fixed by anchoring the warm-up window to today's actual market open
(09:30 `America/New_York`, converted to UTC) instead of a multi-day
lookback, whenever the market has already opened today --
`AlpacaLiveFeed._todays_session_open_utc()` computes it, correctly across
the EST/EDT boundary since it goes through `zoneinfo` rather than a fixed
UTC offset. Before today's open (premarket), there's no "today" window
yet, so it falls back to the old `history_days`-based lookback -- the
fix only changes behavior for the common case (market open, warming up
before/at the start of a live run), not that edge case. Every ticker
warming up at the same moment now gets the identical `[start, end)`
window; whatever bars actually printed in it is a true reflection of
that ticker's liquidity today, not a side effect of request ordering.

Covered by `tests/test_alpaca_feed.py::TestTodaysSessionOpenUtc` (the
EST/EDT conversion, and the date boundary just after UTC midnight where
the Eastern calendar date is still "yesterday") and
`TestGetHistoryAnchorsToTodaysOpen` (the after-open/before-open branches,
and -- the actual regression this fixes -- that two tickers queried at
the same instant get byte-identical start/end windows).

## Extending this toward something real

- **Live data and paper-broker fills are done** (`AlpacaLiveFeed`,
  `AlpacaBroker` — see the section above). `LiveBrokerStub` in
  `src/execution/broker.py` is still what you'd implement for a different
  broker (Interactive Brokers via `ib_insync`, etc.) — it documents exactly
  what a real integration needs beyond what `AlpacaBroker` already shows
  one example of: fills, rejections, rate limits, reconciliation, secrets
  management.
- **Persist state across restarts** before running live unattended for
  long stretches — see the known limitation above. The replay buffer
  already supports this (`ReplayBuffer.save()`/`.load()`); the risk
  manager's counters and drift monitor's rolling windows don't yet.
- **Richer features / cross-asset context.** `src/data/features.py` is
  intentionally simple (a dozen-odd technical indicators per ticker,
  computed independently). Order-book features now exist (see the section
  above) but aren't wired into `FEATURE_COLUMNS` yet, and still need a real
  L2 vendor for equities. Cross-sectional features (sector/market relative
  strength) and macro/news features are still open — each is just another
  column once you have the data.
- **Scale up the model.** If you widen the feature set or lookback window
  significantly, `TemporalEncoder` can be swapped for a small Transformer
  encoder without touching the heads, loss, or anything downstream.
- **If this ever connects to the agentic LLM trading-firm pipeline** you've
  also been building: this network is a natural drop-in replacement (or
  second opinion alongside) the Market Analyst / Trader agents in that
  blackboard architecture — it would write its `Signal` and sizing
  decision to the same shared session store the LLM agents read/write,
  and the monitoring dashboard's decision-timeline view would show both
  side by side.

## Before you even think about live trading

This scaffold is built to make *safety* visible and testable, not to make
any claim that the strategy itself is profitable — those are two separate
problems, and this project only addresses the first one. A neural network
that is well-engineered can still lose money; nothing here is investment
advice, and past or simulated performance (including anything the
backtest script prints) is not indicative of future results.

Concretely, before any of this touches a funded account:

- Run it in paper mode for an extended period (months, not days) across
  different market regimes, and look hard at the decision log, not just
  the final equity number.
- Account for costs this scaffold only approximates: real slippage under
  stress, exchange/data fees, taxes, and borrow costs for shorting.
- Have a real plan for what happens when the model, the data feed, or the
  broker connection fails mid-session — not just when the model is wrong.
- Understand the pattern-day-trader rule, margin requirements, and
  wash-sale rules if applicable, and consider talking to a licensed
  financial/tax professional before committing real capital.
- Treat the kill switch as a last resort, not a strategy — the risk
  parameters in `config.yaml` (position caps, daily loss limit, drawdown
  ceiling) are starting points, not tuned recommendations, and you should
  set them deliberately for your own risk tolerance.
