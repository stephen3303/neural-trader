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

`run_paper_trading.py` prints final equity, number of fills, how many
retrain attempts were promoted, the drift monitor's final snapshot, and
whether the kill switch tripped. The full decision trail is in
`logs/decisions.jsonl`.

## Watching it live: the dashboard

`dashboard.html` is a self-contained monitoring UI (equity curve, rolling
hit-rate/calibration, retrain history, per-ticker signal traces, a
searchable decision feed). You can always open it and drag `decisions.jsonl`
onto the page by hand, but while a loop above is actively running, it's more
useful with auto-refresh:

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
  and `alpaca:` — `risk.account_equity` is a separate static assumption
  used for position-sizing math, not read from your actual Alpaca balance,
  so update it if your paper account's equity isn't the default $100,000.

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

**Still a known limitation — read before leaving this running unattended:**
the replay buffer, the risk manager's daily-loss/consecutive-loss counters,
and the drift monitor's rolling windows all live in memory only and are
*not* covered by the checkpoint above. If the process restarts mid-session,
that state still resets — the kill switch forgets today's drawdown, the
buffer forgets today's outcomes, even though the model itself now resumes
correctly. Fine for an initial test; worth adding persistence for these
three (the replay buffer already has `.save()`/`.load()` — the other two
don't yet) before running this unattended for extended stretches.

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
