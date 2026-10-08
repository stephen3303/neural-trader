#!/usr/bin/env python
"""
Pre-market readiness check.

Confirms today is an actual NYSE trading day (via Alpaca's own market
calendar -- this correctly skips weekends AND market holidays like
Thanksgiving or July 4th, unlike a plain "Mon-Fri" schedule), then makes
sure run_live_alpaca.py and scripts/serve_dashboard.py are both running,
starting whichever one isn't.

Meant to be triggered by Windows Task Scheduler ~30 minutes before the
9:30am ET open, via the companion scripts/premarket_check.bat -- see the
README section "Automated pre-market check" for how to register it. It's
safe to run every day, including weekends: on a non-trading day it just
logs that and exits with nothing started.

This script does NOT place trades or touch any model/risk state itself --
it only starts/checks OS processes. run_live_alpaca.py already does its
own internal wait_for_market_open() before trading, so starting it 30
minutes early just means it sits there printing "market closed, checking
again..." until 9:30, exactly as if you'd started it by hand early.

Run manually to test it:
    python scripts/premarket_check.py

Logs go to stdout/stderr; premarket_check.bat redirects that to
logs/premarket_check.log so a Task Scheduler run leaves a trail even when
nobody's watching.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv is optional; env vars can be exported directly instead

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover -- Python < 3.9, not expected here
    ZoneInfo = None

LOG_DIR = PROJECT_ROOT / "logs"

# (friendly name used for matching + logging, script path, extra CLI args)
WATCHED = [
    ("run_live_alpaca.py", PROJECT_ROOT / "scripts" / "run_live_alpaca.py", []),
    ("serve_dashboard.py", PROJECT_ROOT / "scripts" / "serve_dashboard.py", ["--no-open"]),
]


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"Missing required environment variable {name}. Set it (or add it to "
            f".env -- see .env.example) before this check can run."
        )
    return val


def today_eastern() -> datetime.date:
    """NYSE's calendar is dated in US/Eastern, not whatever timezone this
    machine happens to be in -- matters most right around midnight ET."""
    if ZoneInfo is not None:
        try:
            return datetime.datetime.now(ZoneInfo("America/New_York")).date()
        except Exception:
            pass  # tzdata not installed (see requirements.txt) -- fall back below
    return datetime.date.today()


def is_trading_day(trading_client) -> bool:
    """Ask Alpaca's own market calendar rather than guessing from the day
    of the week, so holidays are handled correctly with zero maintenance."""
    from alpaca.trading.requests import GetCalendarRequest

    today = today_eastern()
    calendar = trading_client.get_calendar(GetCalendarRequest(start=today, end=today))
    return len(calendar) > 0


def is_running(script_filename: str) -> bool:
    """True if a python process is currently running that script.
    `tasklist` alone only shows the image name (python.exe for all of
    them, indistinguishable), so this reads full command lines via WMI
    instead, which is reliable without adding a dependency like psutil.

    Matches python.exe, pythonw.exe, AND py.exe (the Windows "py" launcher)
    -- not just python.exe. A process started by hand with `py
    run_live_alpaca.py` (or a pythonw-based one) used to be invisible to
    this check, which could let it run alongside a second, automatically
    started instance of the same script: Alpaca's live data websocket
    allows only one concurrent connection per API key, so the duplicate
    silently starved every ticker not already held by whichever process
    got there first (see the README/commit history for how this surfaced:
    9 of 12 configured tickers never received a single live bar while a
    leftover process from an older, shorter ticker list kept running)."""
    if os.name != "nt":
        log(f"  (not running on Windows -- skipping process check for {script_filename})")
        return False
    ps_cmd = (
        "(Get-CimInstance Win32_Process "
        "-Filter \"Name='python.exe' OR Name='pythonw.exe' OR Name='py.exe'\" "
        "| Select-Object -ExpandProperty CommandLine) -join \"`n\""
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as e:
        log(f"  warning: couldn't query running processes ({e}); assuming not running")
        return False
    return script_filename.lower() in (result.stdout or "").lower()


def start_script(name: str, path: Path, extra_args: list[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"{path.stem}_stdout.log"
    f = open(logfile, "a")
    kwargs: dict = dict(cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL, stdout=f, stderr=f)
    if os.name == "nt":
        # Task Scheduler runs this script inside a Job Object. Without
        # these flags, Windows can silently kill the child the moment
        # THIS script exits -- which would defeat the entire point.
        # CREATE_BREAKAWAY_FROM_JOB detaches it from that job;
        # DETACHED_PROCESS/CREATE_NEW_PROCESS_GROUP give it its own
        # console-less session so it outlives this process.
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | CREATE_BREAKAWAY_FROM_JOB
        )
    # -u forces unbuffered stdout/stderr. Without it, Python block-buffers
    # when output isn't a terminal (which it never is here), so even the
    # startup prints can sit in memory indefinitely -- making a perfectly
    # healthy process look dead because its log file stays at 0 bytes.
    proc = subprocess.Popen([sys.executable, "-u", str(path), *extra_args], **kwargs)
    log(f"  started {name} (pid={proc.pid}); its own output is going to {logfile}")


def main() -> int:
    log("=== Pre-market check ===")
    api_key = require_env("ALPACA_API_KEY")
    secret_key = require_env("ALPACA_SECRET_KEY")

    from alpaca.trading.client import TradingClient
    trading_client = TradingClient(api_key, secret_key, paper=True)

    if not is_trading_day(trading_client):
        log(f"{today_eastern()} is not a trading day (weekend or market holiday) -- nothing to do.")
        return 0

    clock = trading_client.get_clock()
    if clock.is_open:
        log("Trading day confirmed. Market is already open.")
    else:
        log(f"Trading day confirmed. Next open: {clock.next_open}.")

    any_started = False
    for name, path, extra_args in WATCHED:
        if is_running(name):
            log(f"{name}: already running -- OK.")
        else:
            log(f"{name}: NOT running -- starting it now.")
            start_script(name, path, extra_args)
            any_started = True

    if any_started:
        log("Done. Started the process(es) noted above -- check their own log "
            "files (or the dashboard) in a few minutes to confirm they came up cleanly.")
    else:
        log("Done. Everything was already running.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        log(f"ERROR: pre-market check failed: {e!r}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
