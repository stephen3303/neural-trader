#!/usr/bin/env python
"""
Stop run_live_alpaca.py and/or serve_dashboard.py.

If you're looking at the terminal window one of them is running in, the
simplest shutdown is just Ctrl+C there -- run_live_alpaca.py catches it,
prints a session summary (final equity, fills, drift snapshot, kill
switch state), and exits cleanly.

This script is for the other case: either script was started detached
(e.g. by scripts/premarket_check.py, which has no visible console), or
you just don't want to go hunt down the window. It finds the matching
python.exe process(es) by command line (same technique premarket_check.py
uses to check if they're running) and terminates them with `taskkill`.

That's a hard stop, not a graceful one -- it skips run_live_alpaca.py's
own Ctrl+C handling, so you won't get the printed session summary, and
whatever's in memory (replay buffer, today's risk counters, drift
windows -- see the README's "Known limitation") is simply gone, same as
if the process had crashed. That's expected and harmless for paper
trading; it's exactly the in-memory-state tradeoff already documented
for this scaffold. It does NOT touch your Alpaca paper account itself --
any open paper position stays exactly as it was, since nothing here
places or cancels orders.

Usage:
    python scripts/stop_trading.py                 # stops both, if running
    python scripts/stop_trading.py run_live_alpaca.py   # stops just this one
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys

WATCHED = ["run_live_alpaca.py", "serve_dashboard.py"]


def find_pids(script_filename: str) -> list[str]:
    """Returns PIDs of python.exe processes whose command line mentions
    script_filename. CSV (not plain text) so a command line containing a
    comma or quote doesn't get misparsed."""
    if os.name != "nt":
        print(f"  (not on Windows -- can't look up processes for {script_filename} here)")
        return []
    ps_cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as e:
        print(f"  warning: couldn't query running processes ({e})")
        return []
    pids = []
    for row in csv.DictReader(io.StringIO(result.stdout)):
        cmdline = (row.get("CommandLine") or "")
        if script_filename.lower() in cmdline.lower():
            pid = row.get("ProcessId")
            if pid:
                pids.append(pid)
    return pids


def stop(script_filename: str) -> None:
    pids = find_pids(script_filename)
    if not pids:
        print(f"{script_filename}: not running.")
        return
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, text=True, timeout=10)
            print(f"{script_filename}: stopped (pid {pid}).")
        except Exception as e:
            print(f"{script_filename}: found pid {pid} but failed to stop it ({e}). "
                  f"Try Task Manager -- look for python.exe with this PID.")


def main() -> int:
    targets = sys.argv[1:] or WATCHED
    unknown = [t for t in targets if t not in WATCHED]
    if unknown:
        print(f"Unknown target(s): {unknown}. Expected one or more of {WATCHED}.")
        return 1
    for name in targets:
        stop(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
