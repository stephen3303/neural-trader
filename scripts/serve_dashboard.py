#!/usr/bin/env python
"""
Tiny local server that gives the monitoring dashboard (dashboard.html)
auto-refresh while a trading loop (run_paper_trading.py / run_live_alpaca.py)
is running, instead of requiring you to re-drag decisions.jsonl into the
page every time you want to see what's new.

It does five things. The first four are read-only:

  GET /              -> serves dashboard.html
  GET /live-log       -> serves the current contents of your decision log
  GET /logs           -> JSON listing of the process logs in logs/ (name,
                         size, last-modified) -- run_live_alpaca_stdout.log,
                         serve_dashboard_stdout.log, premarket_check.log,
                         and anything else matching *.log dropped in
                         alongside decisions.jsonl, picked up automatically
  GET /log/<name>     -> the tail of one of those files (plain text),
                         `name` must be one of the names /logs just
                         listed -- there's no other way to name a path,
                         so this can't be used to read an arbitrary file

The fifth is the one deliberate exception to "read-only":

  POST /reset-kill-switch -> writes a reset-request sentinel file (see
                         RiskManager.write_reset_request's docstring)
                         into --risk-state-dir, for the live trading
                         process to pick up on its own next bar and
                         actually clear via
                         RiskManager.reset_kill_switch(human_confirmed=True).
                         This process still never touches the trading
                         process directly, and never flips
                         trading_enabled itself -- it only ever leaves a
                         request lying around. A click here IS the human
                         confirmation reset_kill_switch() requires; see
                         the dashboard's "Reset kill switch" button.

The dashboard's own "Logs" page polls /logs and /log/<name> the same way
its main view polls /live-log -- added after a live incident where
diagnosing a stuck websocket reconnect loop meant grepping these stdout
files by hand over an SSH-like device shell; the point of this page is to
make that first look something you can do from the dashboard itself.
This process never writes to any of the discovered logs or to
decisions.jsonl, and has no direct connection to the trading process
itself -- stopping it (Ctrl+C) does not stop or affect a running
run_live_alpaca.py / run_paper_trading.py in any way, and vice versa.
POST /reset-kill-switch is the one exception to "no connection at all":
it writes a small sentinel file the trading process polls for on its
own, the only channel between these two otherwise-independent
processes.

Usage:

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --log logs/decisions.jsonl --port 8787

Then open the printed URL (it also tries to open it in your browser for
you). Leave this running alongside the trading script and the dashboard
will keep itself current.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from src...` below resolves regardless of cwd

from src.risk.manager import RiskManager  # noqa: E402 -- needs the sys.path.insert above first


def discover_text_logs(log_dir: Path, exclude: Path | None = None) -> list[Path]:
    """Every *.log file directly inside log_dir -- the process stdout
    files premarket_check.py's start_script() and run_live_alpaca.py
    write to, plus this script's own serve_dashboard_stdout.log and
    premarket_check.log. Sorted by name for a stable order in the
    dashboard's log picker. `exclude` keeps a specific path (the decision
    log itself, if it somehow ends in .log) out of the discovered set.
    New *_stdout.log files from a future script are picked up here with
    no code change needed."""
    if not log_dir.is_dir():
        return []
    paths = sorted(p for p in log_dir.glob("*.log") if p.is_file())
    if exclude is not None:
        exclude = exclude.resolve()
        paths = [p for p in paths if p.resolve() != exclude]
    return paths


def tail_bytes(path: Path, max_bytes: int) -> bytes:
    """The last `max_bytes` of `path`, without ever reading more than
    that much into memory. These process logs can balloon well past
    100k lines during a reconnect-retry storm (see the README's
    "connection limit exceeded" incident) -- reading the whole file on
    every poll is not an option. Returns b"" for a missing file, the
    same "not written yet" handling /live-log already uses below."""
    try:
        size = path.stat().st_size
    except OSError:
        return b""
    with open(path, "rb") as f:
        if size <= max_bytes:
            return f.read()
        f.seek(size - max_bytes)
        data = f.read()
        # The seek almost certainly landed mid-line; drop that partial
        # first line so everything shown starts at a real line boundary
        # instead of a truncated traceback frame.
        nl = data.find(b"\n")
        return data[nl + 1:] if nl != -1 else data


def log_listing(paths: list[Path]) -> list[dict]:
    """JSON-able {name, size, mtime} for each discovered log, so the
    dashboard's picker can show size / "updated Ns ago" without fetching
    each log's full body just to find that out."""
    out = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            continue  # disappeared between the glob and here -- skip it, not a 500
        out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


def make_handler(dashboard_path: Path, log_path: Path, log_tail_bytes: int = 200_000,
                  risk_state_dir: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes, no_store: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if no_store:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib's naming convention)
            path = self.path.split("?", 1)[0]
            if path in ("/", "/dashboard.html", "/index.html"):
                try:
                    body = dashboard_path.read_bytes()
                except OSError as e:
                    self._send(500, "text/plain; charset=utf-8",
                               f"Could not read {dashboard_path}: {e}".encode())
                    return
                self._send(200, "text/html; charset=utf-8", body)
            elif path == "/live-log":
                try:
                    body = log_path.read_bytes()
                except FileNotFoundError:
                    # The server is up but the trading script hasn't written
                    # anything yet -- that's a normal startup state, not an
                    # error, so hand back an empty log rather than a 404
                    # (a 404 would make the dashboard fall back to the
                    # bundled sample instead of staying in live mode).
                    body = b""
                except OSError as e:
                    self._send(500, "text/plain; charset=utf-8", str(e).encode())
                    return
                self._send(200, "text/plain; charset=utf-8", body, no_store=True)
            elif path == "/logs":
                logs = discover_text_logs(log_path.parent, exclude=log_path)
                body = json.dumps(log_listing(logs)).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body, no_store=True)
            elif path.startswith("/log/"):
                name = path[len("/log/"):]
                logs = discover_text_logs(log_path.parent, exclude=log_path)
                match = next((p for p in logs if p.name == name), None)
                if match is None:
                    # Deliberately NOT "here's what's available" -- `name`
                    # only ever reaches a real path by matching a filename
                    # /logs just listed, so there's no path-traversal
                    # surface here worth being more informative about.
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                body = tail_bytes(match, log_tail_bytes)
                self._send(200, "text/plain; charset=utf-8", body, no_store=True)
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        def do_POST(self) -> None:  # noqa: N802 (stdlib's naming convention)
            path = self.path.split("?", 1)[0]
            if path != "/reset-kill-switch":
                self._send(404, "text/plain; charset=utf-8", b"not found")
                return
            if risk_state_dir is None:
                # Shouldn't happen via main() below (it always passes a
                # default), but a hand-rolled make_handler() call without
                # one must fail loudly rather than silently writing
                # nowhere and reporting success.
                body = json.dumps({"ok": False, "error": "no risk-state directory configured"}).encode()
                self._send(500, "application/json; charset=utf-8", body, no_store=True)
                return
            try:
                RiskManager.write_reset_request(risk_state_dir, source="dashboard")
            except OSError as e:
                body = json.dumps({"ok": False, "error": str(e)}).encode()
                self._send(500, "application/json; charset=utf-8", body, no_store=True)
                return
            # 202, not 200: this only ever leaves a request lying around
            # (see RiskManager.write_reset_request's docstring) -- the
            # actual reset happens in the OTHER process, on its next bar,
            # not synchronously as part of this response.
            body = json.dumps({
                "ok": True,
                "note": "Reset requested. The live trading process picks this up on its own "
                        "next bar (usually within a few seconds) and only actually clears the "
                        "kill switch if it's still engaged at that point.",
            }).encode("utf-8")
            self._send(202, "application/json; charset=utf-8", body, no_store=True)

        def log_message(self, fmt: str, *args) -> None:
            # The dashboard polls /live-log every few seconds -- the default
            # per-request stderr line would just be noise. Errors still
            # print via the normal exception path.
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dashboard", default=str(ROOT / "dashboard.html"),
        help="Path to dashboard.html (default: dashboard.html next to this project's scripts/ dir)",
    )
    parser.add_argument(
        "--log", default=str(ROOT / "logs" / "decisions.jsonl"),
        help="Path to the decision log to watch (default: logs/decisions.jsonl, "
             "matching config.yaml's logging.log_path)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--log-tail-bytes", type=int, default=200_000,
        help="Max bytes of each process log's tail served by /log/<name> (default: 200000, "
             "~2-3k lines -- these logs can run past 100k lines during a reconnect storm, so "
             "this is a cap on what gets read/served per request, not a truncation you need to "
             "raise for normal use).",
    )
    parser.add_argument(
        "--risk-state-dir", default=str(ROOT / "checkpoints"),
        help="Directory the live trading process persists risk_state.json into (default: "
             "checkpoints/, matching config.yaml's trainer.checkpoint_dir) -- this is also "
             "where POST /reset-kill-switch writes its reset-request sentinel file, so the "
             "'Reset kill switch' button only actually works if this matches the directory "
             "the trading process you're watching was started with.",
    )
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    dashboard_path = Path(args.dashboard).resolve()
    log_path = Path(args.log).resolve()
    risk_state_dir = Path(args.risk_state_dir).resolve()

    if not dashboard_path.exists():
        print(f"error: dashboard file not found at {dashboard_path}", file=sys.stderr)
        sys.exit(1)

    if not log_path.exists():
        print(
            f"note: {log_path} doesn't exist yet. That's fine if "
            f"run_live_alpaca.py / run_paper_trading.py hasn't started (or "
            f"hasn't logged its first event yet) -- the dashboard will just "
            f"show 0 events until it does, and pick it up automatically "
            f"once the file appears."
        )

    handler = make_handler(dashboard_path, log_path, log_tail_bytes=args.log_tail_bytes,
                           risk_state_dir=risk_state_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"

    print(f"Serving {dashboard_path.name} at {url}")
    print(f"Watching {log_path} -- the page refreshes itself every few seconds while this runs.")
    print(f"Reset-kill-switch requests are written to {risk_state_dir} -- make sure this matches "
          f"the trading process's own checkpoint directory if you use that button.")
    print("Otherwise read-only: no other connection to the trading process. Ctrl+C to stop.")

    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
