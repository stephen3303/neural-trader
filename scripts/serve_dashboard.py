#!/usr/bin/env python
"""
Tiny local server that gives the monitoring dashboard (dashboard.html)
auto-refresh while a trading loop (run_paper_trading.py / run_live_alpaca.py)
is running, instead of requiring you to re-drag decisions.jsonl into the
page every time you want to see what's new.

It does exactly two things, both read-only:

  GET /              -> serves dashboard.html
  GET /live-log       -> serves the current contents of your decision log

The dashboard's own JS polls /live-log every few seconds and re-renders.
This process never writes to the log and has no connection to the trading
process itself -- stopping it (Ctrl+C) does not stop or affect a running
run_live_alpaca.py / run_paper_trading.py in any way, and vice versa.

Usage:

    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --log logs/decisions.jsonl --port 8787

Then open the printed URL (it also tries to open it in your browser for
you). Leave this running alongside the trading script and the dashboard
will keep itself current.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_handler(dashboard_path: Path, log_path: Path):
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
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

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
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open a browser tab")
    args = parser.parse_args()

    dashboard_path = Path(args.dashboard).resolve()
    log_path = Path(args.log).resolve()

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

    handler = make_handler(dashboard_path, log_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"

    print(f"Serving {dashboard_path.name} at {url}")
    print(f"Watching {log_path} -- the page refreshes itself every few seconds while this runs.")
    print("Read-only: this has no connection to the trading process. Ctrl+C to stop.")

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
