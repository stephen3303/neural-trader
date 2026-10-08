import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest
from http.server import ThreadingHTTPServer

from serve_dashboard import discover_text_logs, log_listing, make_handler, tail_bytes


class TestDiscoverTextLogs:
    def test_finds_log_files_sorted_by_name(self, tmp_path):
        (tmp_path / "b.log").write_text("b")
        (tmp_path / "a.log").write_text("a")
        (tmp_path / "ignored.txt").write_text("not a .log")
        found = discover_text_logs(tmp_path)
        assert [p.name for p in found] == ["a.log", "b.log"]

    def test_excludes_the_given_path(self, tmp_path):
        decisions = tmp_path / "decisions.log"  # contrived, but exercises the guard
        decisions.write_text("x")
        (tmp_path / "run_live_alpaca_stdout.log").write_text("y")
        found = discover_text_logs(tmp_path, exclude=decisions)
        assert [p.name for p in found] == ["run_live_alpaca_stdout.log"]

    def test_empty_list_for_a_missing_directory(self, tmp_path):
        assert discover_text_logs(tmp_path / "does-not-exist") == []

    def test_a_new_stdout_log_is_picked_up_with_no_code_change(self, tmp_path):
        # The whole point of globbing instead of a hardcoded filename list:
        # a script that doesn't exist yet, as long as it names its log
        # *_stdout.log, shows up here automatically.
        (tmp_path / "some_future_script_stdout.log").write_text("z")
        found = discover_text_logs(tmp_path)
        assert [p.name for p in found] == ["some_future_script_stdout.log"]


class TestTailBytes:
    def test_returns_whole_file_when_under_the_cap(self, tmp_path):
        p = tmp_path / "small.log"
        p.write_text("line one\nline two\n")
        assert tail_bytes(p, max_bytes=1_000) == b"line one\nline two\n"

    def test_returns_empty_bytes_for_a_missing_file(self, tmp_path):
        assert tail_bytes(tmp_path / "nope.log", max_bytes=1_000) == b""

    def test_truncates_to_the_tail_and_starts_on_a_line_boundary(self, tmp_path):
        p = tmp_path / "big.log"
        lines = [f"line {i:04d}" for i in range(1000)]
        p.write_text("\n".join(lines) + "\n")
        full_size = p.stat().st_size
        tail = tail_bytes(p, max_bytes=full_size // 4)
        text = tail.decode()
        # Never starts mid-line -- the seek point almost certainly lands
        # inside a line, and that partial line must be dropped.
        assert text == "" or text.startswith("line ")
        assert not text.startswith("\n")
        # It's actually the END of the file, not an arbitrary slice.
        assert text.rstrip("\n").endswith("line 0999")
        assert len(tail) <= full_size // 4 + 1  # +1: the dropped partial line's own newline

    def test_never_reads_more_than_max_bytes_plus_one_partial_line(self, tmp_path):
        # The whole reason this exists instead of path.read_bytes(): a
        # reconnect-storm stdout log can run past 100k lines, and this
        # must never load the whole thing to serve one request.
        p = tmp_path / "huge.log"
        p.write_text("x" * 50 + "\n" + ("y" * 500 + "\n") * 2000)
        tail = tail_bytes(p, max_bytes=10_000)
        assert len(tail) <= 10_000 + 501  # cap + at most one dropped partial line's width


class TestLogListing:
    def test_reports_name_size_and_mtime(self, tmp_path):
        p = tmp_path / "x.log"
        p.write_text("hello")
        [entry] = log_listing([p])
        assert entry["name"] == "x.log"
        assert entry["size"] == 5
        assert isinstance(entry["mtime"], float)

    def test_skips_a_path_that_no_longer_exists(self, tmp_path):
        missing = tmp_path / "gone.log"
        assert log_listing([missing]) == []

    def test_preserves_input_order(self, tmp_path):
        a, b = tmp_path / "a.log", tmp_path / "b.log"
        a.write_text("1"); b.write_text("22")
        out = log_listing([b, a])
        assert [e["name"] for e in out] == ["b.log", "a.log"]


class _LiveServer:
    """Runs the real Handler on an ephemeral local port for the handful of
    routing tests below -- these exercise do_GET's dispatch and HTTP-level
    behavior (status codes, content-type, 404s), which the pure-function
    tests above don't touch at all."""

    def __init__(self, tmp_path, log_tail_bytes=200_000):
        self.tmp_path = tmp_path
        self.dashboard_path = tmp_path / "dashboard.html"
        self.dashboard_path.write_text("<html>dashboard</html>")
        self.log_path = tmp_path / "logs" / "decisions.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text('{"type": "equity", "equity": 1.0}\n')
        handler = make_handler(self.dashboard_path, self.log_path, log_tail_bytes=log_tail_bytes)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def get(self, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def live_server(tmp_path):
    srv = _LiveServer(tmp_path)
    yield srv
    srv.close()


class TestLogsRoute:
    def test_lists_discovered_process_logs_as_json(self, live_server):
        (live_server.log_path.parent / "run_live_alpaca_stdout.log").write_text("hello\n")
        status, content_type, body = live_server.get("/logs")
        assert status == 200
        assert "application/json" in content_type
        names = [e["name"] for e in json.loads(body)]
        assert names == ["run_live_alpaca_stdout.log"]

    def test_excludes_decisions_jsonl_even_if_it_somehow_matched(self, live_server):
        status, _, body = live_server.get("/logs")
        assert status == 200
        assert json.loads(body) == []  # decisions.jsonl isn't *.log to begin with


class TestLogByNameRoute:
    def test_serves_the_tail_of_a_discovered_log(self, live_server):
        (live_server.log_path.parent / "serve_dashboard_stdout.log").write_text("Serving...\n")
        status, content_type, body = live_server.get("/log/serve_dashboard_stdout.log")
        assert status == 200
        assert "text/plain" in content_type
        assert body == b"Serving...\n"

    def test_404s_for_a_name_that_was_never_discovered(self, live_server):
        # Nothing on disk is named this -- must not fall back to reading
        # an arbitrary path from the URL.
        status, _, _ = live_server.get("/log/does_not_exist.log")
        assert status == 404

    def test_404s_for_a_path_traversal_attempt(self, live_server):
        (live_server.log_path.parent / "real.log").write_text("secret-looking, but fine to serve")
        outside = live_server.tmp_path / "outside.txt"
        outside.write_text("should never be reachable via /log/")
        status, _, body = live_server.get("/log/../outside.txt")
        # Either blocked at the HTTP layer (most likely) or, if it ever
        # reached our handler, rejected because "../outside.txt" (or its
        # normalized form) is not a name /logs discovered.
        assert status == 404 or b"should never be reachable" not in body

    def test_response_respects_the_configured_tail_cap(self, tmp_path):
        srv = _LiveServer(tmp_path, log_tail_bytes=100)
        try:
            big = "z" * 5000 + "\n"
            (srv.log_path.parent / "big_stdout.log").write_text(big)
            status, _, body = srv.get("/log/big_stdout.log")
            assert status == 200
            assert len(body) <= 101
        finally:
            srv.close()
