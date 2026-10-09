"""Tests for scripts/run_live_alpaca.py's two guards against Alpaca API
calls raising -- specifically, against `trading_client.get_clock()` raising
a transient network/DNS error (observed live: a `requests.exceptions.
ConnectionError` from a brief failure to resolve paper-api.alpaca.markets
took down the entire run_live_alpaca.py process, recovering only on the
next external restart -- see the README's "fifteenth finding").

Both wait_for_market_open() and market_is_closed() are plain top-level
functions taking a trading_client, so they're testable directly with a
fake client -- no real Alpaca credentials, network, or Orchestrator
needed.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_alpaca import market_is_closed, wait_for_market_open


class _Clock:
    def __init__(self, is_open, next_open="2024-01-02T09:30:00"):
        self.is_open = is_open
        self.next_open = next_open


class _FlakyClient:
    """A fake trading_client whose get_clock() raises for the first
    `fail_times` calls (simulating a transient DNS/connection error), then
    returns `final_clock` on every call after that."""

    def __init__(self, fail_times, final_clock, exc=None):
        self.fail_times = fail_times
        self.final_clock = final_clock
        self.exc = exc or ConnectionError("Failed to resolve 'paper-api.alpaca.markets'")
        self.calls = 0

    def get_clock(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return self.final_clock


class TestWaitForMarketOpenSurvivesApiErrors:
    def test_a_transient_get_clock_error_is_retried_not_raised(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        client = _FlakyClient(fail_times=2, final_clock=_Clock(is_open=True))
        # Must not raise, and must eventually return once get_clock()
        # starts succeeding again.
        wait_for_market_open(client, poll_seconds=1.0)
        assert client.calls == 3

    def test_market_still_closed_after_the_error_clears_keeps_polling(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        clocks_after_error = [_Clock(is_open=False), _Clock(is_open=False), _Clock(is_open=True)]
        client = _FlakyClient(fail_times=1, final_clock=None)
        # Swap in a get_clock() that raises once, then returns closed,
        # closed, open -- confirms the error-retry path and the normal
        # closed-market-retry path compose correctly.
        calls = {"n": 0}

        def get_clock():
            calls["n"] += 1
            if calls["n"] == 1:
                raise client.exc
            return clocks_after_error.pop(0)

        client.get_clock = get_clock
        wait_for_market_open(client, poll_seconds=5.0)
        assert calls["n"] == 4
        assert sleeps == [5.0, 5.0, 5.0]

    def test_a_non_network_exception_is_also_swallowed_and_retried(self, monkeypatch):
        # The guard is a broad `except Exception`, matching this codebase's
        # existing convention in AlpacaBroker._submit_market_order -- not
        # narrowed to requests.exceptions.RequestException -- so any
        # exception from get_clock() (not just a connection error) must not
        # crash the process before trading has even started.
        monkeypatch.setattr(time, "sleep", lambda _s: None)
        client = _FlakyClient(fail_times=1, final_clock=_Clock(is_open=True), exc=RuntimeError("boom"))
        wait_for_market_open(client, poll_seconds=1.0)
        assert client.calls == 2


class TestMarketIsClosedSurvivesApiErrors:
    def test_a_transient_get_clock_error_is_treated_as_still_open(self):
        # This is the fix for the actual crash: market_is_closed() is
        # passed as Orchestrator.run()'s stop_check and polled once per
        # bar for as long as the process is alive. Before this fix, a
        # raise here propagated straight out of run() and killed the
        # whole live-trading loop.
        client = _FlakyClient(fail_times=1, final_clock=_Clock(is_open=True))
        assert market_is_closed(client) is False
        assert client.calls == 1

    def test_a_real_open_clock_reports_not_closed(self):
        client = _FlakyClient(fail_times=0, final_clock=_Clock(is_open=True))
        assert market_is_closed(client) is False

    def test_a_real_closed_clock_reports_closed(self):
        client = _FlakyClient(fail_times=0, final_clock=_Clock(is_open=False))
        assert market_is_closed(client) is True
