import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import queue as queue_mod

import pandas as pd

import src.data.alpaca_feed as alpaca_feed_mod
from src.data.alpaca_feed import AlpacaLiveFeed
from src.data.feed import Bar


def _make_feed(tickers):
    # Bypasses __init__ (which imports the `alpaca` package and constructs a
    # real StockHistoricalDataClient) so this test doesn't need network
    # access or real API keys -- only stream()'s own bookkeeping is under
    # test here.
    feed = object.__new__(AlpacaLiveFeed)
    feed.tickers = list(tickers)
    feed.api_key = "test-key"
    feed.secret_key = "test-secret"
    feed.data_feed = None
    feed.history_days = 10
    feed.reconnect_backoff_s = 5.0
    feed._hist_client = None
    feed._bar_queue = queue_mod.Queue()
    # Truthy sentinel (not None): stream() must see this and skip spinning
    # up a real background thread (_run_stream_forever would otherwise try
    # to import the `alpaca` package and open a real websocket).
    feed._thread = object()
    feed._cursor = {t: 0 for t in feed.tickers}
    return feed


def _bar(ticker, minute_offset=0):
    ts = pd.Timestamp("2024-01-01 09:30:00") + pd.Timedelta(minutes=minute_offset)
    return Bar(ticker=ticker, timestamp=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)


class TestStreamDoesNotChangeBarDelivery:
    def test_yields_the_exact_bar_put_on_the_queue(self):
        feed = _make_feed(["AAPL"])
        feed._bar_queue.put(_bar("AAPL"))
        bar = next(feed.stream(feed.tickers))
        assert bar.ticker == "AAPL"
        assert bar.close == 1.0

    def test_yields_multiple_bars_in_order(self):
        feed = _make_feed(["AAPL", "MSFT"])
        feed._bar_queue.put(_bar("AAPL", 0))
        feed._bar_queue.put(_bar("MSFT", 0))
        feed._bar_queue.put(_bar("AAPL", 1))
        gen = feed.stream(feed.tickers)
        got = [next(gen).ticker for _ in range(3)]
        assert got == ["AAPL", "MSFT", "AAPL"]

    def test_does_not_touch_thread_when_already_set(self):
        # Regression guard for the tracking added alongside this test: the
        # new first-bar bookkeeping in stream() must stay purely additive
        # and never interfere with the existing "only start the background
        # thread once" logic.
        feed = _make_feed(["AAPL"])
        sentinel = feed._thread
        feed._bar_queue.put(_bar("AAPL"))
        next(feed.stream(feed.tickers))
        assert feed._thread is sentinel


class TestStreamFirstBarTracking:
    """Added after a live incident ("tickers missing from the Performance
    Summary"): 9 of 12 configured tickers never received a single bar from
    Alpaca's websocket, and there was no way to tell from the log which
    tickers *were* getting through without writing a one-off script to
    parse decisions.jsonl. stream() now prints a one-line, one-time note
    the first time each ticker's bar arrives, so that's visible directly
    in run_live_alpaca_stdout.log going forward."""

    def test_prints_once_per_ticker_on_first_bar(self, capsys):
        feed = _make_feed(["AAPL", "MSFT", "GOOGL"])
        feed._bar_queue.put(_bar("AAPL", 0))
        feed._bar_queue.put(_bar("AAPL", 1))  # second AAPL bar -- must not re-print
        feed._bar_queue.put(_bar("MSFT", 0))
        gen = feed.stream(feed.tickers)
        for _ in range(3):
            next(gen)
        out = capsys.readouterr().out
        assert out.count("first live bar received for AAPL") == 1
        assert out.count("first live bar received for MSFT") == 1

    def test_never_mentions_a_ticker_that_never_streamed(self, capsys):
        # The exact symptom this is meant to surface: GOOGL is subscribed
        # but (in this test, deliberately) never produces a bar.
        feed = _make_feed(["AAPL", "MSFT", "GOOGL"])
        feed._bar_queue.put(_bar("AAPL", 0))
        next(feed.stream(feed.tickers))
        out = capsys.readouterr().out
        assert "GOOGL" not in out

    def test_progress_fraction_is_against_the_full_subscribed_list(self, capsys):
        feed = _make_feed(["AAPL", "MSFT", "GOOGL"])
        feed._bar_queue.put(_bar("AAPL", 0))
        feed._bar_queue.put(_bar("MSFT", 0))
        gen = feed.stream(feed.tickers)
        next(gen)
        next(gen)
        out = capsys.readouterr().out
        assert "(1/3 tickers streaming" in out
        assert "(2/3 tickers streaming" in out


class _FakeNow:
    """Swapped in for the module's `datetime` name so get_history()'s
    `datetime.now(timezone.utc)` call returns a fixed instant instead of
    the real wall clock, without needing to subclass the real datetime."""
    def __init__(self, fixed):
        self._fixed = fixed

    def now(self, tz=None):
        return self._fixed


class _FakeHistBar:
    def __init__(self, timestamp, close=1.0):
        self.timestamp = timestamp
        self.open = self.high = self.low = self.close = close
        self.volume = 1.0


class _FakeBarSet:
    """Minimal stand-in exposing just the `.data` attribute get_history()
    actually reads -- real BarSet-shape coverage (and the pydantic-
    __iter__-shadows-`in` bug that bit `ticker in barset` in production)
    lives in TestGetHistoryExtractsBarsFromRealBarSet below, which uses
    the real alpaca-py BarSet class instead of this fake."""
    def __init__(self, data):
        self.data = dict(data)


class _CapturingHistClient:
    """Stands in for StockHistoricalDataClient: hands back a configured
    response per call and remembers every request it was asked for, so a
    test can assert on the exact start/end window(s) used. `responses` is
    either a single {ticker: [bars]} dict (returned on every call, for
    the common single-request case) or a list of such dicts consumed one
    per call in order (to simulate the today-anchored request coming back
    empty and a fallback request coming back with bars)."""
    def __init__(self, responses):
        self._responses = responses
        self.requests = []

    def get_stock_bars(self, req):
        self.requests.append(req)
        if isinstance(self._responses, list):
            idx = min(len(self.requests) - 1, len(self._responses) - 1)
            return _FakeBarSet(self._responses[idx])
        return _FakeBarSet(self._responses)

    @property
    def last_request(self):
        return self.requests[-1] if self.requests else None


class TestTodaysSessionOpenUtc:
    """_todays_session_open_utc() backs the fix for a real incident: two
    tickers warming up at the exact same moment came back with wildly
    different bar counts (118 vs 65) because get_history() was pulling a
    rolling multi-day window and `limit` could get eaten by sparse older
    days before reaching today's bars at all. Anchoring to today's actual
    session open removes that inconsistency."""

    def test_january_is_1430_utc_est(self):
        # US/Eastern is EST (UTC-5) in January, so 09:30 ET = 14:30 UTC.
        now = datetime(2024, 1, 15, 18, 0, tzinfo=timezone.utc)
        assert AlpacaLiveFeed._todays_session_open_utc(now) == datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

    def test_july_is_1330_utc_edt(self):
        # US/Eastern is EDT (UTC-4) in July, so 09:30 ET = 13:30 UTC.
        now = datetime(2024, 7, 15, 18, 0, tzinfo=timezone.utc)
        assert AlpacaLiveFeed._todays_session_open_utc(now) == datetime(2024, 7, 15, 13, 30, tzinfo=timezone.utc)

    def test_uses_now_utcs_own_calendar_date(self):
        # Just after midnight UTC is still the *previous* calendar day in
        # New York -- the open returned must track the Eastern date, not
        # the UTC one.
        now = datetime(2024, 1, 16, 2, 0, tzinfo=timezone.utc)  # 2024-01-15 21:00 ET
        assert AlpacaLiveFeed._todays_session_open_utc(now) == datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)


class TestGetHistoryAnchorsToTodaysOpen:
    def test_after_open_uses_todays_open_as_start(self, monkeypatch):
        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)  # 11:00 ET, well after open
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])
        hist_client = _CapturingHistClient({"AAPL": [_FakeHistBar(pd.Timestamp("2024-01-15T15:00:00Z"))]})
        feed._hist_client = hist_client

        feed.get_history("AAPL", 400)

        # alpaca-py's StockBarsRequest (a pydantic model) stores these as
        # naive datetimes -- comparing the naive wall-clock value is still
        # exactly what matters here: did it get today's 09:30 ET open.
        assert hist_client.last_request.start == datetime(2024, 1, 15, 14, 30)
        assert hist_client.last_request.end == fixed_now.replace(tzinfo=None)

    def test_before_open_falls_back_to_multi_day_lookback(self, monkeypatch):
        fixed_now = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)  # 06:00 ET, premarket
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])
        feed.history_days = 10
        hist_client = _CapturingHistClient({"AAPL": [_FakeHistBar(pd.Timestamp("2024-01-10T15:00:00Z"))]})
        feed._hist_client = hist_client

        feed.get_history("AAPL", 400)

        # Today's 09:30 ET open hasn't happened yet, so start must fall
        # back to the old history_days-based window instead of a start
        # that's after end.
        assert hist_client.last_request.start < hist_client.last_request.end
        assert hist_client.last_request.start == (fixed_now - alpaca_feed_mod.timedelta(days=10)).replace(tzinfo=None)

    def test_two_tickers_same_moment_get_the_identical_window(self, monkeypatch):
        # The exact symptom from the incident: every ticker warming up at
        # the same instant must be given the same [start, end) window,
        # whatever bars they individually have within it.
        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL", "THIN"])
        liquid_client = _CapturingHistClient({"AAPL": [_FakeHistBar(pd.Timestamp("2024-01-15T15:00:00Z"))]})
        thin_client = _CapturingHistClient({"THIN": [_FakeHistBar(pd.Timestamp("2024-01-15T15:00:00Z"))]})

        feed._hist_client = liquid_client
        feed.get_history("AAPL", 400)
        feed._hist_client = thin_client
        feed.get_history("THIN", 400)

        assert liquid_client.last_request.start == thin_client.last_request.start
        assert liquid_client.last_request.end == thin_client.last_request.end


class TestGetHistoryFallsBackWhenTodaysWindowIsEmpty:
    """Live regression caught right after the today-open-anchoring fix
    shipped and the trading processes were restarted: get_history() came
    back with ZERO bars for every ticker -- worse than before the fix --
    hours after the market had opened. Alpaca's historical bars endpoint
    apparently had nothing for the still-in-progress current session via
    this call, even well past the open. get_history() must fall back to
    the old multi-day window whenever the today-anchored request comes
    back empty, not only in the premarket case, so a day/account where
    today's bars aren't being served this way doesn't regress to nothing."""

    def test_falls_back_to_multi_day_window_when_todays_window_is_empty(self, monkeypatch):
        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)  # well after today's open
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])
        feed.history_days = 10
        hist_client = _CapturingHistClient([
            {"AAPL": []},  # today-anchored attempt: nothing
            {"AAPL": [_FakeHistBar(pd.Timestamp("2024-01-10T15:00:00Z"))]},  # fallback: real bars
        ])
        feed._hist_client = hist_client

        df = feed.get_history("AAPL", 400)

        assert len(hist_client.requests) == 2
        assert hist_client.requests[0].start == datetime(2024, 1, 15, 14, 30)
        assert hist_client.requests[1].start == (fixed_now - alpaca_feed_mod.timedelta(days=10)).replace(tzinfo=None)
        assert len(df) == 1

    def test_does_not_fall_back_when_todays_window_already_has_bars(self, monkeypatch):
        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])
        hist_client = _CapturingHistClient({"AAPL": [_FakeHistBar(pd.Timestamp("2024-01-15T15:00:00Z"))]})
        feed._hist_client = hist_client

        feed.get_history("AAPL", 400)

        assert len(hist_client.requests) == 1

    def test_before_open_makes_only_the_fallback_request_not_two(self, monkeypatch):
        fixed_now = datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc)  # 06:00 ET, premarket
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])
        hist_client = _CapturingHistClient({"AAPL": [_FakeHistBar(pd.Timestamp("2024-01-10T15:00:00Z"))]})
        feed._hist_client = hist_client

        feed.get_history("AAPL", 400)

        # session_open > end here, so the today-anchored branch is never
        # attempted at all -- only the premarket fallback request fires.
        assert len(hist_client.requests) == 1
        assert hist_client.requests[0].start == (fixed_now - alpaca_feed_mod.timedelta(days=10)).replace(tzinfo=None)

    def test_both_requests_empty_still_returns_an_empty_frame_not_an_error(self, monkeypatch):
        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])
        hist_client = _CapturingHistClient([{"AAPL": []}, {"AAPL": []}])
        feed._hist_client = hist_client

        df = feed.get_history("AAPL", 400)

        assert len(hist_client.requests) == 2
        assert len(df) == 0


class TestGetHistoryExtractsBarsFromRealBarSet:
    """Regression test for the actual root cause of "get_history() always
    returns no bars", found live: `barset[ticker] if ticker in barset
    else []` always took the empty branch, because `BarSet` is a pydantic
    BaseModel and pydantic's own __iter__ (over the model's field names,
    for dict(model) support) shadows dict-like "in" membership -- a
    ticker string can never match one of those (field_name, value) pairs,
    so `ticker in barset` was unconditionally False regardless of what
    Alpaca actually returned. Every other test in this file uses a plain
    dict as the fake response, which doesn't reproduce this at all (a
    plain dict's `in` works normally) -- that's exactly how this slipped
    through. This test goes through the real alpaca-py BarSet class
    instead, so this bug can't silently come back."""

    def test_real_barset_with_bars_is_actually_extracted(self, monkeypatch):
        from alpaca.data.models.bars import BarSet

        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])

        raw = {"AAPL": [
            {"t": "2024-01-15T15:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0, "n": 1, "vw": 1.0},
            {"t": "2024-01-15T15:01:00Z", "o": 2.0, "h": 2.0, "l": 2.0, "c": 2.0, "v": 2.0, "n": 1, "vw": 2.0},
        ]}

        class _RealBarSetHistClient:
            def __init__(self, raw_data):
                self._raw_data = raw_data
                self.requests = []

            def get_stock_bars(self, req):
                self.requests.append(req)
                return BarSet(self._raw_data)

        hist_client = _RealBarSetHistClient(raw)
        feed._hist_client = hist_client

        df = feed.get_history("AAPL", 400)

        assert len(hist_client.requests) == 1  # today's window had bars -- no fallback needed
        assert len(df) == 2
        assert list(df["close"]) == [1.0, 2.0]

    def test_real_barset_with_no_key_for_ticker_falls_back_cleanly(self, monkeypatch):
        from alpaca.data.models.bars import BarSet

        fixed_now = datetime(2024, 1, 15, 16, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(alpaca_feed_mod, "datetime", _FakeNow(fixed_now))
        feed = _make_feed(["AAPL"])

        class _RealBarSetHistClient:
            def __init__(self):
                self.calls = 0

            def get_stock_bars(self, req):
                self.calls += 1
                # First (today-anchored) call: no bars at all for AAPL.
                # Second (fallback) call: real bars.
                if self.calls == 1:
                    return BarSet({})
                return BarSet({"AAPL": [{"t": "2024-01-10T15:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0, "n": 1, "vw": 1.0}]})

        hist_client = _RealBarSetHistClient()
        feed._hist_client = hist_client

        df = feed.get_history("AAPL", 400)

        assert hist_client.calls == 2
        assert len(df) == 1
