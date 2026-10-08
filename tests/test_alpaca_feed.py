import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import queue as queue_mod

import pandas as pd

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
