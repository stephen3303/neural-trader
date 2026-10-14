import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.feed import CSVFeed, HistoricalReplayFeed


def _frame(times, start_price=100.0):
    idx = pd.DatetimeIndex(pd.to_datetime(times)).tz_localize("America/New_York")
    px = [start_price + k for k in range(len(idx))]
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)


def test_stream_merges_tickers_by_timestamp_when_one_has_a_gap():
    a = _frame(["2026-01-02 09:30", "2026-01-02 09:45", "2026-01-02 10:00"])
    b = _frame(["2026-01-02 09:30", "2026-01-02 10:00"])  # no 09:45 bar
    feed = HistoricalReplayFeed({"A": a, "B": b})
    bars = list(feed.stream(["A", "B"]))
    stamps = [b.timestamp for b in bars]
    assert stamps == sorted(stamps)
    assert [(b.ticker, b.timestamp.strftime("%H:%M")) for b in bars] == [
        ("A", "09:30"), ("B", "09:30"), ("A", "09:45"), ("A", "10:00"), ("B", "10:00")]


def test_replay_from_warms_up_on_earlier_bars_and_replays_from_start():
    times = pd.date_range("2026-01-02 09:30", periods=10, freq="15min")
    frame = _frame(times)
    feed = HistoricalReplayFeed({"A": frame}, replay_from=times[6])
    warm = feed.get_history("A", lookback=4)
    assert list(warm.index) == list(frame.index[2:6])
    assert [b.timestamp for b in feed.stream(["A"])] == list(frame.index[6:])


def test_csv_feed_start_keeps_history_for_warm_up(tmp_path):
    times = pd.date_range("2026-01-02 09:30", periods=10, freq="15min", tz="America/New_York")
    df = _frame(times.tz_localize(None))
    df.index.name = "timestamp"
    df.to_csv(tmp_path / "A.csv")
    feed = CSVFeed(["A"], str(tmp_path), start="2026-01-02 11:00")
    assert len(feed.get_history("A", lookback=400)) == 6
    assert feed.stream(["A"]).__next__().timestamp == times[6]
