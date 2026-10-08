import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import premarket_check


def _fake_result(stdout):
    return MagicMock(stdout=stdout, stderr="")


class TestIsRunning:
    def test_false_when_not_windows(self):
        with patch.object(premarket_check.os, "name", "posix"):
            assert premarket_check.is_running("run_live_alpaca.py") is False

    def test_matches_python_exe_command_line(self):
        result = _fake_result("C:\\Python\\python.exe scripts\\run_live_alpaca.py\n")
        with patch.object(premarket_check.os, "name", "nt"), \
                patch.object(premarket_check.subprocess, "run", return_value=result):
            assert premarket_check.is_running("run_live_alpaca.py") is True

    def test_matches_py_launcher_command_line(self):
        # This is exactly the gap the fix closes: a process started by hand
        # with the Windows `py` launcher (py.exe, not python.exe) used to be
        # invisible to this check. That let a duplicate run_live_alpaca.py
        # run alongside an automatically-started one -- and since Alpaca's
        # live data websocket allows only one concurrent connection per API
        # key, the duplicate silently starved most configured tickers of
        # any live bars at all (the incident that prompted this fix: 9 of
        # 12 tickers never streamed a single bar while a leftover process
        # from an older, shorter ticker list kept the one connection slot).
        result = _fake_result("C:\\Python\\py.exe scripts\\run_live_alpaca.py\n")
        with patch.object(premarket_check.os, "name", "nt"), \
                patch.object(premarket_check.subprocess, "run", return_value=result):
            assert premarket_check.is_running("run_live_alpaca.py") is True

    def test_matches_pythonw_exe_command_line(self):
        result = _fake_result("C:\\Python\\pythonw.exe scripts\\run_live_alpaca.py\n")
        with patch.object(premarket_check.os, "name", "nt"), \
                patch.object(premarket_check.subprocess, "run", return_value=result):
            assert premarket_check.is_running("run_live_alpaca.py") is True

    def test_false_when_no_matching_command_line(self):
        result = _fake_result("C:\\Python\\python.exe scripts\\serve_dashboard.py\n")
        with patch.object(premarket_check.os, "name", "nt"), \
                patch.object(premarket_check.subprocess, "run", return_value=result):
            assert premarket_check.is_running("run_live_alpaca.py") is False

    def test_false_on_subprocess_error(self):
        with patch.object(premarket_check.os, "name", "nt"), \
                patch.object(premarket_check.subprocess, "run", side_effect=OSError("boom")):
            assert premarket_check.is_running("run_live_alpaca.py") is False

    def test_ps_filter_includes_all_three_process_names(self):
        result = _fake_result("")
        with patch.object(premarket_check.os, "name", "nt"), \
                patch.object(premarket_check.subprocess, "run", return_value=result) as mock_run:
            premarket_check.is_running("run_live_alpaca.py")
            ps_cmd = mock_run.call_args[0][0][-1]
            assert "python.exe" in ps_cmd
            assert "pythonw.exe" in ps_cmd
            assert "py.exe" in ps_cmd
