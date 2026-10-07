@echo off
setlocal

:: Same PYTHON_EXE override as premarket_check.bat, if needed -- see that
:: file's comment for why/when you'd set this.
set PYTHON_EXE=python

cd /d "%~dp0.."
"%PYTHON_EXE%" scripts\stop_trading.py %*

echo.
pause
endlocal
