@echo off
setlocal

:: ---------------------------------------------------------------------
:: Configuration
::
:: If Task Scheduler reports "'python' is not recognized..." it's because
:: a task run "whether the user is logged on or not" doesn't load your
:: normal interactive PATH. Fix it by hardcoding your python.exe here --
:: find the right path by running `where python` in a normal terminal,
:: e.g.:
::   set PYTHON_EXE=C:\Users\steph\AppData\Local\Programs\Python\Python312\python.exe
:: Leave it as "python" to just use whatever PATH resolves at run time.
set PYTHON_EXE=python
:: ---------------------------------------------------------------------

cd /d "%~dp0.."
if not exist logs mkdir logs

"%PYTHON_EXE%" scripts\premarket_check.py >> logs\premarket_check.log 2>&1

endlocal
