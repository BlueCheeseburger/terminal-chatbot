@echo off
setlocal
pushd "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    set "PYTHON_COMMAND=python"
) else (
    set "PYTHON_COMMAND=py"
)

%PYTHON_COMMAND% -c "import curses" >nul 2>nul
if errorlevel 1 (
    echo Installing Windows terminal support...
    %PYTHON_COMMAND% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Unable to install Windows terminal support.
        popd
        exit /b 1
    )
)

%PYTHON_COMMAND% gemini_legacy_tui.py %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %APP_EXIT_CODE%
