# Windows Setup

Use Python 3.9 or newer in Windows Terminal or PowerShell. If the project is in the usual Downloads location:

```powershell
cd "$HOME\Downloads\terminal-chatbot"
.\run_windows.bat
```

The launcher uses the Python Launcher (`py`) when available and falls back to `python`. It installs `windows-curses` if the standard Python installation does not provide `curses`, then starts the TUI. It does not save your API key.

To install the dependency yourself instead:

```powershell
py -m pip install -r requirements.txt
py .\gemini_legacy_tui.py
```

Set a key for the current PowerShell session with `$env:GEMINI_API_KEY = "your-key"`, or press `F4` after the app opens. Use `Ctrl+Plus` and `Ctrl+Minus` to adjust Windows Terminal's font. App controls such as `Ctrl+L` for models and `Ctrl+N` for a new context use Control on Windows.

If `py` and `python` are both unavailable, install Python 3.9 or newer and enable the option that adds Python to `PATH`. If the folder has a different name or location, change the `cd` command accordingly.
