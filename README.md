# Gemini Legacy TUI

A lightweight terminal UI for the Gemini Flash catalog you chose, plus the single PaLM chat candidate `chat-bison-001`.

It intentionally excludes every Pro model, every Flash-Lite model, `text-bison`, and the moving `chat-bison` alias. It does not provide LaMDA because Google never exposed LaMDA as a public API model.

## Run it on macOS or Linux

Python 3.9+ is enough; no package installation is needed. If the repository is in your Downloads folder, run:

```bash
cd ~/Downloads/terminal-chatbot
GEMINI_API_KEY="your-key" python3 gemini_legacy_tui.py
```

If you extracted a ZIP named `terminal-chatbot-main`, use that folder name instead. You can run the same commands from any other location by changing the first line.

You can also start it without an environment variable and press `F4` to enter a key for that process only:

```bash
python3 gemini_legacy_tui.py
```

## Run it on Windows

Use Python 3.9+ in Windows Terminal or PowerShell. The launcher installs the small `windows-curses` compatibility package automatically when it is missing:

```powershell
cd "$HOME\Downloads\terminal-chatbot"
.\run_windows.bat
```

If you extracted a ZIP named `terminal-chatbot-main`, use that folder name instead. You can also install and run it manually:

```powershell
py -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-key"
py .\gemini_legacy_tui.py
```

The API key is optional at launch on every platform; press `F4` inside the app to enter it for the current process. See [WINDOWS.md](WINDOWS.md) for Windows troubleshooting and controls.

The app saves its transcript, chosen model, and system instruction in `./.gemini-legacy-tui/session.json` with private file permissions where the operating system supports them. It repairs malformed session fields when possible and never writes an API key to disk. Supply another location with `--state-dir PATH`.

## Controls

| Control | Action |
| --- | --- |
| `F1` or `/help` | Help |
| `Tab` | Main menu |
| `/` | Searchable slash-command menu |
| `Page Up` / `Page Down` | Scroll through the reflowed transcript |
| `Up` / `Down` | Recall previously submitted prompts |
| `/model` | Searchable model picker |
| `/settings` | Streaming toggle, system instruction, API key, and availability check |
| `/clear` | Clear the shared conversation context |
| `/retry` | Retry the most recent failed prompt |
| `/restart` or `/reopen` | Close and reopen the TUI without rerunning the shell command |
| `/models` | Show the built-in catalog |
| `/quit` | Exit |

## Model catalog

Main Flash models:

```text
gemini-3.6-flash
gemini-3.5-flash
gemini-3-flash-preview
gemini-2.5-flash
gemini-flash-latest
```

Legacy Flash candidates:

```text
gemini-1.5-flash
gemini-1.5-flash-001
gemini-1.5-flash-002
gemini-2.0-flash
gemini-2.0-flash-001
gemini-2.0-flash-exp
```

PaLM chat candidate:

```text
chat-bison-001
```

The availability check is deliberately advisory: it uses the current `models.list` endpoint, while the PaLM entry uses its original legacy chat endpoint. Google retired the PaLM API and has also shut down some older Gemini Flash versions, so receiving a `404`, `403`, or “not found” message for those entries is expected for most current API keys.

## API behavior

Gemini entries stream by default through `v1beta/models/{model}:streamGenerateContent`; toggle Streaming off in Settings to use the non-streaming `generateContent` endpoint instead. `chat-bison-001` uses the old PaLM `v1beta3/models/chat-bison-001:generateMessage` request shape. Keeping those code paths separate is important: PaLM Chat was a conversational API and does not accept the modern Gemini `contents` schema.

Settings also accepts a custom Gemini model ID. Custom IDs use the modern Gemini API and appear on replies exactly as entered. Terminal controls font size rather than the TUI. macOS Terminal uses `Command-Plus` and `Command-Minus` and may resize its window while zooming; Windows Terminal uses `Ctrl-Plus` and `Ctrl-Minus`. A curses app cannot override the terminal application's font behavior.

The transcript is shared when you switch Gemini models, so the selected model sees the prior conversation. Each assistant reply is tagged with the model that generated it. Use `Ctrl+N`, `/new`, or `/clear` to start without that context.

For macOS Terminal Command-key equivalents, see [MACOS_TERMINAL_SHORTCUTS.md](MACOS_TERMINAL_SHORTCUTS.md).

Sources: [Google Gemini models](https://ai.google.dev/gemini-api/docs/models), [Gemini model deprecations](https://ai.google.dev/gemini-api/docs/deprecations), and [Generative Language API models reference](https://ai.google.dev/api/models).
