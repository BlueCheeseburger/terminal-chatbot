# Gemini Legacy TUI

A lightweight terminal UI for Gemini 4 Argon, the Gemini Flash catalog, and the legacy PaLM chat model `chat-bison-001`.

It intentionally excludes Pro and Flash-Lite models. Models that Google no longer serves (404 or 501) are listed under "Unavailable" in the model picker, greyed out but still selectable.

## Run it on macOS or Linux

Python 3.9+ is enough; no package installation is needed. If the repository is in your Downloads folder, run:

```bash
cd ~/Downloads/terminal-chatbot
GEMINI_API_KEY="your-key" python3 gemini_legacy_tui.py
```

If you extracted a ZIP named `terminal-chatbot-main`, use that folder name instead. You can run the same commands from any other location by changing the first line.

The API key is read once, at launch, from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, or `--api-key-file`). There is no in-app key entry; to change keys, quit and relaunch. Your conversation is stored in the session file and does not depend on the key, so it carries over.

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

Without a key the app still opens (the mock model works), but real models report "No API key". See [WINDOWS.md](WINDOWS.md) for Windows troubleshooting and controls.

To load a key from a private file, use `python3 gemini_legacy_tui.py --api-key-file PATH`. The app sends API keys in Google's `x-goog-api-key` header, never places them in request URLs, and does not accept secrets as command-line values where process listings could expose them.

The app saves its transcript, chosen model, and system instruction in `./.gemini-legacy-tui/session.json` with private file permissions where the operating system supports them. It repairs malformed session fields when possible, refuses symlinked or oversized session files, and never writes an API key to disk. Supply another location with `--state-dir PATH`.

## Controls

| Control | Action |
| --- | --- |
| `F1` | Help (also in the Tab menu) |
| `Tab` | Main menu |
| `/` | Searchable slash-command menu |
| `Up` / `Down` or `Page Up` / `Page Down` | Scroll the chat (Up/Down move through a multi-line prompt first) |
| `Shift+Enter` | New line in the prompt; `Enter` sends. Needs terminal support (see below) |
| `F6` | Open the complete transcript reader |
| `Ctrl+P` / `Ctrl+O` | Recall previous / next submitted prompt |
| `/model` | Searchable model picker, including Custom Gemini model ID |
| `/thinking` | Thinking level and other generation parameters (temperature, top P, top K, max output tokens, seed, penalties, stop sequences) |
| `/permissions` | Choose how terminal commands are approved (only when Terminal access is on) |
| `/clear` | Clear the shared conversation context |
| `/retry` | Retry the most recent failed prompt (only listed after a prompt has failed) |
| `/restart` or `/reopen` | Close and reopen the TUI without rerunning the shell command |

Settings, Check availability, Help, Usage stats, and Exit live in the `Tab` menu. Usage stats shows all-time tokens in and out, the estimated total cost, and each model used; unlike the header it is never reset by `/clear`. The top line shows the selected model ID, the number of turns, the input and output tokens used in this conversation, the current context size over the model's context window (for example `context 4,321/1,048,576`, read from the API), and an estimated cost in dollars (all reset with `/clear`). The cost uses the prices on Google's pricing page, including announced price changes by date, and counts only tokens: it ignores caching, batch discounts, and tool fees. A trailing `+` means some usage came from a model with no known price, and `cost unknown` means all of it did.

The prompt box grows with each new line up to 5 lines tall, then scrolls. Plain `Enter` and `Shift+Enter` are the same byte in most terminals, so Shift+Enter only works where the terminal sends its own code. In macOS Terminal, map Shift-Return to the hex bytes `1B 5B 31 33 3B 32 75` (see [MACOS_TERMINAL_SHORTCUTS.md](MACOS_TERMINAL_SHORTCUTS.md)); kitty-style terminals that send `CSI 13;2u` work as is. `tools/show_keys.py` shows what your terminal sends. Ctrl+J and Alt+Enter are deliberately not shortcuts.
## Model catalog

```text
gemini-4-argon
gemini-3.8-flash
gemini-3.7-flash
gemini-3.6-flash
gemini-3-flash-preview
gemini-2.5-flash        (previously active users only)
gemini-flash-latest     (moving alias)
```

Greyed out in the picker (likely 404 or 501, but still selectable):

```text
gemini-3.5-flash
gemini-1.5-flash
gemini-1.5-flash-001
gemini-1.5-flash-002
gemini-2.0-flash
gemini-2.0-flash-001
gemini-2.0-flash-exp
chat-bison-001          (PaLM, HTTP 501)
```

The availability check is advisory: it uses the current `models.list` endpoint. A `404` or `403` means the model is not available to your API key.

## Terminal access

Settings has a **Terminal access** toggle, off by default. When it is on, `gemini-flash-latest`, Gemini 4 models (`gemini-4-*`), and Gemini 3.8 Flash and later can call a `run_terminal_command` tool. Other models (including custom IDs that do not match) ignore the toggle.

- Every command shows in a dialog and runs only after you press `Y`. `N` or `Esc` denies it, and `A` switches to "allow all safe" until you restart. Commands too long to display in full are denied.
- The dialog labels each command `READ`, `CREATION`, `WRITE`, or `REMOVAL` (a command mixing several types, like a read piped into a write, shows only the most severe one). Labels come from a best-effort command analysis, so unrecognized commands and scripts count as `WRITE` and show a note saying so. They are not treated as risky, which means "allow all safe" runs them without asking. Treat the label as a hint and read the command itself.
- `/permissions` (available when Terminal access is on) picks the mode: ask approval each time (default), allow all safe, or allow literally everything. The mode resets to "ask" when you restart or turn Terminal access off, and "everything" asks for one extra confirmation.
- In "allow all safe", removals and risky commands still ask, and the `A` option is hidden for them. Risky means: `sudo`/`su`, `git push`, `git reset --hard`, `git clean`, software installs, recursive permission changes, system commands (`kill`, `dd`, `launchctl`, ...), inline `python -c`-style code, piping into a shell, and writing outside the working directory. Removals are shown in red as `REMOVAL`.
- Commands run in the directory you launched the app from, with a 30 second timeout, no stdin, and output capped at 8,000 characters.
- Output over 8,000 characters is cut and ends with `[truncated at 8000 characters; N characters omitted]` (N is marked "about" when output was so large that part of it was discarded unread), so the model knows it is incomplete.
- The model gets at most 10 terminal results per turn. The 8th and 9th results end with a note saying how many accesses are left, the 10th says it was the last, and the model must then answer in text.
- Press `Esc` while a command runs to kill it and end the turn. While a turn is running the status line reads `Waiting for <model>...` or `Running: <command>` with a dim `press esc to stop turn.` beside it. While the app waits for Gemini's reply, Esc stops the turn after that step finishes. Other keys typed while a command runs are discarded.
- `GEMINI_API_KEY`, `GOOGLE_API_KEY`, and the restart key are removed from the command's environment.
- Tool use disables streaming for that reply. Commands that ran are noted at the top of the reply as `[terminal] ran ...` lines, shown dimmed so they stand apart from the model's own text.
- Approving a command gives it your full user permissions. Read each one before pressing `Y`.

## Testing without API credits

Pick **Mock model (local test server)** (`gemini-mock-model`) in `/model`. It needs no API key. The app starts `tools/mock_gemini_server.py` on `127.0.0.1:8765` by itself, stops it when you quit, and sends the placeholder key `mock` to it, never your real key. Turn on Terminal access in Settings and type `run ls -la` or `run rm notes.txt` to make the fake model request that command. Any other message gets an echo reply that also shows the generation settings it received, so `/thinking` changes are visible.

You can also run the server yourself with `python3 tools/mock_gemini_server.py` and point any model at it with `python3 gemini_legacy_tui.py --api-base-url http://127.0.0.1:8765`. That flag accepts only loopback `http://` addresses, so a real API key cannot be redirected to another host.

## Thinking and generation settings

`/thinking` opens a menu for the thinking level and the other parameters the API accepts for text: temperature, top P, top K, max output tokens, seed, presence and frequency penalties, and stop sequences. Only the values you set are sent. Leaving one unset uses the model's own default. Models that take a thinking level (Gemini 3 and later) get `thinkingLevel`; Gemini 2.5 models take a token budget instead (`-1` dynamic, `0` off). A saved level the current model does not accept is ignored instead of causing an API error. Thought summaries are never requested, and any thought parts the API returns are hidden from the chat.

## API behavior

Gemini entries stream by default through `v1beta/models/{model}:streamGenerateContent`; toggle Streaming off in Settings to use the non-streaming `generateContent` endpoint instead. `chat-bison-001` uses the old PaLM `v1beta3/models/chat-bison-001:generateMessage` request shape, which does not accept the modern Gemini `contents` schema.

Settings includes persistent terminal themes: Midnight Cyan, Matrix Green, Amber CRT, Arctic Blue, and Neon Noir. Matrix Green uses green text on a black background. Settings also accepts a custom Gemini model ID. Custom IDs use the modern Gemini API and appear on replies exactly as entered. Terminal controls font size rather than the TUI. macOS Terminal uses `Command-Plus` and `Command-Minus` and may resize its window while zooming; Windows Terminal uses `Ctrl-Plus` and `Ctrl-Minus`. A curses app cannot override the terminal application's font behavior.

New sessions include an editable system instruction that tells models they are in a terminal chat window: plain text only with no Markdown or emoji, answers that lead with the point, lists as simple lines, code indented four spaces, and an honest "I am not sure" instead of guessing. Open Settings to edit it in a full-screen multi-line editor (`Enter` adds a line, `Ctrl+D` saves, `Esc` cancels). Sessions that never customised it pick up improvements to the default automatically.

The transcript is shared when you switch Gemini models, so the selected model sees the prior conversation. Each assistant reply is tagged with the model that generated it. Use `/clear` to start without that context.

For macOS Terminal Command-key equivalents, see [MACOS_TERMINAL_SHORTCUTS.md](MACOS_TERMINAL_SHORTCUTS.md).

Sources: [Google Gemini models](https://ai.google.dev/gemini-api/docs/models), [Gemini model deprecations](https://ai.google.dev/gemini-api/docs/deprecations), and [Generative Language API models reference](https://ai.google.dev/api/models).
