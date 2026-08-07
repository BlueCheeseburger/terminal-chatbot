# Gemini Legacy TUI

A dependency-free terminal UI for the Gemini Flash catalog you chose, plus the single PaLM chat candidate `chat-bison-001`.

It intentionally excludes every Pro model, every Flash-Lite model, `text-bison`, and the moving `chat-bison` alias. It does not provide LaMDA because Google never exposed LaMDA as a public API model.

## Run it

Python 3.9+ is enough; no package installation is needed.

```bash
cd terminal-chatbot
GEMINI_API_KEY="your-key" python3 gemini_legacy_tui.py
```

You can also start it without an environment variable and press `F4` to enter a key for that process only:

```bash
python3 gemini_legacy_tui.py
```

The app saves its transcript, chosen model, and system instruction in `./.gemini-legacy-tui/session.json`. It never writes an API key to disk. Supply another location with `--state-dir PATH`.

## Controls

| Control | Action |
| --- | --- |
| `F1` or `/help` | Help |
| `Tab` | Main menu |
| `/` | Searchable slash-command menu |
| `/model` | Searchable model picker |
| `/settings` | Streaming toggle, system instruction, API key, and availability check |
| `/clear` | Clear the shared conversation context |
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

Settings also accepts a custom Gemini model ID. Custom IDs use the modern Gemini API and appear on replies exactly as entered. Terminal controls font size rather than the TUI: in macOS Terminal, use `Command-Plus` to enlarge and `Command-Minus` to reduce the font size. Terminal keeps its character grid stable while zooming, so this can also resize the macOS window; a curses app cannot override that behavior.

The transcript is shared when you switch Gemini models, so the selected model sees the prior conversation. Each assistant reply is tagged with the model that generated it. Use `Ctrl+N`, `/new`, or `/clear` to start without that context.

For macOS Terminal Command-key equivalents, see [MACOS_TERMINAL_SHORTCUTS.md](MACOS_TERMINAL_SHORTCUTS.md).

Sources: [Google Gemini models](https://ai.google.dev/gemini-api/docs/models), [Gemini model deprecations](https://ai.google.dev/gemini-api/docs/deprecations), and [Generative Language API models reference](https://ai.google.dev/api/models).
