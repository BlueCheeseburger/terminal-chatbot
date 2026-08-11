#!/usr/bin/env python3
"""A lightweight terminal chat client for Gemini Flash and legacy PaLM Chat."""

import argparse
import json
import os
import stat
import sys
import tempfile
import textwrap
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import curses
except ImportError as error:
    if os.name == "nt":
        raise SystemExit(
            "Windows terminal support is not installed. Run: py -m pip install -r requirements.txt"
        ) from error
    raise


APP_NAME = "Gemini Legacy TUI"
STATE_FILE_NAME = "session.json"
USER_AGENT = "gemini-legacy-tui/1.0"
MAX_API_ERROR_BYTES = 64 * 1024
MAX_API_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_API_KEY_CHARS = 1024
MAX_API_KEY_FILE_BYTES = 4 * 1024
MAX_API_REQUEST_BYTES = 8 * 1024 * 1024
MAX_DIALOG_CHARS = 64 * 1024
MAX_INPUT_CHARS = 64 * 1024
MAX_MODEL_ID_CHARS = 256
MAX_SESSION_BYTES = 8 * 1024 * 1024
MAX_CHARACTER_STREAM_CHUNK_CHARS = 512
MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK = 64
STREAM_RENDER_DELAY_MS = 12
MAX_STREAM_EVENT_BYTES = 1024 * 1024
MAX_STREAM_TEXT_CHARS = 2 * 1024 * 1024


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    label: str
    group: str
    protocol: str  # gemini or palm
    availability_note: str


@dataclass(frozen=True)
class ThemeSpec:
    identifier: str
    label: str
    description: str
    header: int
    accent: int
    text: int
    user: int
    warning: int
    error: int
    selected_text: int
    selected_background: int


THEMES: Sequence[ThemeSpec] = (
    ThemeSpec(
        "midnight",
        "Midnight Cyan",
        "Cyan signals on a calm black canvas",
        curses.COLOR_CYAN,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_RED,
        curses.COLOR_BLACK,
        curses.COLOR_CYAN,
    ),
    ThemeSpec(
        "matrix",
        "Matrix Green",
        "Green phosphor text on pure black",
        curses.COLOR_GREEN,
        curses.COLOR_GREEN,
        curses.COLOR_GREEN,
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_RED,
        curses.COLOR_BLACK,
        curses.COLOR_GREEN,
    ),
    ThemeSpec(
        "amber",
        "Amber CRT",
        "Warm amber prompts with a crisp monochrome body",
        curses.COLOR_YELLOW,
        curses.COLOR_YELLOW,
        curses.COLOR_WHITE,
        curses.COLOR_YELLOW,
        curses.COLOR_MAGENTA,
        curses.COLOR_RED,
        curses.COLOR_BLACK,
        curses.COLOR_YELLOW,
    ),
    ThemeSpec(
        "arctic",
        "Arctic Blue",
        "Cool blue structure with cyan conversation markers",
        curses.COLOR_BLUE,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
        curses.COLOR_CYAN,
        curses.COLOR_YELLOW,
        curses.COLOR_RED,
        curses.COLOR_WHITE,
        curses.COLOR_BLUE,
    ),
    ThemeSpec(
        "neon",
        "Neon Noir",
        "Magenta headers and cyan responses on black",
        curses.COLOR_MAGENTA,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_RED,
        curses.COLOR_BLACK,
        curses.COLOR_MAGENTA,
    ),
)
THEME_BY_ID = {theme.identifier: theme for theme in THEMES}
DEFAULT_THEME_ID = "midnight"


MODEL_CATALOG: Sequence[ModelSpec] = (
    ModelSpec("gemini-3.6-flash", "Gemini 3.6 Flash", "Main Gemini Flash", "gemini", "Current stable"),
    ModelSpec("gemini-3.5-flash", "Gemini 3.5 Flash", "Main Gemini Flash", "gemini", "Current stable"),
    ModelSpec("gemini-3-flash-preview", "Gemini 3 Flash", "Main Gemini Flash", "gemini", "Preview"),
    ModelSpec("gemini-2.5-flash", "Gemini 2.5 Flash", "Main Gemini Flash", "gemini", "Current stable"),
    ModelSpec("gemini-flash-latest", "Gemini Flash (latest alias)", "Main Gemini Flash", "gemini", "Moving alias"),
    ModelSpec("gemini-1.5-flash", "Gemini 1.5 Flash", "Legacy Gemini Flash", "gemini", "Legacy; shut down by Google"),
    ModelSpec("gemini-1.5-flash-001", "Gemini 1.5 Flash 001", "Legacy Gemini Flash", "gemini", "Legacy; likely unavailable"),
    ModelSpec("gemini-1.5-flash-002", "Gemini 1.5 Flash 002", "Legacy Gemini Flash", "gemini", "Legacy; likely unavailable"),
    ModelSpec("gemini-2.0-flash", "Gemini 2.0 Flash", "Legacy Gemini Flash", "gemini", "Legacy; availability varies"),
    ModelSpec("gemini-2.0-flash-001", "Gemini 2.0 Flash 001", "Legacy Gemini Flash", "gemini", "Legacy; availability varies"),
    ModelSpec("gemini-2.0-flash-exp", "Gemini 2.0 Flash Experimental", "Legacy Gemini Flash", "gemini", "Experimental; likely unavailable"),
    ModelSpec("chat-bison-001", "PaLM 2 Chat-Bison 001", "PaLM Chat", "palm", "Legacy; shut down by Google"),
)

MODEL_BY_ID = {model.identifier: model for model in MODEL_CATALOG}
DEFAULT_MODEL_ID = "gemini-2.5-flash"
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are replying inside a terminal chat interface. Use plain text only and do not use "
    "Markdown emphasis or other visual styling, including bold, italics, headings, block quotes, "
    "tables, or decorative separators. Blank lines between distinct blocks are allowed when they "
    "make the response easier to read."
)
SLASH_COMMANDS: Sequence[Tuple[str, str]] = (
    ("/clear", "Clear the shared conversation context"),
    ("/new", "Start a new conversation context"),
    ("/model", "Choose a model"),
    ("/models", "Show the configured model catalog"),
    ("/settings", "Open settings"),
    ("/transcript", "Read the complete transcript"),
    ("/retry", "Retry the last failed prompt"),
    ("/system", "Set the system instruction"),
    ("/key", "Enter an API key for this run"),
    ("/check", "Check model availability"),
    ("/restart", "Close and reopen the TUI"),
    ("/help", "Show help and shortcuts"),
    ("/quit", "Exit Gemini Legacy"),
)


class ApiError(RuntimeError):
    """A clean message for API and transport errors."""


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep credential headers on the fixed API origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


GOOGLE_API_OPENER = urllib.request.build_opener(NoRedirectHandler())


def validate_api_key(value: str) -> str:
    key = value.strip()
    if len(key) > MAX_API_KEY_CHARS:
        raise ApiError("The API key is too long.")
    if any(character.isspace() or unicodedata.category(character).startswith("C") for character in key):
        raise ApiError("The API key contains invalid characters.")
    return key


def validate_model_id(value: str) -> str:
    identifier = value.strip()
    if not identifier or len(identifier) > MAX_MODEL_ID_CHARS:
        raise ApiError("The model ID must be between 1 and {} characters.".format(MAX_MODEL_ID_CHARS))
    if not all(
        character.isascii() and (character.isalnum() or character in ("-", "_", "."))
        for character in identifier
    ):
        raise ApiError("Model IDs can contain only letters, numbers, hyphens, underscores, and periods.")
    return identifier


def read_api_key_file(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(MAX_API_KEY_FILE_BYTES + 1)
    if len(raw) > MAX_API_KEY_FILE_BYTES:
        raise ValueError("the file is too large")
    return validate_api_key(raw.decode("utf-8"))


class GoogleApiClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = validate_api_key(api_key)

    def _request(self, url: str, body: Optional[dict] = None) -> dict:
        if not self.api_key:
            raise ApiError("No API key. Set GEMINI_API_KEY or press F4 to enter one.")

        headers = {"User-Agent": USER_AGENT, "x-goog-api-key": self.api_key}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            if len(data) > MAX_API_REQUEST_BYTES:
                raise ApiError("Conversation context exceeded the request safety limit. Use /clear and try again.")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with GOOGLE_API_OPENER.open(request, timeout=90) as response:
                raw = self._read_limited(response, MAX_API_RESPONSE_BYTES, "Google response")
                parsed = json.loads(raw.decode("utf-8", errors="replace"))
                if not isinstance(parsed, dict):
                    raise ApiError("Google returned an unexpected JSON response.")
                return parsed
        except urllib.error.HTTPError as error:
            raise ApiError("HTTP {}: {}".format(error.code, self._http_error_message(error))) from error
        except urllib.error.URLError as error:
            raise ApiError("Network error: {}".format(error.reason)) from error
        except TimeoutError as error:
            raise ApiError("Request timed out. Check the network connection and try again.") from error
        except OSError as error:
            raise ApiError("Network error: {}".format(error)) from error
        except json.JSONDecodeError as error:
            raise ApiError("Google returned an invalid JSON response.") from error

    @staticmethod
    def _read_limited(response: object, limit: int, label: str) -> bytes:
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ApiError("{} exceeded the {} byte safety limit.".format(label, limit))
        return raw

    def _http_error_message(self, error: urllib.error.HTTPError) -> str:
        try:
            raw_bytes = error.read(MAX_API_ERROR_BYTES + 1)
        except (AttributeError, OSError):
            raw_bytes = b""
        truncated = len(raw_bytes) > MAX_API_ERROR_BYTES
        raw = raw_bytes[:MAX_API_ERROR_BYTES].decode("utf-8", errors="replace").strip()
        message = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                details = parsed.get("error")
                if isinstance(details, dict) and isinstance(details.get("message"), str):
                    message = details["message"]
        except json.JSONDecodeError:
            pass
        if not message:
            message = str(error.reason or "request failed")
        if self.api_key:
            message = message.replace(self.api_key, "[redacted]")
        if truncated:
            message += " [truncated]"
        return message

    def list_models(self) -> List[str]:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        response = self._request(url)
        models = response.get("models", [])
        if not isinstance(models, list):
            raise ApiError("Google returned an unexpected model list.")
        return [
            model.get("name", "").removeprefix("models/")
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        ]

    def generate(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        if model.protocol == "palm":
            return self._generate_palm(model, history, system_instruction)
        return self._generate_gemini(model, history, system_instruction)

    def generate_stream(
        self,
        model: ModelSpec,
        history: Sequence[Dict[str, str]],
        system_instruction: str,
        on_text: Callable[[str], None],
    ) -> str:
        if model.protocol == "palm":
            # PaLM's retired chat API did not use the Gemini streaming endpoint.
            answer = self._generate_palm(model, history, system_instruction)
            on_text(answer)
            return answer
        return self._stream_gemini(model, history, system_instruction, on_text)

    @staticmethod
    def _gemini_body(history: Sequence[Dict[str, str]], system_instruction: str) -> dict:
        contents = [
            {"role": item["role"], "parts": [{"text": item["content"]}]}
            for item in history
            if item["role"] in ("user", "model") and item["content"]
        ]
        body = {"contents": contents, "generationConfig": {"temperature": 0.7}}
        if system_instruction.strip():
            body["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}
        return body

    def _generate_gemini(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent".format(
            urllib.parse.quote(model.identifier, safe="")
        )
        response = self._request(url, self._gemini_body(history, system_instruction))
        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            feedback = response.get("promptFeedback", {})
            detail = feedback.get("blockReason", "no candidate returned") if isinstance(feedback, dict) else "no candidate returned"
            raise ApiError("The request produced no response: {}.".format(detail))
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
        if len(text) > MAX_STREAM_TEXT_CHARS:
            raise ApiError("The response exceeded the safety limit.")
        return text

    def _stream_gemini(
        self,
        model: ModelSpec,
        history: Sequence[Dict[str, str]],
        system_instruction: str,
        on_text: Callable[[str], None],
    ) -> str:
        if not self.api_key:
            raise ApiError("No API key. Set GEMINI_API_KEY or press F4 to enter one.")
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse".format(
            urllib.parse.quote(model.identifier, safe="")
        )
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-goog-api-key": self.api_key,
        }
        request_data = json.dumps(self._gemini_body(history, system_instruction)).encode("utf-8")
        if len(request_data) > MAX_API_REQUEST_BYTES:
            raise ApiError("Conversation context exceeded the request safety limit. Use /clear and try again.")
        request = urllib.request.Request(url, data=request_data, headers=headers, method="POST")
        received: List[str] = []
        received_chars = 0
        try:
            with GOOGLE_API_OPENER.open(request, timeout=90) as response:
                while True:
                    raw_line = response.readline(MAX_STREAM_EVENT_BYTES + 1)
                    if not raw_line:
                        break
                    if len(raw_line) > MAX_STREAM_EVENT_BYTES:
                        raise ApiError("A streaming event exceeded the safety limit.")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError as error:
                        raise ApiError("Google returned an invalid streaming event.") from error
                    if not isinstance(event, dict):
                        raise ApiError("Google returned an unexpected streaming event.")
                    candidates = event.get("candidates", [])
                    if not isinstance(candidates, list) or not candidates:
                        continue
                    first = candidates[0] if isinstance(candidates[0], dict) else {}
                    content = first.get("content", {})
                    parts = content.get("parts", []) if isinstance(content, dict) else []
                    chunk = "".join(
                        part["text"]
                        for part in parts
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
                    if chunk:
                        received_chars += len(chunk)
                        if received_chars > MAX_STREAM_TEXT_CHARS:
                            raise ApiError("The streamed response exceeded the safety limit.")
                        received.append(chunk)
                        on_text(chunk)
        except urllib.error.HTTPError as error:
            raise ApiError("HTTP {}: {}".format(error.code, self._http_error_message(error))) from error
        except urllib.error.URLError as error:
            raise ApiError("Network error: {}".format(error.reason)) from error
        except TimeoutError as error:
            raise ApiError("Request timed out. Check the network connection and try again.") from error
        except OSError as error:
            raise ApiError("Network error: {}".format(error)) from error
        text = "".join(received).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
        return text

    def _generate_palm(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        # PaLM Chat used a different, now-retired API shape. Keeping this request
        # lets accounts with an available legacy endpoint attempt the original model.
        url = "https://generativelanguage.googleapis.com/v1beta3/models/{}:generateMessage".format(
            urllib.parse.quote(model.identifier, safe="")
        )
        messages = [
            {"author": "0" if item["role"] == "user" else "1", "content": item["content"]}
            for item in history
        ]
        prompt = {"messages": messages}
        if system_instruction.strip():
            prompt["context"] = system_instruction.strip()
        try:
            response = self._request(url, {"prompt": prompt, "temperature": 0.7})
        except ApiError as error:
            if str(error).startswith("HTTP 501:"):
                raise ApiError(
                    "PaLM Chat-Bison 001 is retired and cannot be called by Google's public API. Press F2 and select a Gemini Flash model."
                ) from error
            raise
        candidates = response.get("candidates", [])
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(candidates[0], dict)
            or not isinstance(candidates[0].get("content"), str)
        ):
            raise ApiError("The PaLM endpoint returned no text.")
        text = candidates[0]["content"].strip()
        if not text:
            raise ApiError("The PaLM endpoint returned no text.")
        if len(text) > MAX_STREAM_TEXT_CHARS:
            raise ApiError("The response exceeded the safety limit.")
        return text


class SessionStore:
    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / STATE_FILE_NAME

    def load(self) -> dict:
        try:
            if self.path.is_symlink():
                return self.normalize({})
            flags = os.O_RDONLY
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(str(self.path), flags)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_SESSION_BYTES:
                    return self.normalize({})
                with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                    descriptor = -1
                    raw = handle.read(MAX_SESSION_BYTES + 1)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            if len(raw.encode("utf-8")) > MAX_SESSION_BYTES:
                return self.normalize({})
            data = json.loads(raw)
            return self.normalize(data)
        except (FileNotFoundError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return self.normalize({})

    @staticmethod
    def normalize(data: object) -> dict:
        if not isinstance(data, dict):
            data = {}
        model = data.get("model", DEFAULT_MODEL_ID)
        try:
            model = validate_model_id(model) if isinstance(model, str) else DEFAULT_MODEL_ID
        except ApiError:
            model = DEFAULT_MODEL_ID
        raw_system_instruction = data.get("system_instruction")
        system_instruction = (
            raw_system_instruction
            if isinstance(raw_system_instruction, str)
            else DEFAULT_SYSTEM_INSTRUCTION
        )
        history = []
        raw_history = data.get("history", [])
        if isinstance(raw_history, list):
            for item in raw_history:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role not in ("user", "model") or not isinstance(content, str) or not content:
                    continue
                normalized = {"role": role, "content": content}
                model_id = item.get("model_id")
                if role == "model" and isinstance(model_id, str) and model_id:
                    normalized["model_id"] = model_id
                history.append(normalized)
        settings = data.get("settings", {})
        streaming = settings.get("streaming", True) if isinstance(settings, dict) else True
        theme = settings.get("theme", DEFAULT_THEME_ID) if isinstance(settings, dict) else DEFAULT_THEME_ID
        if not isinstance(theme, str) or theme not in THEME_BY_ID:
            theme = DEFAULT_THEME_ID
        default_instruction_configured = isinstance(raw_system_instruction, str) and bool(raw_system_instruction)
        system_instruction_configured = (
            settings.get("system_instruction_configured", default_instruction_configured)
            if isinstance(settings, dict)
            else default_instruction_configured
        )
        if not isinstance(system_instruction_configured, bool):
            system_instruction_configured = bool(system_instruction)
        if not system_instruction_configured:
            system_instruction = DEFAULT_SYSTEM_INSTRUCTION
        return {
            "model": model,
            "system_instruction": system_instruction,
            "history": history,
            "settings": {
                "streaming": streaming if isinstance(streaming, bool) else True,
                "system_instruction_configured": system_instruction_configured,
                "theme": theme,
            },
        }

    def save(self, data: dict) -> None:
        temporary: Optional[Path] = None
        descriptor: Optional[int] = None
        try:
            serialized = json.dumps(data, indent=2, ensure_ascii=True).encode("utf-8")
            if len(serialized) > MAX_SESSION_BYTES:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".session-", suffix=".tmp", dir=str(self.path.parent), text=True
            )
            temporary = Path(temporary_name)
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.path))
            temporary = None
            os.chmod(self.path, 0o600)
        except OSError:
            pass  # A read-only directory should not stop a chat session.
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass


class Tui:
    def __init__(self, api_key: str, store: SessionStore) -> None:
        self.api_key = api_key.strip()
        self.store = store
        self.data = store.load()
        if not isinstance(self.data.get("model"), str) or not self.data["model"].strip():
            self.data["model"] = DEFAULT_MODEL_ID
        if not isinstance(self.data.get("settings"), dict):
            self.data["settings"] = {}
        self.data["settings"].setdefault("streaming", True)
        self.data["settings"].setdefault("system_instruction_configured", False)
        self.data["settings"].setdefault("theme", DEFAULT_THEME_ID)
        self.status = "Ready"
        self.available_models: Optional[set] = None
        self.running = True
        self.restart_requested = False
        self.colors: Dict[str, int] = {}
        self.scroll_offset = 0
        self.prompt_history = [
            item["content"] for item in self.data["history"] if item["role"] == "user"
        ]
        self.last_failed_prompt: Optional[str] = None

    @property
    def model(self) -> ModelSpec:
        identifier = self.data["model"]
        return MODEL_BY_ID.get(
            identifier,
            ModelSpec(identifier, identifier, "Custom Gemini Model", "gemini", "Custom model ID"),
        )

    @property
    def streaming_enabled(self) -> bool:
        return bool(self.data["settings"].get("streaming", True))

    @property
    def theme(self) -> ThemeSpec:
        return THEME_BY_ID.get(self.data["settings"].get("theme"), THEME_BY_ID[DEFAULT_THEME_ID])

    @staticmethod
    def wrap_content(content: str, width: int) -> List[str]:
        wrapped: List[str] = []
        for source_line in content.splitlines() or [""]:
            wrapped.extend(
                textwrap.wrap(source_line, width=max(width, 1), replace_whitespace=False) or [""]
            )
        return wrapped

    @staticmethod
    def history_window(
        lines: Sequence[Tuple[str, int]], height: int, offset: int
    ) -> Tuple[List[Tuple[str, int]], int, int]:
        window_height = max(height, 1)
        maximum_offset = max(len(lines) - window_height, 0)
        normalized_offset = min(max(offset, 0), maximum_offset)
        end = len(lines) - normalized_offset
        start = max(end - window_height, 0)
        return list(lines[start:end]), normalized_offset, maximum_offset

    def render_history(self, width: int, compact: bool = False) -> List[Tuple[str, int]]:
        lines: List[Tuple[str, int]] = []
        content_width = max(width - (5 if compact else 8), 8)
        for item in self.data["history"]:
            is_user = item["role"] == "user"
            label = "YOU" if is_user else item.get("model_id", "UNKNOWN MODEL").upper()
            role_style = self.style("user" if is_user else "assistant", curses.A_BOLD)
            wrapped = self.wrap_content(item["content"], content_width)
            if compact:
                compact_label = "You" if is_user else item.get("model_id", "Unknown model")
                lines.append((compact_label + ": " + wrapped[0], role_style))
                lines.extend(("   " + line, self.style("text")) for line in wrapped[1:])
            elif is_user:
                lines.append(("  YOU  " + wrapped[0], role_style))
                lines.extend(("       " + line, self.style("text")) for line in wrapped[1:])
            else:
                lines.append(("  " + label, role_style))
                lines.extend(("       " + line, self.style("text")) for line in wrapped)
            lines.append(("", 0))
        return lines

    def transcript_lines(self) -> List[str]:
        lines: List[str] = []
        for item in self.data["history"]:
            if item["role"] == "user":
                label = "YOU"
            else:
                label = item.get("model_id", "UNKNOWN MODEL").upper()
            lines.append(label)
            lines.extend(item["content"].splitlines() or [""])
            lines.append("")
        return lines or ["No messages in this session."]

    def remember_prompt(self, text: str) -> None:
        if text and (not self.prompt_history or self.prompt_history[-1] != text):
            self.prompt_history.append(text)

    def run(self, screen: "curses._CursesWindow") -> None:
        self.initialize_theme(screen)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        screen.keypad(True)
        screen.timeout(-1)
        while self.running:
            self.draw_chat(screen)
            prompt = self.read_line(screen, "You > ")
            if prompt is None:
                continue
            self.handle_input(screen, prompt)

    def draw_chat(self, screen: "curses._CursesWindow") -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 10 or width < 42:
            self.draw_compact_chat(screen)
            return

        self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("base"))
        self.add(screen, 0, 1, "[ GEMINI LEGACY ]", self.style("header", curses.A_BOLD))
        self.add(screen, 0, 20, "CHAT", self.style("header_muted"))
        if self.model.protocol == "palm":
            state, state_style = " RETIRED API ", self.style("error", curses.A_BOLD)
        elif self.api_key:
            state, state_style = " READY ", self.style("ok", curses.A_BOLD)
        else:
            state, state_style = " KEY REQUIRED ", self.style("warning", curses.A_BOLD)
        self.add(screen, 0, max(width - len(state) - 1, 1), state, state_style)

        metadata = "  {}  |  {}  |  local session  |  {} turn{}".format(
            self.model.identifier,
            self.model.protocol.upper(),
            len(self.data["history"]) // 2,
            "s" if len(self.data["history"]) // 2 != 1 else "",
        )
        self.add(screen, 1, 0, metadata[: max(width - 1, 0)], self.style("muted"))
        self.add(screen, 2, 0, "-" * max(width - 1, 0), self.style("border"))

        history_height = max(height - 9, 1)
        lines = self.render_history(width)

        if not lines:
            empty_row = max(5, (height - 8) // 2)
            self.add(screen, empty_row, 4, "Start a conversation", self.style("text", curses.A_BOLD))
            self.add(screen, empty_row + 2, 4, "Tab opens the menu. / opens commands.", self.style("muted"))
        visible, self.scroll_offset, _ = self.history_window(lines, history_height, self.scroll_offset)
        for row, (line, style) in enumerate(visible, start=3):
            self.add(screen, row, 0, line[: max(width - 1, 0)], style)
        self.add(screen, height - 6, 0, "-" * max(width - 1, 0), self.style("border"))
        for index, line in enumerate(self.status_lines(width), start=height - 5):
            self.add(screen, index, 1, line, self.status_style())
        self.add(screen, height - 3, 0, "-" * max(width - 1, 0), self.style("border"))
        self.add(screen, height - 2, 1, ">", self.style("accent", curses.A_BOLD))
        if self.scroll_offset:
            footer = "PgUp/Ctrl+U Scroll   {} lines above latest".format(self.scroll_offset)
        else:
            footer = "Tab Menu   / Commands   PgUp/Ctrl+U Scroll   Up Recall"
        self.add(screen, height - 1, 1, footer[: width - 2], self.style("footer"))
        screen.refresh()

    def draw_compact_chat(self, screen: "curses._CursesWindow") -> None:
        height, width = screen.getmaxyx()
        title = " {} | {} ".format(APP_NAME, self.model.label)
        self.add(screen, 0, 0, title[: max(width - 1, 0)], curses.A_REVERSE)
        history_height = max(height - 4, 1)
        lines = self.render_history(width, compact=True)
        visible, self.scroll_offset, _ = self.history_window(lines, history_height, self.scroll_offset)
        if self.scroll_offset:
            title = "{} | +{} lines".format(title.rstrip(), self.scroll_offset)
            self.add(screen, 0, 0, title[: max(width - 1, 0)], curses.A_REVERSE)
        for row, (line, style) in enumerate(visible, start=1):
            self.add(screen, row, 0, line[: max(width - 1, 0)], style)
        self.add(screen, height - 3, 0, "-" * max(width - 1, 0))
        self.add(screen, height - 2, 0, self.status[: max(width - 1, 0)], curses.A_DIM)
        self.add(screen, height - 1, 0, "You > ")
        screen.refresh()

    def read_line(self, screen: "curses._CursesWindow", label: str) -> Optional[str]:
        buffer: List[str] = []
        cursor = 0
        history_index = len(self.prompt_history)
        draft: List[str] = []
        while True:
            height, width = screen.getmaxyx()
            prompt_row = height - 2 if height >= 10 and width >= 42 else height - 1
            prompt_label = "> " if prompt_row != height - 1 else label
            available_width = max(width - len(prompt_label) - 2, 1)
            visible = "".join(buffer)
            start = max(0, cursor - available_width + 1)
            rendered = visible[start : start + available_width]
            self.add(screen, prompt_row, 0, " " * max(width - 1, 0))
            self.add(screen, prompt_row, 1 if prompt_row != height - 1 else 0, prompt_label + rendered, self.style("input"))
            try:
                screen.move(
                    prompt_row,
                    min((1 if prompt_row != height - 1 else 0) + len(prompt_label) + cursor - start, width - 1),
                )
            except curses.error:
                pass
            screen.refresh()
            key = screen.get_wch()
            if key == curses.KEY_RESIZE:
                self.draw_chat(screen)
                continue
            if key in (curses.KEY_PPAGE, "\x15"):
                self.scroll_offset += max(height - 10, 1)
                self.draw_chat(screen)
                continue
            if key in (curses.KEY_NPAGE, "\x04"):
                self.scroll_offset = max(0, self.scroll_offset - max(height - 10, 1))
                self.draw_chat(screen)
                continue
            if key in ("\n", "\r"):
                return "".join(buffer).strip()
            if key == "\x03":
                self.running = False
                return None
            if key == curses.KEY_F1:
                self.show_help(screen)
                return None
            if key in ("\t", "\x0b"):
                self.open_action_menu(screen)
                return None
            if key == "\x0e":
                self.clear_context()
                return None
            if key in (curses.KEY_F2, "\x0c"):
                self.select_model(screen)
                return None
            if key == curses.KEY_F3:
                self.edit_system_instruction(screen)
                return None
            if key == curses.KEY_F4:
                self.configure_api_key(screen)
                return None
            if key == curses.KEY_F5:
                self.check_availability(screen)
                return None
            if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                if cursor:
                    del buffer[cursor - 1]
                    cursor -= 1
                continue
            if key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor = min(len(buffer), cursor + 1)
                continue
            if key == curses.KEY_UP and self.prompt_history:
                if history_index == len(self.prompt_history):
                    draft = list(buffer)
                history_index = max(0, history_index - 1)
                buffer = list(self.prompt_history[history_index])
                cursor = len(buffer)
                continue
            if key == curses.KEY_DOWN and history_index < len(self.prompt_history):
                history_index += 1
                buffer = list(draft if history_index == len(self.prompt_history) else self.prompt_history[history_index])
                cursor = len(buffer)
                continue
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(buffer)
                continue
            if isinstance(key, str) and key.isprintable():
                if len(buffer) >= MAX_INPUT_CHARS:
                    self.status = "Prompt limit reached ({} characters).".format(MAX_INPUT_CHARS)
                    try:
                        curses.beep()
                    except curses.error:
                        pass
                    continue
                buffer.insert(cursor, key)
                cursor += 1
                if "".join(buffer) == "/":
                    command = self.open_command_palette(screen)
                    if command:
                        self.handle_command(screen, command)
                        return None
                    buffer = []
                    cursor = 0

    def handle_input(self, screen: "curses._CursesWindow", text: str) -> None:
        if not text:
            return
        if text.startswith("/"):
            self.handle_command(screen, text)
            return
        self.remember_prompt(text)
        self.scroll_offset = 0
        selected_model = self.model
        self.data["history"].append({"role": "user", "content": text})
        self.store.save(self.data)
        self.status = "Streaming {}...".format(selected_model.identifier) if self.streaming_enabled else "Waiting for {}...".format(selected_model.identifier)
        try:
            if selected_model.protocol == "gemini" and self.streaming_enabled:
                reply = {"role": "model", "content": "", "model_id": selected_model.identifier}
                self.data["history"].append(reply)
                self.draw_chat(screen)

                def append_chunk(chunk: str) -> None:
                    fragments = self.display_fragments(chunk)
                    for fragment in fragments:
                        reply["content"] += fragment
                        self.status = "Streaming {}... {} chars".format(selected_model.identifier, len(reply["content"]))
                        self.draw_chat(screen)
                        if len(fragments) > 1:
                            try:
                                curses.napms(STREAM_RENDER_DELAY_MS)
                            except curses.error:
                                pass

                answer = GoogleApiClient(self.api_key).generate_stream(
                    selected_model, self.data["history"][:-1], self.data.get("system_instruction", ""), append_chunk
                )
                reply["content"] = answer
            else:
                self.draw_chat(screen)
                answer = GoogleApiClient(self.api_key).generate(
                    selected_model, self.data["history"], self.data.get("system_instruction", "")
                )
                self.data["history"].append({"role": "model", "content": answer, "model_id": selected_model.identifier})
            self.status = "Response complete."
            self.last_failed_prompt = None
        except ApiError as error:
            if self.data["history"] and self.data["history"][-1].get("role") == "model":
                self.data["history"].pop()
            if self.data["history"] and self.data["history"][-1].get("role") == "user":
                self.data["history"].pop()
            self.last_failed_prompt = text
            self.status = "{} Use /retry or Up to recall the prompt.".format(
                self.friendly_error(error, selected_model)
            )
        self.store.save(self.data)

    def friendly_error(self, error: ApiError, model: ModelSpec) -> str:
        raw = str(error)
        if raw.startswith("HTTP 429:") and "quota" in raw.lower():
            return "Quota unavailable for {}. Check plan/billing or choose another model.".format(model.identifier)
        if raw.startswith("HTTP 404:") and ("not found" in raw.lower() or "not supported" in raw.lower()):
            return "{} is unavailable for this API key or endpoint. Choose another model with Ctrl+L.".format(model.identifier)
        return raw

    def clear_context(self) -> None:
        self.data["history"] = []
        self.scroll_offset = 0
        self.store.save(self.data)
        self.status = "Context cleared."

    def retry_last(self, screen: "curses._CursesWindow") -> None:
        if not self.last_failed_prompt:
            self.status = "There is no failed prompt to retry."
            return
        prompt = self.last_failed_prompt
        self.status = "Retrying the last failed prompt..."
        self.handle_input(screen, prompt)

    def handle_command(self, screen: "curses._CursesWindow", text: str) -> None:
        command, _, argument = text.partition(" ")
        command = command.lower()
        if command in ("/quit", "/exit"):
            self.running = False
        elif command in ("/restart", "/reopen"):
            self.restart_requested = True
            self.running = False
        elif command in ("/new", "/clear"):
            self.clear_context()
        elif command == "/model":
            self.select_model(screen)
        elif command == "/models":
            self.show_models(screen)
        elif command == "/retry":
            self.retry_last(screen)
        elif command == "/settings":
            self.open_settings(screen)
        elif command in ("/transcript", "/history"):
            self.show_transcript(screen)
        elif command == "/system":
            if argument.strip():
                self.data["system_instruction"] = argument.strip()[:MAX_DIALOG_CHARS]
                self.data["settings"]["system_instruction_configured"] = True
                self.store.save(self.data)
                self.status = "System instruction updated."
            else:
                self.edit_system_instruction(screen)
        elif command == "/key":
            self.configure_api_key(screen)
        elif command == "/check":
            self.check_availability(screen)
        elif command == "/help":
            self.show_help(screen)
        else:
            self.status = "Unknown command. Try /help."

    def open_action_menu(self, screen: "curses._CursesWindow") -> None:
        actions = (
            ("Choose model", "Browse and filter the model catalog", "model"),
            ("Settings", "Themes, streaming, system instruction, API key, and availability", "settings"),
            ("View transcript", "Read the complete rewrapped conversation", "transcript"),
            ("Clear context", "Remove the shared conversation history", "clear"),
            ("Retry failed prompt", "Run the most recent failed prompt again", "retry"),
            ("Restart interface", "Close and reopen the terminal interface", "restart"),
            ("Help", "Show shortcuts and slash commands", "help"),
            ("Exit", "Close Gemini Legacy", "exit"),
        )
        selected = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("base"))
            self.add(screen, 0, 2, "ACTIONS", self.style("header", curses.A_BOLD))
            self.add(screen, 2, 2, "Use arrows and Enter. Esc returns to chat.", self.style("muted"))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            for index, (label, description, _) in enumerate(actions):
                row = 5 + index * 2
                if row >= height - 2:
                    break
                marker = ">" if index == selected else " "
                style = self.style("selected") if index == selected else self.style("text")
                self.add(screen, row, 2, "{}  {: <23}".format(marker, label)[: width - 4], style)
                if row + 1 < height - 1:
                    self.add(screen, row + 1, 5, description[: width - 7], self.style("muted"))
            self.add(screen, height - 1, 2, "Up/Down select   Enter open   Esc close", self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if key in ("\x1b", "\t", "q", "Q"):
                break
            if key in (curses.KEY_UP, "k"):
                selected = max(0, selected - 1)
                continue
            if key in (curses.KEY_DOWN, "j"):
                selected = min(len(actions) - 1, selected + 1)
                continue
            if key not in ("\n", "\r"):
                continue
            action = actions[selected][2]
            if action == "clear":
                self.clear_context()
            elif action == "model":
                self.select_model(screen)
            elif action == "settings":
                self.open_settings(screen)
            elif action == "transcript":
                self.show_transcript(screen)
            elif action == "retry":
                self.retry_last(screen)
            elif action == "restart":
                self.restart_requested = True
                self.running = False
            elif action == "help":
                self.show_help(screen)
            elif action == "exit":
                self.running = False
            break
        self.show_cursor()

    def open_settings(self, screen: "curses._CursesWindow") -> None:
        selected = 0
        settings = (
            ("Streaming", "Render Gemini output as it arrives", "stream"),
            ("Theme", "Choose the terminal color palette", "theme"),
            ("Custom model ID", "Use a Gemini model ID not in the catalog", "custom_model"),
            ("System instruction", "Set or clear the instruction for this chat", "system"),
            ("API key", "Enter an API key for this run only", "key"),
            ("Model availability", "Check models visible to this API key", "check"),
            ("Back", "Return to chat", "back"),
        )
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("base"))
            self.add(screen, 0, 2, "SETTINGS", self.style("header", curses.A_BOLD))
            if os.name == "nt":
                font_hint = "Terminal font: Ctrl+Plus / Ctrl+Minus (terminal-managed)"
            elif sys.platform == "darwin":
                font_hint = "Terminal font: Command+Plus / Command-Minus (Terminal resizes window)"
            else:
                font_hint = "Terminal font size is managed by your terminal application"
            self.add(screen, 2, 2, font_hint, self.style("muted"))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            for index, (label, description, _) in enumerate(settings):
                row = 5 + index * 2
                if row >= height - 2:
                    break
                marker = ">" if index == selected else " "
                style = self.style("selected") if index == selected else self.style("text")
                if label == "Streaming":
                    value = "[ON]" if self.streaming_enabled else "[OFF]"
                elif label == "Theme":
                    value = self.theme.label
                elif label == "Custom model ID" and self.data["model"] not in MODEL_BY_ID:
                    value = self.data["model"]
                elif label == "System instruction":
                    value = (
                        "Custom"
                        if self.data["settings"].get("system_instruction_configured")
                        else "Terminal default"
                    )
                else:
                    value = ""
                self.add(screen, row, 2, "{}  {: <22} {}".format(marker, label, value)[: width - 4], style)
                if row + 1 < height - 1:
                    self.add(screen, row + 1, 5, description[: width - 7], self.style("muted"))
            self.add(screen, height - 1, 2, "Up/Down select   Enter open   Space toggle   Esc close", self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if key in ("\x1b", "q", "Q"):
                break
            if key in (curses.KEY_UP, "k"):
                selected = max(0, selected - 1)
                continue
            if key in (curses.KEY_DOWN, "j"):
                selected = min(len(settings) - 1, selected + 1)
                continue
            action = settings[selected][2]
            if key == " " and action != "stream":
                continue
            if key not in (" ", "\n", "\r"):
                continue
            if action == "stream":
                self.data["settings"]["streaming"] = not self.streaming_enabled
                self.store.save(self.data)
                self.status = "Streaming {}.".format("enabled" if self.streaming_enabled else "disabled")
            elif action == "theme":
                self.select_theme(screen)
            elif action == "custom_model":
                value = self.prompt_dialog(
                    screen,
                    "Custom Gemini model ID",
                    self.data["model"],
                    max_length=MAX_MODEL_ID_CHARS,
                )
                if value:
                    try:
                        self.data["model"] = validate_model_id(value)
                        self.store.save(self.data)
                        self.status = "Selected custom model {}. Shared context is retained.".format(value)
                    except ApiError as error:
                        self.status = str(error)
            elif action == "system":
                self.edit_system_instruction(screen)
            elif action == "key":
                self.configure_api_key(screen)
            elif action == "check":
                self.check_availability(screen)
            elif action == "back":
                break
        self.show_cursor()

    def select_theme(self, screen: "curses._CursesWindow") -> None:
        selected_id = self.theme.identifier
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("base"))
            self.add(screen, 0, 2, "THEMES", self.style("header", curses.A_BOLD))
            self.add(screen, 2, 2, "Choose a palette. Enter applies it. Esc returns to Settings.", self.style("muted"))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            selected = next(
                (index for index, theme in enumerate(THEMES) if theme.identifier == selected_id),
                0,
            )
            for index, theme in enumerate(THEMES):
                row = 5 + index * 3
                if row >= height - 2:
                    break
                is_selected = theme.identifier == selected_id
                marker = ">" if is_selected else " "
                style = self.style("selected") if is_selected else self.style("text")
                self.add(screen, row, 2, "{}  {}".format(marker, theme.label)[: width - 4], style)
                if row + 1 < height - 1:
                    self.add(screen, row + 1, 5, theme.description[: width - 7], self.style("muted"))
                if row + 2 < height - 1:
                    preview = "YOU  sample prompt     GEMINI  sample response"
                    self.add(screen, row + 2, 5, preview[: width - 7], self.style("accent"))
            self.add(screen, height - 1, 2, "Up/Down select   Enter apply   Esc close", self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if key in ("\x1b", "q", "Q"):
                return
            if key in (curses.KEY_UP, "k"):
                selected_id = THEMES[max(0, selected - 1)].identifier
                continue
            if key in (curses.KEY_DOWN, "j"):
                selected_id = THEMES[min(len(THEMES) - 1, selected + 1)].identifier
                continue
            if key in ("\n", "\r"):
                selected_theme = THEME_BY_ID[selected_id]
                self.data["settings"]["theme"] = selected_theme.identifier
                self.store.save(self.data)
                self.initialize_theme(screen)
                self.status = "Theme set to {}.".format(selected_theme.label)
                return

    def open_command_palette(self, screen: "curses._CursesWindow") -> Optional[str]:
        query = ""
        selected_command = "/clear"
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            matches = [
                command
                for command in SLASH_COMMANDS
                if query.lower() in "{} {}".format(command[0], command[1]).lower()
            ]
            if matches and selected_command not in {command[0] for command in matches}:
                selected_command = matches[0][0]
            selected = next((index for index, command in enumerate(matches) if command[0] == selected_command), 0)
            screen.erase()
            height, width = screen.getmaxyx()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("base"))
            self.add(screen, 0, 2, "COMMANDS", self.style("header", curses.A_BOLD))
            self.add(screen, 2, 2, "/ " + query, self.style("input", curses.A_BOLD))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            visible_count = max((height - 6) // 2, 1)
            offset = min(
                max(selected - visible_count + 1, 0),
                max(len(matches) - visible_count, 0),
            )
            for displayed, index in enumerate(
                range(offset, min(offset + visible_count, len(matches)))
            ):
                command, description = matches[index]
                row = 5 + displayed * 2
                is_selected = command == selected_command
                self.add(screen, row, 2, ("> " if is_selected else "  ") + command, self.style("selected") if is_selected else self.style("accent", curses.A_BOLD))
                if row + 1 < height - 1:
                    self.add(screen, row + 1, 6, description[: width - 8], self.style("muted"))
            if not matches:
                self.add(screen, 5, 2, "No matching command.", self.style("warning"))
            position = "{} / {}".format(selected + 1, len(matches)) if matches else "0 / 0"
            footer = "Type filter   Up/Down select   Enter run   Esc close   {}".format(position)
            self.add(screen, height - 1, 2, footer[: width - 4], self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if key == "\x1b":
                self.show_cursor()
                return None
            if key == curses.KEY_UP and matches:
                selected_command = matches[max(0, selected - 1)][0]
                continue
            if key == curses.KEY_DOWN and matches:
                selected_command = matches[min(len(matches) - 1, selected + 1)][0]
                continue
            if key in ("\n", "\r") and matches:
                self.show_cursor()
                return matches[selected][0]
            if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                query = query[:-1]
            elif key == "\x15":
                query = ""
            elif isinstance(key, str) and key.isprintable() and len(query) < 128:
                query += key

    def select_model(self, screen: "curses._CursesWindow") -> None:
        selected_id = self.model.identifier
        query = ""
        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            matches = [
                model
                for model in MODEL_CATALOG
                if query.lower() in "{} {} {}".format(model.identifier, model.label, model.group).lower()
            ]
            if matches and selected_id not in {model.identifier for model in matches}:
                selected_id = matches[0].identifier
            selected = next((index for index, model in enumerate(matches) if model.identifier == selected_id), 0)
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("header"))
            self.add(screen, 0, 1, "MODEL PICKER", self.style("header", curses.A_BOLD))
            self.add(screen, 0, max(width - 18, 1), "{} MODELS".format(len(matches)), self.style("header_muted"))
            self.add(screen, 2, 2, "Filter > " + query, self.style("input"))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            rows = max(height - 10, 1)
            offset = min(max(selected - rows + 1, 0), max(len(matches) - rows, 0))
            last_group = None
            for displayed, index in enumerate(range(offset, min(offset + rows, len(matches))), start=4):
                model = matches[index]
                prefix = "> " if model.identifier == selected_id else "  "
                availability = self.availability_marker(model)
                group = model.group.upper() if model.group != last_group else " " * len(model.group)
                line = "{}{: <20}  {: <34} {}".format(prefix, group, model.label, availability)
                self.add(screen, displayed, 0, line[: width - 1], self.style("selected") if model.identifier == selected_id else self.style("text"))
                last_group = model.group
            if matches:
                focused = matches[selected]
                self.add(screen, height - 5, 0, "-" * max(width - 1, 0), self.style("border"))
                self.add(screen, height - 4, 2, focused.identifier[: width - 4], self.style("accent", curses.A_BOLD))
                self.add(screen, height - 3, 2, focused.availability_note[: width - 4], self.style("muted"))
            else:
                self.add(screen, 5, 2, "No matching models.", self.style("warning"))
            self.add(screen, height - 1, 1, "Type to filter   Up/Down select   Enter choose   F5 check   Esc close"[: width - 2], self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if key in ("\n", "\r"):
                if not matches:
                    continue
                self.data["model"] = matches[selected].identifier
                self.store.save(self.data)
                if self.model.protocol == "palm":
                    self.status = "PaLM Chat-Bison 001 is retired. Select a Gemini Flash model (Ctrl+L)."
                else:
                    self.status = "Selected {}. Shared context is retained; Ctrl+N clears it.".format(self.model.label)
                return
            if key in ("\x1b", "q", "Q"):
                return
            if key in (curses.KEY_UP, "k"):
                if matches:
                    selected_id = matches[max(0, selected - 1)].identifier
            elif key in (curses.KEY_DOWN, "j"):
                if matches:
                    selected_id = matches[min(len(matches) - 1, selected + 1)].identifier
            elif key == curses.KEY_F5:
                self.check_availability(screen)
            elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                query = query[:-1]
            elif key == "\x15":
                query = ""
            elif isinstance(key, str) and key.isprintable() and key not in ("j", "k") and len(query) < 128:
                query += key

    def availability_marker(self, model: ModelSpec) -> str:
        if self.available_models is None:
            return model.availability_note
        return "available" if model.identifier in self.available_models else "not listed for this key"

    def check_availability(self, screen: "curses._CursesWindow") -> None:
        self.status = "Checking the models visible to this API key..."
        self.draw_chat(screen)
        try:
            self.available_models = set(GoogleApiClient(self.api_key).list_models())
            count = sum(1 for model in MODEL_CATALOG if model.identifier in self.available_models)
            self.status = "{} catalog model(s) are listed for this key. F2 shows details.".format(count)
        except ApiError as error:
            self.status = str(error)

    def edit_system_instruction(self, screen: "curses._CursesWindow") -> None:
        value = self.prompt_dialog(
            screen,
            "System instruction",
            self.data.get("system_instruction", ""),
            max_length=MAX_DIALOG_CHARS,
        )
        if value is not None:
            self.data["system_instruction"] = value
            self.data["settings"]["system_instruction_configured"] = True
            self.store.save(self.data)
            self.status = "System instruction {}.".format("updated" if value else "cleared")

    def configure_api_key(self, screen: "curses._CursesWindow") -> None:
        value = self.prompt_dialog(
            screen,
            "API key (kept only for this run)",
            "",
            secret=True,
            max_length=MAX_API_KEY_CHARS,
        )
        if value:
            try:
                self.api_key = validate_api_key(value)
                self.available_models = None
                self.status = "API key updated for this session."
            except ApiError as error:
                self.status = str(error)
        elif value == "":
            self.status = "API key was unchanged."

    def prompt_dialog(
        self,
        screen: "curses._CursesWindow",
        title: str,
        initial: str,
        secret: bool = False,
        max_length: int = MAX_DIALOG_CHARS,
    ) -> Optional[str]:
        height, width = screen.getmaxyx()
        window_height = 7
        if height < window_height or width < 20:
            self.status = "Terminal is too small for this dialog. Resize it and try again."
            return None
        window_width = min(max(50, len(title) + 8), max(width - 2, 20))
        window = curses.newwin(window_height, window_width, max((height - window_height) // 2, 0), max((width - window_width) // 2, 0))
        window.keypad(True)
        buffer = list(initial[:max_length])
        cursor = len(buffer)
        while True:
            window.erase()
            window.box()
            self.add(window, 1, 2, title[: window_width - 4], curses.A_BOLD)
            shown = "*" * len(buffer) if secret else "".join(buffer)
            start = max(0, cursor - (window_width - 6))
            self.add(window, 3, 2, shown[start : start + window_width - 5])
            self.add(window, 5, 2, "Enter: confirm  Esc: cancel"[: window_width - 4], curses.A_DIM)
            try:
                window.move(3, min(2 + cursor - start, window_width - 3))
            except curses.error:
                pass
            window.refresh()
            key = window.get_wch()
            if key in ("\n", "\r"):
                return "".join(buffer).strip()
            if key == "\x1b":
                return None
            if key in (curses.KEY_BACKSPACE, "\b", "\x7f") and cursor:
                del buffer[cursor - 1]
                cursor -= 1
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(buffer), cursor + 1)
            elif isinstance(key, str) and key.isprintable():
                if len(buffer) >= max_length:
                    try:
                        curses.beep()
                    except curses.error:
                        pass
                    continue
                buffer.insert(cursor, key)
                cursor += 1

    def show_help(self, screen: "curses._CursesWindow") -> None:
        self.show_message(
            screen,
            "Help",
            [
                "Tab opens the main menu. Type / for the searchable command palette.",
                "Page Up/Page Down or Ctrl+U/Ctrl+D scroll the chat. Up and Down recall submitted prompts.",
                "/transcript opens the complete conversation reader with Up/Down scrolling.",
                "Settings contains streaming, custom model ID, system instruction, API key, and model availability.",
                "/clear erases context; /retry repeats a failed prompt; /restart reopens the UI.",
                "API keys are never written to disk. Transcripts live in the selected state folder.",
            ],
        )

    def show_models(self, screen: "curses._CursesWindow") -> None:
        lines = ["{} [{}] - {}".format(model.identifier, model.group, self.availability_marker(model)) for model in MODEL_CATALOG]
        self.show_message(screen, "Configured model catalog", lines)

    def show_transcript(self, screen: "curses._CursesWindow") -> None:
        self.show_message(screen, "Complete transcript", self.transcript_lines())

    def show_message(self, screen: "curses._CursesWindow", title: str, lines: Sequence[str]) -> None:
        offset = 0
        while True:
            height, width = screen.getmaxyx()
            wrapped_lines: List[str] = []
            for line in lines:
                wrapped_lines.extend(self.wrap_content(line, max(width - 2, 8)))
            page_height = max(height - 3, 1)
            maximum_offset = max(len(wrapped_lines) - page_height, 0)
            offset = min(max(offset, 0), maximum_offset)
            screen.erase()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("header"))
            self.add(screen, 0, 1, title[: width - 2], self.style("header", curses.A_BOLD))
            for row, line in enumerate(wrapped_lines[offset : offset + page_height], start=2):
                self.add(screen, row, 1, line[: width - 2], self.style("text"))
            if wrapped_lines:
                first = offset + 1
                last = min(offset + page_height, len(wrapped_lines))
                position = "{}-{}/{}".format(first, last, len(wrapped_lines))
            else:
                position = "0/0"
            footer = "Up/Down/PgUp/PgDn/Ctrl+U/Ctrl+D scroll   Enter/Esc close   {}".format(position)
            self.add(screen, height - 1, 1, footer[: width - 2], self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if key in ("\x1b", "\n", "\r", "q", "Q"):
                return
            if key == curses.KEY_RESIZE:
                continue
            if key == curses.KEY_UP:
                offset -= 1
            elif key == curses.KEY_DOWN:
                offset += 1
            elif key in (curses.KEY_PPAGE, "\x15"):
                offset -= page_height
            elif key in (curses.KEY_NPAGE, "\x04"):
                offset += page_height
            elif key == curses.KEY_HOME:
                offset = 0
            elif key == curses.KEY_END:
                offset = maximum_offset

    @staticmethod
    def add(screen: "curses._CursesWindow", row: int, column: int, value: str, style: int = 0) -> None:
        try:
            screen.addstr(row, column, Tui.sanitize_display(value), style)
        except curses.error:
            pass

    @staticmethod
    def sanitize_display(value: str) -> str:
        """Remove control and formatting characters before writing to a terminal."""
        cleaned: List[str] = []
        for character in str(value):
            if character == "\t":
                cleaned.append("    ")
            elif character in ("\r", "\n"):
                cleaned.append(" ")
            elif unicodedata.category(character) in ("Cc", "Cf", "Cs"):
                continue
            else:
                cleaned.append(character)
        return "".join(cleaned)

    @staticmethod
    def display_fragments(chunk: str, width: int = 1) -> List[str]:
        """Animate ordinary stream chunks while bounding work for oversized ones."""
        minimum_width = max(width, 1)
        if minimum_width == 1 and len(chunk) <= MAX_CHARACTER_STREAM_CHUNK_CHARS:
            fragment_width = 1
        else:
            fragment_width = max(
                minimum_width,
                (len(chunk) + MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK - 1)
                // MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK,
            )
        return [chunk[index : index + fragment_width] for index in range(0, len(chunk), fragment_width)] or [""]

    @staticmethod
    def show_cursor() -> None:
        try:
            curses.curs_set(1)
        except curses.error:
            pass

    def initialize_theme(self, screen: "curses._CursesWindow") -> None:
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            theme = self.theme
            # Paint a predictable dark canvas instead of inheriting an arbitrary
            # terminal profile background (some macOS profiles default to blue).
            curses.init_pair(1, theme.header, curses.COLOR_BLACK)
            curses.init_pair(2, theme.accent, curses.COLOR_BLACK)
            curses.init_pair(3, theme.text, curses.COLOR_BLACK)
            curses.init_pair(4, theme.user, curses.COLOR_BLACK)
            curses.init_pair(5, theme.warning, curses.COLOR_BLACK)
            curses.init_pair(6, theme.error, curses.COLOR_BLACK)
            curses.init_pair(8, theme.selected_text, theme.selected_background)
            curses.init_pair(9, theme.text, curses.COLOR_BLACK)
            self.colors = {
                "base": curses.color_pair(9),
                "header": curses.color_pair(1),
                "header_muted": curses.color_pair(1) | curses.A_DIM,
                "accent": curses.color_pair(2),
                "assistant": curses.color_pair(2),
                "text": curses.color_pair(3),
                "user": curses.color_pair(4),
                "ok": curses.color_pair(4),
                "warning": curses.color_pair(5),
                "error": curses.color_pair(6),
                "muted": curses.color_pair(3) | curses.A_DIM,
                "border": curses.color_pair(3) | curses.A_DIM,
                "footer": curses.color_pair(3) | curses.A_DIM,
                "input": curses.color_pair(3),
                "selected": curses.color_pair(8) | curses.A_BOLD,
            }
            screen.bkgd(" ", self.style("base"))
            screen.erase()
        except curses.error:
            self.colors = {}

    def style(self, name: str, extra: int = 0) -> int:
        return self.colors.get(name, 0) | extra

    def status_style(self) -> int:
        lowered = self.status.lower()
        if any(token in lowered for token in ("http", "error", "no api key", "network", "invalid", "timed out")):
            return self.style("error", curses.A_BOLD)
        if any(token in lowered for token in ("waiting", "checking", "legacy", "unavailable")):
            return self.style("warning")
        return self.style("ok")

    def status_lines(self, width: int) -> List[str]:
        lines = textwrap.wrap(self.status, width=max(width - 3, 20), break_long_words=True) or [""]
        if len(lines) > 2:
            lines = [lines[0], lines[1] + " ..."]
        return lines + [""] * (2 - len(lines))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal UI for Gemini Flash and legacy PaLM chat models.")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read the Gemini API key from a file instead of exposing it in command-line arguments.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.cwd() / ".gemini-legacy-tui",
        help="Directory used for the local conversation transcript (default: ./.gemini-legacy-tui).",
    )
    parser.add_argument("--list-models", action="store_true", help="Print the built-in model catalog and exit.")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.list_models:
        for model in MODEL_CATALOG:
            print("{}\t{}\t{}".format(model.identifier, model.group, model.availability_note))
        return 0
    restart_key = os.environ.pop("GEMINI_LEGACY_TUI_RESTART_KEY", "")
    api_key_from_file = ""
    if arguments.api_key_file:
        try:
            api_key_from_file = read_api_key_file(arguments.api_key_file)
        except (ApiError, OSError, UnicodeError, ValueError) as error:
            print("Could not read API key file: {}".format(error), file=sys.stderr)
            return 2
    api_key = (
        api_key_from_file
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or restart_key
    )
    try:
        api_key = validate_api_key(api_key)
    except ApiError as error:
        print("Could not use API key: {}".format(error), file=sys.stderr)
        return 2
    store = SessionStore(arguments.state_dir)
    app = Tui(api_key, store)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        return 0
    if app.restart_requested:
        restart_environment = os.environ.copy()
        if app.api_key:
            restart_environment["GEMINI_LEGACY_TUI_RESTART_KEY"] = app.api_key
        try:
            os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:], restart_environment)
        except OSError:
            # A rare exec failure should still leave the user in a usable TUI.
            return main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
