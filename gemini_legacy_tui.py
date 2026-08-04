#!/usr/bin/env python3
"""A dependency-free terminal chat client for Gemini Flash and legacy PaLM Chat."""

import argparse
import curses
import json
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


APP_NAME = "Gemini Legacy TUI"
STATE_FILE_NAME = "session.json"
USER_AGENT = "gemini-legacy-tui/1.0"


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    label: str
    group: str
    protocol: str  # gemini or palm
    availability_note: str


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
SLASH_COMMANDS: Sequence[Tuple[str, str]] = (
    ("/clear", "Clear the shared conversation context"),
    ("/model", "Choose a model"),
    ("/models", "View the configured model catalog"),
    ("/system", "Set the system instruction"),
    ("/key", "Enter an API key for this run"),
    ("/check", "Check models available to this key"),
    ("/restart", "Close and reopen the TUI"),
    ("/help", "Show help and shortcuts"),
    ("/quit", "Exit Gemini Legacy"),
)


class ApiError(RuntimeError):
    """A clean message for API and transport errors."""


class GoogleApiClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()

    def _request(self, url: str, body: Optional[dict] = None) -> dict:
        if not self.api_key:
            raise ApiError("No API key. Set GEMINI_API_KEY or press F4 to enter one.")

        headers = {"User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(raw).get("error", {}).get("message", raw)
            except json.JSONDecodeError:
                message = raw
            raise ApiError("HTTP {}: {}".format(error.code, message)) from error
        except urllib.error.URLError as error:
            raise ApiError("Network error: {}".format(error.reason)) from error
        except json.JSONDecodeError as error:
            raise ApiError("Google returned an invalid JSON response.") from error

    def list_models(self) -> List[str]:
        url = "https://generativelanguage.googleapis.com/v1beta/models?key={}".format(
            urllib.parse.quote(self.api_key, safe="")
        )
        response = self._request(url)
        return [model.get("name", "").removeprefix("models/") for model in response.get("models", [])]

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
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(
            urllib.parse.quote(model.identifier, safe=""), urllib.parse.quote(self.api_key, safe="")
        )
        response = self._request(url, self._gemini_body(history, system_instruction))
        candidates = response.get("candidates", [])
        if not candidates:
            detail = response.get("promptFeedback", {}).get("blockReason", "no candidate returned")
            raise ApiError("The request produced no response: {}.".format(detail))
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part["text"] for part in parts if "text" in part).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
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
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse&key={}".format(
            urllib.parse.quote(model.identifier, safe=""), urllib.parse.quote(self.api_key, safe="")
        )
        headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(url, data=json.dumps(self._gemini_body(history, system_instruction)).encode("utf-8"), headers=headers, method="POST")
        received: List[str] = []
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError as error:
                        raise ApiError("Google returned an invalid streaming event.") from error
                    candidates = event.get("candidates", [])
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    chunk = "".join(part["text"] for part in parts if "text" in part)
                    if chunk:
                        received.append(chunk)
                        on_text(chunk)
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(raw).get("error", {}).get("message", raw)
            except json.JSONDecodeError:
                message = raw
            raise ApiError("HTTP {}: {}".format(error.code, message)) from error
        except urllib.error.URLError as error:
            raise ApiError("Network error: {}".format(error.reason)) from error
        text = "".join(received).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
        return text

    def _generate_palm(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        # PaLM Chat used a different, now-retired API shape. Keeping this request
        # lets accounts with an available legacy endpoint attempt the original model.
        url = "https://generativelanguage.googleapis.com/v1beta3/models/{}:generateMessage?key={}".format(
            urllib.parse.quote(model.identifier, safe=""), urllib.parse.quote(self.api_key, safe="")
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
        if not candidates or not candidates[0].get("content"):
            raise ApiError("The PaLM endpoint returned no text.")
        return candidates[0]["content"].strip()


class SessionStore:
    def __init__(self, state_dir: Path) -> None:
        self.path = state_dir / STATE_FILE_NAME

    def load(self) -> dict:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data.get("history", []), list):
                raise ValueError("history is not a list")
            return data
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return {"model": DEFAULT_MODEL_ID, "system_instruction": "", "history": []}

    def save(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=True)
            temporary.replace(self.path)
        except OSError:
            pass  # A read-only directory should not stop a chat session.


class Tui:
    def __init__(self, api_key: str, store: SessionStore) -> None:
        self.api_key = api_key.strip()
        self.store = store
        self.data = store.load()
        if self.data.get("model") not in MODEL_BY_ID:
            self.data["model"] = DEFAULT_MODEL_ID
        self.status = "Ready"
        self.available_models: Optional[set] = None
        self.running = True
        self.restart_requested = False
        self.colors: Dict[str, int] = {}

    @property
    def model(self) -> ModelSpec:
        return MODEL_BY_ID[self.data["model"]]

    def run(self, screen: "curses._CursesWindow") -> None:
        self.initialize_theme(screen)
        curses.curs_set(1)
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
        lines: List[Tuple[str, int]] = []
        for item in self.data["history"]:
            is_user = item["role"] == "user"
            label = "YOU" if is_user else item.get("model_id", "UNKNOWN MODEL").upper()
            role_style = self.style("user" if is_user else "assistant", curses.A_BOLD)
            wrapped = textwrap.wrap(item["content"], width=max(width - 7, 20), replace_whitespace=False) or [""]
            lines.append(("  {}  {}".format(label, wrapped[0]), role_style))
            lines.extend(("       " + line, self.style("text")) for line in wrapped[1:])
            lines.append(("", 0))

        if not lines:
            empty_row = max(5, (height - 8) // 2)
            self.add(screen, empty_row, 4, "Start a conversation", self.style("text", curses.A_BOLD))
            self.add(screen, empty_row + 2, 4, "Tab opens actions. Ctrl+L chooses a model.", self.style("muted"))
        visible = lines[-history_height:]
        for row, (line, style) in enumerate(visible, start=3):
            self.add(screen, row, 0, line[: max(width - 1, 0)], style)
        self.add(screen, height - 6, 0, "-" * max(width - 1, 0), self.style("border"))
        for index, line in enumerate(self.status_lines(width), start=height - 5):
            self.add(screen, index, 1, line, self.status_style())
        self.add(screen, height - 3, 0, "-" * max(width - 1, 0), self.style("border"))
        self.add(screen, height - 2, 1, ">", self.style("accent", curses.A_BOLD))
        self.add(screen, height - 1, 1, "Tab Actions   Ctrl+L Models   Ctrl+N Clear   F3 System   F4 Key   F5 Check", self.style("footer"))
        screen.refresh()

    def draw_compact_chat(self, screen: "curses._CursesWindow") -> None:
        height, width = screen.getmaxyx()
        title = " {} | {} ".format(APP_NAME, self.model.label)
        self.add(screen, 0, 0, title[: max(width - 1, 0)], curses.A_REVERSE)
        history_height = max(height - 4, 1)
        lines: List[Tuple[str, int]] = []
        for item in self.data["history"]:
            label = "You" if item["role"] == "user" else item.get("model_id", "Unknown model")
            wrapped = textwrap.wrap(item["content"], width=max(width - 5, 20), replace_whitespace=False) or [""]
            lines.append((label + ": " + wrapped[0], curses.A_BOLD))
            lines.extend(("   " + line, 0) for line in wrapped[1:])
            lines.append(("", 0))
        for row, (line, style) in enumerate(lines[-history_height:], start=1):
            self.add(screen, row, 0, line[: max(width - 1, 0)], style)
        self.add(screen, height - 3, 0, "-" * max(width - 1, 0))
        self.add(screen, height - 2, 0, self.status[: max(width - 1, 0)], curses.A_DIM)
        self.add(screen, height - 1, 0, "You > ")

    def read_line(self, screen: "curses._CursesWindow", label: str) -> Optional[str]:
        height, width = screen.getmaxyx()
        buffer: List[str] = []
        cursor = 0
        while True:
            prompt_row = height - 2 if height >= 10 and width >= 42 else height - 1
            prompt_label = "> " if prompt_row != height - 1 else label
            available_width = max(width - len(prompt_label) - 2, 1)
            visible = "".join(buffer)
            start = max(0, cursor - available_width + 1)
            rendered = visible[start : start + available_width]
            self.add(screen, prompt_row, 0, " " * max(width - 1, 0))
            self.add(screen, prompt_row, 1 if prompt_row != height - 1 else 0, prompt_label + rendered, self.style("input"))
            screen.move(prompt_row, min((1 if prompt_row != height - 1 else 0) + len(prompt_label) + cursor - start, width - 1))
            screen.refresh()
            key = screen.get_wch()
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
            if key == curses.KEY_HOME:
                cursor = 0
                continue
            if key == curses.KEY_END:
                cursor = len(buffer)
                continue
            if isinstance(key, str) and key.isprintable():
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
        selected_model = self.model
        self.data["history"].append({"role": "user", "content": text})
        self.store.save(self.data)
        self.status = "Streaming {}...".format(selected_model.label)
        try:
            if selected_model.protocol == "gemini":
                reply = {"role": "model", "content": "", "model_id": selected_model.identifier}
                self.data["history"].append(reply)
                self.draw_chat(screen)

                def append_chunk(chunk: str) -> None:
                    reply["content"] += chunk
                    self.draw_chat(screen)

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
        except ApiError as error:
            if self.data["history"] and self.data["history"][-1].get("role") == "model":
                self.data["history"].pop()
            if self.data["history"] and self.data["history"][-1].get("role") == "user":
                self.data["history"].pop()
            self.status = self.friendly_error(error, selected_model)
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
        self.store.save(self.data)
        self.status = "Context cleared."

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
        elif command == "/system":
            self.data["system_instruction"] = argument.strip()
            self.store.save(self.data)
            self.status = "System instruction {}.".format("updated" if argument.strip() else "cleared")
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
            ("Clear context", "Remove the shared conversation history", "clear"),
            ("Choose model", "Browse and filter the model catalog", "model"),
            ("System instruction", "Set a persistent instruction for this chat", "system"),
            ("API key", "Enter a key for this run only", "key"),
            ("Check availability", "List models visible to this API key", "check"),
            ("Restart interface", "Close and reopen the terminal interface", "restart"),
            ("Help", "Show shortcuts and slash commands", "help"),
            ("Exit", "Close Gemini Legacy", "exit"),
        )
        selected = 0
        try:
            curses.curs_set(0)
        except curses.error:
            pass

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
            for index, (command, description) in enumerate(matches):
                row = 5 + index * 2
                if row >= height - 2:
                    break
                is_selected = command == selected_command
                self.add(screen, row, 2, ("> " if is_selected else "  ") + command, self.style("selected") if is_selected else self.style("accent", curses.A_BOLD))
                if row + 1 < height - 1:
                    self.add(screen, row + 1, 6, description[: width - 8], self.style("muted"))
            if not matches:
                self.add(screen, 5, 2, "No matching command.", self.style("warning"))
            self.add(screen, height - 1, 2, "Type to filter   Up/Down select   Enter run   Esc return to input", self.style("footer"))
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
            elif isinstance(key, str) and key.isprintable():
                query += key
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
            elif action == "system":
                self.edit_system_instruction(screen)
            elif action == "key":
                self.configure_api_key(screen)
            elif action == "check":
                self.check_availability(screen)
            elif action == "restart":
                self.restart_requested = True
                self.running = False
            elif action == "help":
                self.show_help(screen)
            elif action == "exit":
                self.running = False
            break
        try:
            curses.curs_set(1)
        except curses.error:
            pass

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
            elif isinstance(key, str) and key.isprintable() and key not in ("j", "k"):
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
        value = self.prompt_dialog(screen, "System instruction", self.data.get("system_instruction", ""))
        if value is not None:
            self.data["system_instruction"] = value
            self.store.save(self.data)
            self.status = "System instruction {}.".format("updated" if value else "cleared")

    def configure_api_key(self, screen: "curses._CursesWindow") -> None:
        value = self.prompt_dialog(screen, "API key (kept only for this run)", "", secret=True)
        if value:
            self.api_key = value.strip()
            self.available_models = None
            self.status = "API key updated for this session."
        elif value == "":
            self.status = "API key was unchanged."

    def prompt_dialog(self, screen: "curses._CursesWindow", title: str, initial: str, secret: bool = False) -> Optional[str]:
        height, width = screen.getmaxyx()
        window_height = 7
        window_width = min(max(50, len(title) + 8), max(width - 2, 20))
        window = curses.newwin(window_height, window_width, max((height - window_height) // 2, 0), max((width - window_width) // 2, 0))
        window.keypad(True)
        buffer = list(initial)
        cursor = len(buffer)
        while True:
            window.erase()
            window.box()
            self.add(window, 1, 2, title[: window_width - 4], curses.A_BOLD)
            shown = "*" * len(buffer) if secret else "".join(buffer)
            start = max(0, cursor - (window_width - 6))
            self.add(window, 3, 2, shown[start : start + window_width - 5])
            self.add(window, 5, 2, "Enter: confirm  Esc: cancel"[: window_width - 4], curses.A_DIM)
            window.move(3, min(2 + cursor - start, window_width - 3))
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
                buffer.insert(cursor, key)
                cursor += 1

    def show_help(self, screen: "curses._CursesWindow") -> None:
        self.show_message(
            screen,
            "Help",
            [
                "Type / for the command palette | Tab or Ctrl+K actions | Ctrl+L model picker | Ctrl+N clear context",
                "/new and /clear erase the shared context; /model opens the picker; /models shows the catalog.",
                "/system <text> sets the system instruction; /key enters an API key; /check checks models; /restart reopens the UI.",
                "/quit exits. API keys are never written to disk. Transcripts live in the selected state folder.",
            ],
        )

    def show_models(self, screen: "curses._CursesWindow") -> None:
        lines = ["{} [{}] - {}".format(model.identifier, model.group, self.availability_marker(model)) for model in MODEL_CATALOG]
        self.show_message(screen, "Configured model catalog", lines)

    def show_message(self, screen: "curses._CursesWindow", title: str, lines: Sequence[str]) -> None:
        height, width = screen.getmaxyx()
        screen.erase()
        self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("header"))
        self.add(screen, 0, 1, title[: width - 2], self.style("header", curses.A_BOLD))
        row = 2
        for line in lines:
            for wrapped in textwrap.wrap(line, width=max(width - 2, 20)) or [""]:
                if row >= height - 2:
                    break
                self.add(screen, row, 1, wrapped, self.style("text"))
                row += 1
            if row >= height - 2:
                break
        self.add(screen, height - 1, 1, "Press any key to return."[: width - 2], self.style("footer"))
        screen.refresh()
        screen.get_wch()

    @staticmethod
    def add(screen: "curses._CursesWindow", row: int, column: int, value: str, style: int = 0) -> None:
        try:
            screen.addstr(row, column, value, style)
        except curses.error:
            pass

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
            # Paint a predictable dark canvas instead of inheriting an arbitrary
            # terminal profile background (some macOS profiles default to blue).
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(6, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLACK)
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(9, curses.COLOR_WHITE, curses.COLOR_BLACK)
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
        if any(token in lowered for token in ("http", "error", "no api key", "network", "invalid")):
            return self.style("error", curses.A_BOLD)
        if any(token in lowered for token in ("waiting", "checking", "legacy", "unavailable")):
            return self.style("warning")
        return self.style("ok")

    def status_lines(self, width: int) -> List[str]:
        lines = textwrap.wrap(self.status, width=max(width - 3, 20), break_long_words=False) or [""]
        if len(lines) > 2:
            lines = [lines[0], lines[1] + " ..."]
        return lines + [""] * (2 - len(lines))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal UI for Gemini Flash and legacy PaLM chat models.")
    parser.add_argument("--api-key", help="Gemini API key. Overrides GEMINI_API_KEY and GOOGLE_API_KEY.")
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
    api_key = arguments.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or restart_key
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
