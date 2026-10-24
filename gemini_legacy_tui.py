#!/usr/bin/env python3
"""A lightweight terminal chat client for Gemini models and legacy PaLM Chat."""

import argparse
import atexit
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import textwrap
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
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
MAX_INPUT_ROWS = 5
MAX_MODEL_ID_CHARS = 256
MAX_SESSION_BYTES = 8 * 1024 * 1024
MAX_CHARACTER_STREAM_CHUNK_CHARS = 512
MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK = 64
STREAM_RENDER_DELAY_MS = 12
MAX_STREAM_EVENT_BYTES = 1024 * 1024
MAX_STREAM_TEXT_CHARS = 2 * 1024 * 1024
API_BASE_URL = "https://generativelanguage.googleapis.com"
LMSTUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_CONTEXT_WINDOW = 1_048_576  # input tokens; replaced by the API's inputTokenLimit once known
MOCK_PORT = 8765
MOCK_BASE_URL = "http://127.0.0.1:{}".format(MOCK_PORT)
MOCK_API_KEY = "mock"  # sent to the local mock server instead of your real key
MAX_TERMINAL_RESULTS = 10
STOP_HINT = "press esc to stop turn."
TERMINAL_WARNING_RESULTS = (8, 9)
TOOL_TIMEOUT_SECONDS = 30
MAX_TOOL_OUTPUT_CHARS = 8000
TERMINAL_TOOL_NAME = "run_terminal_command"
TERMINAL_TOOL = {
    "functionDeclarations": [
        {
            "name": TERMINAL_TOOL_NAME,
            "description": (
                "Run a shell command on the user's computer and return its combined output and exit code. "
                "The user approves every command before it runs. Commands must be non-interactive and finish "
                "within {} seconds.".format(TOOL_TIMEOUT_SECONDS)
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {"command": {"type": "STRING", "description": "The shell command to run."}},
                "required": ["command"],
            },
        }
    ]
}
SECRET_ENVIRONMENT_VARIABLES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_LEGACY_TUI_RESTART_KEY")


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    label: str
    group: str
    protocol: str  # gemini or palm
    availability_note: str
    retired: bool = False  # shown greyed out in the picker but still selectable
    local_mock: bool = False  # runs against tools/mock_gemini_server.py; no API key or credits needed


@dataclass(frozen=True)
class ThemeSpec:
    identifier: str
    label: str
    description: str
    background: int
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
        curses.COLOR_BLACK,
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
        curses.COLOR_BLACK,
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
        curses.COLOR_BLACK,
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
        "Cool blue canvas with high-contrast cyan conversation markers",
        curses.COLOR_BLUE,
        curses.COLOR_BLUE,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE,
        curses.COLOR_CYAN,
        curses.COLOR_YELLOW,
        curses.COLOR_RED,
        curses.COLOR_BLACK,
        curses.COLOR_CYAN,
    ),
    ThemeSpec(
        "neon",
        "Neon Noir",
        "Magenta headers and cyan responses on black",
        curses.COLOR_BLACK,
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
    ModelSpec("gemini-4-argon", "Gemini 4 Argon", "Gemini 4", "gemini", "Current stable"),
    ModelSpec("gemini-3.8-flash", "Gemini 3.8 Flash", "Gemini Flash", "gemini", "Current stable"),
    ModelSpec("gemini-3.7-flash", "Gemini 3.7 Flash", "Gemini Flash", "gemini", "Available through your API key"),
    ModelSpec("gemini-3.6-flash", "Gemini 3.6 Flash", "Gemini Flash", "gemini", "Current stable"),
    ModelSpec("gemini-3-flash-preview", "Gemini 3 Flash", "Gemini Flash", "gemini", "Preview"),
    ModelSpec("gemini-2.5-flash", "Gemini 2.5 Flash", "Gemini Flash", "gemini", "Previously active users only"),
    ModelSpec("gemini-flash-latest", "Gemini Flash (latest alias)", "Gemini Flash", "gemini", "Moving alias"),
    ModelSpec("gemini-mock-model", "Mock model (local test server)", "Testing", "gemini", "Fake replies from tools/mock_gemini_server.py on this machine; no API key or credits", False, True),
    ModelSpec("gemini-3.5-flash", "Gemini 3.5 Flash", "Unavailable", "gemini", "Not on Google's current model list; likely 404", True),
    ModelSpec("gemini-1.5-flash", "Gemini 1.5 Flash", "Unavailable", "gemini", "Legacy; shut down by Google", True),
    ModelSpec("gemini-1.5-flash-001", "Gemini 1.5 Flash 001", "Unavailable", "gemini", "Legacy; likely unavailable", True),
    ModelSpec("gemini-1.5-flash-002", "Gemini 1.5 Flash 002", "Unavailable", "gemini", "Legacy; likely unavailable", True),
    ModelSpec("gemini-2.0-flash", "Gemini 2.0 Flash", "Unavailable", "gemini", "Legacy; shut down by Google", True),
    ModelSpec("gemini-2.0-flash-001", "Gemini 2.0 Flash 001", "Unavailable", "gemini", "Legacy; shut down by Google", True),
    ModelSpec("gemini-2.0-flash-exp", "Gemini 2.0 Flash Experimental", "Unavailable", "gemini", "Experimental; likely unavailable", True),
    ModelSpec("chat-bison-001", "PaLM 2 Chat-Bison 001", "Unavailable", "palm", "Legacy PaLM; shut down by Google (HTTP 501)", True),
)

MODEL_BY_ID = {model.identifier: model for model in MODEL_CATALOG}
DEFAULT_MODEL_ID = "gemini-2.5-flash"
CUSTOM_MODEL_PICKER_ENTRY = ModelSpec(
    "__custom_gemini_model__",
    "Custom Gemini model ID",
    "Custom",
    "gemini",
    "Enter a public Gemini model ID manually",
)
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant replying inside a terminal chat window. Write plain text only: no Markdown "
    "(no asterisks, underscores, backticks, heading hashes, quote markers, tables, or separator lines) and no emoji. "
    "Lead with the answer, keep it short and direct, and add detail only when it helps. Separate paragraphs with a "
    "blank line, and write lists as simple lines starting with a hyphen or a number. Put code, commands, and file "
    "contents on their own lines indented by four spaces so they are easy to copy. If a request is ambiguous, make a "
    "reasonable assumption and say so, or ask one brief question. If you are unsure or do not know, say so instead "
    "of guessing."
)
SLASH_COMMANDS: Sequence[Tuple[str, str]] = (
    ("/clear", "Clear the shared conversation context"),
    ("/model", "Choose a model"),
    ("/retry", "Retry the last failed prompt (only shown after a failed prompt)"),
    ("/thinking", "Thinking level and other generation parameters (temperature, top P, max tokens, ...)"),
    ("/permissions", "Choose how terminal commands are approved (needs Terminal access)"),
    ("/restart", "Close and reopen the TUI"),
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
    if identifier.startswith("lmstudio:"):
        # LM Studio IDs can hold slashes, colons, @, spaces, and similar; they only travel in JSON and quoted paths.
        if identifier == "lmstudio:" or not identifier.isprintable():
            raise ApiError("LM Studio model IDs must be non-empty and contain only printable characters.")
        return identifier
    if not all(
        character.isascii() and (character.isalnum() or character in ("-", "_", "."))
        for character in identifier
    ):
        raise ApiError("Model IDs can contain only letters, numbers, hyphens, underscores, and periods.")
    return identifier


def model_supports_terminal_tools(identifier: str) -> bool:
    """Terminal access is limited to gemini-flash-latest, Gemini 4 and later, and Gemini 3.8 Flash and later."""
    if identifier in ("gemini-flash-latest", "gemini-mock-model"):
        return True
    if any(token in identifier for token in ("lite", "image", "tts", "live", "embedding", "transcribe")):
        return False
    match = re.match(r"gemini-(\d+)(?:\.(\d+))?-", identifier)
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2) or 0)
    if major >= 4:
        return True
    return (major, minor) >= (3, 8) and "flash" in identifier


def run_shell_command(
    command: str, timeout: int = TOOL_TIMEOUT_SECONDS, should_stop: Optional[Callable[[], bool]] = None
) -> dict:
    """Run one approved shell command with a timeout and bounded output; API keys are removed from its environment.

    should_stop is polled while the command runs; when it returns True the command is killed."""
    environment = {key: value for key, value in os.environ.items() if key not in SECRET_ENVIRONMENT_VARIABLES}
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=environment,
            start_new_session=(os.name != "nt"),
        )
    except OSError as error:
        return {"error": "Could not start the command: {}".format(error)}
    kept: List[bytes] = []
    state = {"size": 0, "truncated": False, "dropped_bytes": 0}

    def drain() -> None:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            room = MAX_TOOL_OUTPUT_CHARS * 4 - state["size"]
            if room > 0:
                kept.append(chunk[:room])
                state["size"] += min(len(chunk), room)
            if len(chunk) > room:
                state["truncated"] = True
                state["dropped_bytes"] += len(chunk) - max(room, 0)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    stopped = False
    deadline = time.monotonic() + timeout
    while True:
        try:
            process.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            stopped = bool(should_stop and should_stop())
            timed_out = not stopped and time.monotonic() >= deadline
            if stopped or timed_out:
                try:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, 9)
                except OSError:
                    pass
                process.wait()
                break
    reader.join(timeout=2)
    process.stdout.close()
    output = b"".join(kept).decode("utf-8", errors="replace")
    omitted = 0
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        omitted = len(output) - MAX_TOOL_OUTPUT_CHARS
        output = output[:MAX_TOOL_OUTPUT_CHARS]
        state["truncated"] = True
    if state["truncated"]:
        # Bytes past the read cap were discarded unread, so that part of the count is approximate.
        omitted += state["dropped_bytes"]
        output += "\n[truncated at {} characters; {}{} characters omitted]".format(
            MAX_TOOL_OUTPUT_CHARS, "about " if state["dropped_bytes"] else "", omitted
        )
    result = {"exit_code": process.returncode, "output": output}
    if state["truncated"]:
        result["output_truncated"] = True
    if timed_out:
        result["timed_out"] = True
    if stopped:
        result["stopped_by_user"] = True
    return result


CATEGORY_ORDER = ("READ", "CREATION", "WRITE", "REMOVAL")
SHELL_NAMES = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
INTERPRETER_NAMES = {"python", "python3", "node", "ruby", "perl", "php", "deno", "bun", "osascript"}
WRAPPER_NAMES = {"env", "command", "time", "nohup", "nice", "xargs", "builtin", "exec"}
PRIVILEGE_NAMES = {"sudo", "su", "doas"}
XARGS_VALUE_FLAGS = {"-n", "-I", "-P", "-L", "-s", "-d", "-E", "-a"}
READ_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag", "pwd", "echo", "printf", "wc", "file", "stat",
    "du", "df", "ps", "whoami", "id", "date", "uname", "which", "whereis", "type", "env", "printenv", "sort", "uniq",
    "diff", "cmp", "less", "more", "tree", "basename", "dirname", "realpath", "readlink", "cut", "tr", "awk", "jq",
    "hostname", "uptime", "lsof", "top", "true", "false", "test", "[", "man", "tldr", "ping", "dig", "nslookup",
    "sw_vers", "sysctl", "md5", "md5sum", "shasum", "sha256sum", "column", "nl", "rev", "seq", "sleep", "cd", "ifconfig",
    "mdfind", "xxd", "hexdump", "strings", "system_profiler", "ioreg", "vm_stat", "lscpu", "free", "arch", "groups",
    "who", "w", "last", "getconf", "mdls", "netstat", "ss", "traceroute", "host", "cal", "bc", "expr", "lsb_release",
}
REMOVAL_COMMANDS = {"rm", "rmdir", "unlink", "shred", "srm", "trash"}
WRITE_COMMANDS = {
    "mv", "chmod", "chown", "chgrp", "truncate", "patch", "dd", "rsync", "defaults", "xattr", "make", "gzip", "gunzip",
    "kill", "pkill", "killall", "launchctl", "crontab", "diskutil", "fdisk", "shutdown", "reboot", "halt", "networksetup",
    "systemctl", "mount", "umount", "ditto", "cmake", "gcc", "g++", "clang", "cc", "go", "rustc", "javac", "tmux", "screen",
}
CREATION_COMMANDS = {"mkdir", "mktemp", "ln", "zip"}
SYSTEM_RISK_COMMANDS = {
    "dd", "diskutil", "fdisk", "shutdown", "reboot", "halt", "launchctl", "crontab", "kill", "pkill", "killall",
    "networksetup", "systemctl", "mount", "umount", "eval", "osascript",
}
PACKAGE_MANAGERS = {"pip", "pip3", "npm", "pnpm", "yarn", "brew", "cargo", "gem", "apt", "apt-get", "conda"}
GIT_READ = {
    "status", "log", "diff", "show", "blame", "ls-files", "ls-tree", "rev-parse", "describe", "shortlog", "grep",
    "cat-file", "help", "version", "rev-list", "count-objects", "whatchanged",
}
NON_FILE_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}


@dataclass
class CommandReview:
    categories: List[str]  # distinct categories, least to most severe
    risks: List[str]  # reasons the command must always be confirmed, even after "allow all"
    notes: List[str]

    @property
    def primary(self) -> str:
        return self.categories[-1]

    @property
    def needs_confirmation(self) -> bool:
        """Risky commands and removals ask every time unless the user allows literally everything."""
        return bool(self.risks) or "REMOVAL" in self.categories


def _outside_working_directory(argument: str) -> bool:
    if not argument or argument.startswith("-"):
        return False
    if argument.startswith("$") or argument.startswith("~"):
        return True
    path = Path(argument)
    if not path.is_absolute() and ".." not in path.parts:
        return False
    try:
        resolved = (path if path.is_absolute() else Path.cwd() / path).resolve()
        base = Path.cwd().resolve()
    except (OSError, RuntimeError):
        return True
    return resolved != base and base not in resolved.parents


def _split_command(command: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Split on shell operators outside quotes. Returns ([(operator before, segment)], redirect targets)."""
    segments: List[Tuple[str, str]] = []
    redirects: List[str] = []
    current: List[str] = []
    operator = ""
    quote = ""
    index = 0
    length = len(command)

    def flush(next_operator: str) -> None:
        nonlocal operator, current
        segments.append((operator, "".join(current)))
        operator = next_operator
        current = []

    while index < length:
        char = command[index]
        if quote:
            current.append(char)
            if char == "\\" and quote == '"' and index + 1 < length:
                current.append(command[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            current.append(command[index : index + 2])
            index += 2
            continue
        if char in "'\"":
            quote = char
            current.append(char)
            index += 1
            continue
        if command[index : index + 2] in ("&&", "||"):
            flush(command[index : index + 2])
            index += 2
            continue
        if char == ">":
            end = index + 1
            if command[end : end + 1] == ">":
                end += 1
            if command[end : end + 1] == "&":
                index = end + 1
                continue
            while end < length and command[end] in " \t":
                end += 1
            start = end
            while end < length and command[end] not in " \t\n;|&()<>`":
                end += 1
            redirects.append(command[start:end].strip("'\""))
            index = end
            continue
        if char in ";|\n`()":
            flush(char)
            index += 1
            continue
        if char == "&" and command[index + 1 : index + 2] != ">":
            flush("&")
            index += 1
            continue
        current.append(char)
        index += 1
    flush("")
    return segments, redirects


def _words(segment: str) -> List[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _review_words(words: List[str], review: "CommandReview", cats: set, depth: int) -> None:
    index = 0

    def skip_assignments() -> None:
        nonlocal index
        while index < len(words) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[index]):
            index += 1

    skip_assignments()
    while index < len(words) and os.path.basename(words[index]) in WRAPPER_NAMES | PRIVILEGE_NAMES:
        wrapper = os.path.basename(words[index])
        if wrapper in PRIVILEGE_NAMES:
            review.risks.append("runs with elevated privileges ({})".format(wrapper))
        index += 1
        while index < len(words) and words[index].startswith("-"):
            flag = words[index]
            index += 1
            if wrapper == "xargs" and flag in XARGS_VALUE_FLAGS and index < len(words):
                index += 1
        skip_assignments()
    if index >= len(words):
        return
    name = os.path.basename(words[index])
    args = words[index + 1 :]
    flags = [argument for argument in args if argument.startswith("-")]
    operands = [argument for argument in args if not argument.startswith("-")]

    def has_flag(*wanted: str) -> bool:
        return any(flag in wanted for flag in flags)

    def path_check() -> None:
        if any(_outside_working_directory(operand) for operand in operands):
            review.risks.append("touches paths outside the working directory")

    if name in SYSTEM_RISK_COMMANDS or name.startswith("mkfs"):
        review.risks.append("system-level command ({})".format(name))
    if name in REMOVAL_COMMANDS:
        cats.add("REMOVAL")
        path_check()
    elif name in SHELL_NAMES:
        script = None
        for position, argument in enumerate(args):
            if re.fullmatch(r"-[A-Za-z]*c", argument) and position + 1 < len(args):
                script = args[position + 1]
                break
        if script is not None:
            _classify_into(script, review, cats, depth + 1)
        else:
            cats.add("WRITE")
            review.notes.append("Runs a shell script; its effects cannot be analyzed.")
    elif name in INTERPRETER_NAMES:
        cats.add("WRITE")
        review.notes.append("Runs {} code; its effects cannot be analyzed.".format(name))
        if has_flag("-c", "-e", "-E", "-p", "-r") or name == "osascript":
            review.risks.append("runs inline {} code".format(name))
    elif name == "eval":
        cats.add("WRITE")
    elif name == "git":
        _review_git(args, review, cats)
    elif name in PACKAGE_MANAGERS:
        action = operands[0] if operands else ""
        if action in ("install", "add", "i", "ci", "update", "upgrade", "upgrade-all"):
            cats.add("CREATION")
            review.risks.append("installs or updates software ({} {})".format(name, action))
        elif action in ("uninstall", "remove", "rm", "purge", "autoremove"):
            cats.add("REMOVAL")
        elif action in ("list", "ls", "show", "search", "info", "view", "outdated", "freeze", "--version", "help", ""):
            cats.add("READ")
        else:
            cats.add("WRITE")
    elif name == "find":
        if "-exec" in args or "-execdir" in args or "-ok" in args:
            position = max(args.index(token) for token in ("-exec", "-execdir", "-ok") if token in args)
            inner = []
            for token in args[position + 1 :]:
                if token in (";", "+", "\\;"):
                    break
                inner.append(token)
            _review_words(inner, review, cats, depth + 1)
        if has_flag("-delete"):
            cats.add("REMOVAL")
        cats.add("READ")
    elif name == "sed":
        cats.add("WRITE" if any(flag.startswith("-i") or flag == "--in-place" for flag in flags) else "READ")
    elif name == "curl":
        if any(flag in ("-o", "-O", "--output", "--remote-name") or flag.startswith("--output") for flag in flags):
            cats.add("CREATION")
        elif any(flag in ("-d", "-F", "-T", "-X", "--upload-file", "--form") or flag.startswith("--data") for flag in flags):
            cats.add("WRITE")
            review.notes.append("Sends data to a remote server.")
        else:
            cats.add("READ")
    elif name == "wget":
        cats.add("READ" if any(flag in ("-O-", "-qO-") for flag in flags) else "CREATION")
    elif name == "tar":
        mode = "".join(flags + operands[:1])
        cats.add("CREATION" if re.search(r"[xc]", mode) else "READ")
    elif name == "unzip":
        cats.add("READ" if has_flag("-l", "-t") else "CREATION")
    elif name == "tee":
        existing = [operand for operand in operands if Path(operand).exists()]
        cats.add("WRITE" if existing or has_flag("-a") else "CREATION")
        path_check()
    elif name in ("cp", "install", "rename"):
        destination = operands[-1] if operands else ""
        cats.add("WRITE" if destination and Path(destination).exists() else "CREATION")
        path_check()
    elif name == "touch":
        cats.add("WRITE" if operands and all(Path(operand).exists() for operand in operands) else "CREATION")
        path_check()
    elif name in CREATION_COMMANDS:
        cats.add("CREATION")
        path_check()
    elif name in WRITE_COMMANDS:
        cats.add("WRITE")
        path_check()
        if name in ("chmod", "chown", "chgrp") and (has_flag("-R", "-r") or any(re.fullmatch(r"-[A-Za-z]*R[A-Za-z]*", flag) for flag in flags)):
            review.risks.append("recursive permission change")
    elif name in READ_COMMANDS:
        cats.add("READ")
    else:
        cats.add("WRITE")
        review.notes.append("Unrecognized command '{}'; treated as a write.".format(name))


def _review_git(args: List[str], review: "CommandReview", cats: set) -> None:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 2 if args[index] in ("-C", "-c") else 1
    action = args[index] if index < len(args) else ""
    rest = args[index + 1 :]
    rest_flags = [argument for argument in rest if argument.startswith("-")]
    rest_operands = [argument for argument in rest if not argument.startswith("-")]
    if action in GIT_READ or action == "":
        cats.add("READ")
    elif action in ("init", "clone"):
        cats.add("CREATION")
    elif action in ("rm", "clean"):
        cats.add("REMOVAL")
    elif action == "branch":
        if any(flag in ("-d", "-D", "--delete") for flag in rest_flags):
            cats.add("REMOVAL")
        elif any(flag in ("-m", "-M", "-c", "-C") for flag in rest_flags):
            cats.add("WRITE")
        else:
            cats.add("CREATION" if rest_operands else "READ")
    elif action == "tag":
        if any(flag in ("-d", "--delete") for flag in rest_flags):
            cats.add("REMOVAL")
        else:
            cats.add("CREATION" if rest_operands else "READ")
    elif action == "remote":
        sub = rest_operands[0] if rest_operands else ""
        cats.add("REMOVAL" if sub in ("remove", "rm") else "WRITE" if sub in ("add", "set-url", "rename") else "READ")
    elif action == "stash":
        sub = rest_operands[0] if rest_operands else ""
        cats.add("REMOVAL" if sub in ("drop", "clear") else "READ" if sub in ("list", "show") else "WRITE")
    elif action == "config":
        cats.add("READ" if any(flag in ("--get", "--get-all", "--list", "-l") for flag in rest_flags) else "WRITE")
    else:
        cats.add("WRITE")
    if action == "push":
        review.risks.append("publishes changes to a remote (git push)")
    if action == "reset" and "--hard" in rest_flags:
        review.risks.append("discards local changes (git reset --hard)")
    if action in ("checkout", "restore") and ("--" in rest or "." in rest_operands):
        review.risks.append("may discard local changes (git {})".format(action))
    if action == "clean":
        review.risks.append("deletes untracked files (git clean)")


def _classify_into(command: str, review: "CommandReview", cats: set, depth: int) -> None:
    if depth > 3:
        review.risks.append("deeply nested command that cannot be fully analyzed")
        cats.add("WRITE")
        return
    segments, redirects = _split_command(command)
    for operator, segment in segments:
        words = _words(segment)
        if not words:
            continue
        if operator == "|":
            lead = os.path.basename(words[0])
            if lead in SHELL_NAMES or lead == "eval":
                review.risks.append("pipes data into a shell")
        _review_words(words, review, cats, depth)
    # Substitutions inside double quotes are not split above, so review them separately.
    substitutions = re.findall(r"\$\(([^()]*)\)", command) + re.findall(r"`([^`]*)`", command)
    for inner in substitutions:
        _classify_into(inner, review, cats, depth + 1)
    if command.count("$(") > len(re.findall(r"\$\(([^()]*)\)", command)):
        review.risks.append("complex command substitution that cannot be fully analyzed")
    for target in redirects:
        if not target or target in NON_FILE_REDIRECT_TARGETS:
            continue
        cats.add("WRITE" if Path(target).expanduser().exists() else "CREATION")
        if _outside_working_directory(target):
            review.risks.append("writes outside the working directory ({})".format(target))


def classify_command(command: str) -> CommandReview:
    """Label a shell command READ, CREATION, WRITE, or REMOVAL and flag commands that must always be confirmed."""
    review = CommandReview(categories=[], risks=[], notes=[])
    cats: set = set()
    _classify_into(command, review, cats, 0)
    review.categories = [category for category in CATEGORY_ORDER if category in cats] or ["READ"]
    review.risks = list(dict.fromkeys(review.risks))
    review.notes = list(dict.fromkeys(review.notes))
    return review


def read_api_key_file(path: Path) -> str:
    with path.open("rb") as handle:
        raw = handle.read(MAX_API_KEY_FILE_BYTES + 1)
    if len(raw) > MAX_API_KEY_FILE_BYTES:
        raise ValueError("the file is too large")
    return validate_api_key(raw.decode("utf-8"))


@dataclass(frozen=True)
class GenerationParameter:
    key: str  # name in the saved session
    label: str
    kind: str  # float, int, or list
    minimum: float
    maximum: float
    description: str


GENERATION_PARAMETERS: Sequence[GenerationParameter] = (
    GenerationParameter("temperature", "Temperature", "float", 0.0, 2.0, "Randomness, 0 to 2. Leave unset unless you need a specific value."),
    GenerationParameter("top_p", "Top P", "float", 0.0, 1.0, "Nucleus sampling cutoff, 0 to 1."),
    GenerationParameter("top_k", "Top K", "int", 1, 1000, "Only sample from the K most likely tokens."),
    GenerationParameter("max_output_tokens", "Max output tokens", "int", 1, 65536, "Hard cap on the reply length. Thinking tokens count toward it."),
    GenerationParameter("seed", "Seed", "int", 0, 2147483647, "Fixed seed for more repeatable replies."),
    GenerationParameter("presence_penalty", "Presence penalty", "float", -2.0, 2.0, "Discourages tokens that have already appeared. -2 to 2."),
    GenerationParameter("frequency_penalty", "Frequency penalty", "float", -2.0, 2.0, "Discourages tokens in proportion to how often they appeared. -2 to 2."),
    GenerationParameter("stop_sequences", "Stop sequences", "list", 0, 5, "Up to 5 comma-separated strings that end the reply when produced."),
)
GENERATION_PARAMETER_BY_KEY = {parameter.key: parameter for parameter in GENERATION_PARAMETERS}
GENERATION_API_NAMES = {
    "temperature": "temperature",
    "top_p": "topP",
    "top_k": "topK",
    "max_output_tokens": "maxOutputTokens",
    "seed": "seed",
    "presence_penalty": "presencePenalty",
    "frequency_penalty": "frequencyPenalty",
}
THINKING_LEVELS_BY_MODEL = {
    "gemini-4-argon": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("minimal", "low", "medium", "high"),
    "gemini-3-flash-preview": ("minimal", "low", "medium", "high"),
    "gemini-flash-latest": ("low", "medium", "high"),
}
MAX_THINKING_BUDGET = 32768
# USD per million tokens as (last day the price applies or None, input, output); output includes thinking tokens.
MODEL_PRICES: Dict[str, Sequence[Tuple[Optional[date], float, float]]] = {
    "gemini-4-argon": ((date(2026, 11, 30), 2.00, 10.00), (None, 4.00, 20.00)),
    "gemini-3.8-flash": ((date(2026, 12, 31), 0.75, 3.75), (None, 1.50, 7.50)),
    "gemini-3.6-flash": ((date(2026, 12, 31), 0.75, 3.75), (None, 1.50, 7.50)),
    "gemini-3-flash-preview": ((None, 0.50, 3.00),),
    "gemini-2.5-flash": ((None, 0.30, 2.50),),
    "gemini-mock-model": ((None, 0.0, 0.0),),
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int, today: Optional[date] = None) -> Optional[float]:
    """Estimated USD for one request, or None when the model's price is not known."""
    if model_id.startswith("lmstudio:"):
        return 0.0  # local models cost nothing to run
    prices = MODEL_PRICES.get(model_id)
    if prices is None:
        known = [name for name in MODEL_PRICES if model_id.startswith(name + "-")]
        prices = MODEL_PRICES[max(known, key=len)] if known else None
    if prices is None:
        return None
    today = today or date.today()
    for last_day, input_price, output_price in prices:
        if last_day is None or today <= last_day:
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return None


def thinking_support(model_id: str) -> Tuple[str, Sequence[str]]:
    """How a model takes thinking settings: ("level", allowed levels), ("budget", ()), or ("none", ())."""
    if model_id.startswith("gemini-2.5"):
        return "budget", ()
    if model_id in THINKING_LEVELS_BY_MODEL:
        return "level", THINKING_LEVELS_BY_MODEL[model_id]
    if model_id.startswith("gemini-"):
        return "level", ("low", "medium", "high")
    return "none", ()


def parse_generation_value(key: str, text: str) -> object:
    """Parse and range-check one generation parameter typed by the user."""
    text = text.strip()
    if key == "thinking_budget":
        try:
            value = int(text)
        except ValueError:
            raise ApiError("Enter a whole number: -1 for dynamic, 0 for off, or a token count.") from None
        if not -1 <= value <= MAX_THINKING_BUDGET:
            raise ApiError("The thinking budget must be -1 to {}.".format(MAX_THINKING_BUDGET))
        return value
    parameter = GENERATION_PARAMETER_BY_KEY[key]
    if parameter.kind == "list":
        values = [item.strip() for item in text.split(",") if item.strip()]
        if len(values) > parameter.maximum or any(len(item) > 64 for item in values):
            raise ApiError("Use at most 5 stop sequences of up to 64 characters.")
        return values
    try:
        value = int(text) if parameter.kind == "int" else float(text)
    except ValueError:
        raise ApiError("Enter a {}.".format("whole number" if parameter.kind == "int" else "number")) from None
    if value != value:
        raise ApiError("Enter a number.")
    if not parameter.minimum <= value <= parameter.maximum:
        raise ApiError("{} must be between {:g} and {:g}.".format(parameter.label, parameter.minimum, parameter.maximum))
    return value


def normalize_generation(raw: object) -> dict:
    """Keep only valid saved generation parameters."""
    cleaned: dict = {}
    if not isinstance(raw, dict):
        return cleaned
    for key, value in raw.items():
        if isinstance(value, bool):
            continue
        try:
            if key == "thinking_level":
                if value in ("minimal", "low", "medium", "high"):
                    cleaned[key] = value
            elif key == "stop_sequences":
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    parsed = parse_generation_value(key, ",".join(value))
                    if parsed:
                        cleaned[key] = parsed
            elif key == "thinking_budget" or key in GENERATION_PARAMETER_BY_KEY:
                if isinstance(value, (int, float)):
                    cleaned[key] = parse_generation_value(key, repr(value))
        except (ApiError, ValueError):
            continue
    return cleaned


def empty_usage_totals() -> dict:
    return {"input": 0, "output": 0, "cost": 0.0, "unpriced": False}


def normalize_lifetime(raw: object) -> dict:
    """All-time usage across every conversation: totals plus a per-model breakdown."""

    def clean(source: object) -> dict:
        totals = empty_usage_totals()
        if isinstance(source, dict):
            for name in ("input", "output"):
                value = source.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10**13:
                    totals[name] = value
            cost = source.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and 0 <= cost < 10**9:
                totals["cost"] = float(cost)
            totals["unpriced"] = source.get("unpriced") is True
        return totals

    lifetime = clean(raw)
    lifetime["models"] = {}
    models = raw.get("models") if isinstance(raw, dict) else None
    if isinstance(models, dict):
        for name, entry in list(models.items())[:200]:
            if isinstance(name, str) and 0 < len(name) <= MAX_MODEL_ID_CHARS:
                lifetime["models"][name] = clean(entry)
    return lifetime


def build_generation_config(generation: dict, model_id: str) -> dict:
    """The generationConfig sent to the API: only parameters the user set, plus thinking for supporting models."""
    config: dict = {}
    for key, api_name in GENERATION_API_NAMES.items():
        if key in generation:
            config[api_name] = generation[key]
    if generation.get("stop_sequences"):
        config["stopSequences"] = list(generation["stop_sequences"])
    mode, levels = thinking_support(model_id)
    if mode == "budget" and "thinking_budget" in generation:
        config["thinkingConfig"] = {"thinkingBudget": generation["thinking_budget"]}
    elif mode == "level" and generation.get("thinking_level") in levels:
        config["thinkingConfig"] = {"thinkingLevel": generation["thinking_level"].upper()}
    return config


def start_mock_server() -> None:
    """Start tools/mock_gemini_server.py on the loopback port if nothing is listening there yet."""
    address = ("127.0.0.1", MOCK_PORT)
    with socket.socket() as probe:
        probe.settimeout(0.3)
        if probe.connect_ex(address) == 0:
            return
    script = Path(__file__).resolve().parent / "tools" / "mock_gemini_server.py"
    if not script.is_file():
        raise ApiError("tools/mock_gemini_server.py was not found next to the app.")
    process = subprocess.Popen(
        [sys.executable, str(script), "--port", str(MOCK_PORT)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    atexit.register(process.terminate)
    for _ in range(30):
        time.sleep(0.1)
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(address) == 0:
                return
    raise ApiError("The mock server did not start. Run python3 tools/mock_gemini_server.py in another terminal.")


class GoogleApiClient:
    def __init__(self, api_key: str, generation: Optional[dict] = None, base_url: str = "") -> None:
        self.api_key = validate_api_key(api_key)
        self.generation = generation or {}
        self.base_url = base_url
        self.timeout = 90
        self.usage_input = 0  # prompt tokens billed by the requests this client made
        self.usage_output = 0  # response tokens, including thinking tokens
        self.last_prompt_tokens = 0  # prompt size of the most recent request, i.e. the current context
        self.model_version = ""  # the model Google says answered (resolves aliases like gemini-flash-latest)

    def _base(self) -> str:
        return self.base_url or API_BASE_URL

    def _auth_headers(self) -> dict:
        if not self.api_key:
            raise ApiError("No API key. Quit and relaunch with GEMINI_API_KEY set (or --api-key-file).")
        return {"User-Agent": USER_AGENT, "x-goog-api-key": self.api_key}

    @staticmethod
    def _token_counts(response: dict) -> Tuple[int, int]:
        usage = response.get("usageMetadata")
        if not isinstance(usage, dict):
            return 0, 0

        def count(name: str) -> int:
            value = usage.get(name)
            return value if isinstance(value, int) and value > 0 else 0

        return count("promptTokenCount"), count("candidatesTokenCount") + count("thoughtsTokenCount")

    def _note_response(self, response: dict) -> None:
        prompt_tokens, output_tokens = self._token_counts(response)
        if prompt_tokens:
            self.last_prompt_tokens = prompt_tokens
        self.usage_input += prompt_tokens
        self.usage_output += output_tokens
        version = response.get("modelVersion")
        if isinstance(version, str) and version.strip():
            self.model_version = version.strip()[:MAX_MODEL_ID_CHARS]

    def _request(self, url: str, body: Optional[dict] = None) -> dict:
        headers = self._auth_headers()
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            if len(data) > MAX_API_REQUEST_BYTES:
                raise ApiError("Conversation context exceeded the request safety limit. Use /clear and try again.")
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with GOOGLE_API_OPENER.open(request, timeout=self.timeout) as response:
                raw = self._read_limited(response, MAX_API_RESPONSE_BYTES, "Google response")
                parsed = json.loads(raw.decode("utf-8", errors="replace"))
                if not isinstance(parsed, dict):
                    raise ApiError("Google returned an unexpected JSON response.")
                self._note_response(parsed)
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

    def model_input_limit(self, model_id: str) -> Optional[int]:
        """The model's input context window in tokens, from models.get."""
        response = self._request(self._base() + "/v1beta/models/" + urllib.parse.quote(model_id, safe=""))
        limit = response.get("inputTokenLimit")
        return limit if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0 else None

    def list_models(self) -> List[str]:
        url = self._base() + "/v1beta/models"
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

    def _gemini_body(self, history: Sequence[Dict[str, str]], system_instruction: str, model_id: str = "") -> dict:
        contents = [
            {"role": item["role"], "parts": [{"text": item["content"]}]}
            for item in history
            if item["role"] in ("user", "model") and item["content"]
        ]
        body = {"contents": contents}
        config = build_generation_config(self.generation, model_id)
        if config:
            body["generationConfig"] = config
        if system_instruction.strip():
            body["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}
        return body

    def _generate_gemini(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        url = (self._base() + "/v1beta/models/{}:generateContent").format(
            urllib.parse.quote(model.identifier, safe="")
        )
        response = self._request(url, self._gemini_body(history, system_instruction, model.identifier))
        return self._text_from_parts(self._candidate_parts(response))

    @staticmethod
    def _candidate_parts(response: dict) -> list:
        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            feedback = response.get("promptFeedback", {})
            detail = feedback.get("blockReason", "no candidate returned") if isinstance(feedback, dict) else "no candidate returned"
            raise ApiError("The request produced no response: {}.".format(detail))
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        content = first.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        return parts if isinstance(parts, list) else []

    @staticmethod
    def _text_from_parts(parts: list) -> str:
        text = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
        ).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
        if len(text) > MAX_STREAM_TEXT_CHARS:
            raise ApiError("The response exceeded the safety limit.")
        return text

    def generate_with_tools(
        self,
        model: ModelSpec,
        history: Sequence[Dict[str, str]],
        system_instruction: str,
        run_command: Callable[[str], Tuple[dict, str]],
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Tuple[str, List[str]]:
        """Answer with the terminal tool enabled. run_command returns (tool result, log line).

        At most MAX_TERMINAL_RESULTS terminal results are fed back per turn. The results numbered
        in TERMINAL_WARNING_RESULTS (and the last one) end with a note about the accesses left."""
        url = (self._base() + "/v1beta/models/{}:generateContent").format(
            urllib.parse.quote(model.identifier, safe="")
        )
        body = self._gemini_body(history, system_instruction, model.identifier)
        body["tools"] = [TERMINAL_TOOL]
        log: List[str] = []
        used = 0
        while True:
            if used >= MAX_TERMINAL_RESULTS:
                # No accesses left: the model must answer in text.
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
            parts = self._candidate_parts(self._request(url, body))
            calls = [part for part in parts if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)]
            if not calls:
                return self._text_from_parts(parts), log
            if used >= MAX_TERMINAL_RESULTS and "toolConfig" in body:
                raise ApiError("The model kept asking for terminal access after the {} command limit.".format(MAX_TERMINAL_RESULTS))
            # Echo the model's parts back unchanged so signatures and call ids survive.
            body["contents"].append({"role": "model", "parts": parts})
            responses = []
            for part in calls:
                call = part["functionCall"]
                arguments = call.get("args")
                command = arguments.get("command") if isinstance(arguments, dict) else None
                if used >= MAX_TERMINAL_RESULTS:
                    result = {"error": "Terminal access limit reached for this turn. Answer with what you have."}
                elif should_stop and should_stop():
                    result = {"error": "The user stopped this turn."}
                elif call.get("name") != TERMINAL_TOOL_NAME or not isinstance(command, str) or not command.strip():
                    result = {"error": "Unknown tool or missing command argument."}
                else:
                    result, line = run_command(command.strip())
                    log.append(line)
                used += 1
                remaining = MAX_TERMINAL_RESULTS - used
                note = ""
                if used in TERMINAL_WARNING_RESULTS and remaining > 0:
                    note = "[You have {} more terminal access{} in the current turn.]".format(
                        remaining, "" if remaining == 1 else "es"
                    )
                elif used == MAX_TERMINAL_RESULTS:
                    note = "[That was your last terminal access in the current turn.]"
                if note:
                    key = "output" if "output" in result else "error"
                    result = dict(result)
                    result[key] = "{}\n{}".format(result.get(key, ""), note).lstrip("\n")
                response = {"name": call.get("name") or TERMINAL_TOOL_NAME, "response": result}
                if isinstance(call.get("id"), str):
                    response["id"] = call["id"]
                responses.append({"functionResponse": response})
            body["contents"].append({"role": "user", "parts": responses})
            if should_stop and should_stop():
                return "Stopped by you before the model finished.", log

    def _stream_gemini(
        self,
        model: ModelSpec,
        history: Sequence[Dict[str, str]],
        system_instruction: str,
        on_text: Callable[[str], None],
    ) -> str:
        if not self.api_key:
            raise ApiError("No API key. Quit and relaunch with GEMINI_API_KEY set (or --api-key-file).")
        url = (self._base() + "/v1beta/models/{}:streamGenerateContent?alt=sse").format(
            urllib.parse.quote(model.identifier, safe="")
        )
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-goog-api-key": self.api_key,
        }
        request_data = json.dumps(self._gemini_body(history, system_instruction, model.identifier)).encode("utf-8")
        if len(request_data) > MAX_API_REQUEST_BYTES:
            raise ApiError("Conversation context exceeded the request safety limit. Use /clear and try again.")
        request = urllib.request.Request(url, data=request_data, headers=headers, method="POST")
        received: List[str] = []
        received_chars = 0
        latest_usage = (0, 0)  # streamed events carry running totals, so keep only the latest
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
                    counts = self._token_counts(event)
                    if counts != (0, 0):
                        latest_usage = counts
                    version = event.get("modelVersion")
                    if isinstance(version, str) and version.strip():
                        self.model_version = version.strip()[:MAX_MODEL_ID_CHARS]
                    candidates = event.get("candidates", [])
                    if not isinstance(candidates, list) or not candidates:
                        continue
                    first = candidates[0] if isinstance(candidates[0], dict) else {}
                    content = first.get("content", {})
                    parts = content.get("parts", []) if isinstance(content, dict) else []
                    chunk = "".join(
                        part["text"]
                        for part in parts
                        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
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
        self.usage_input += latest_usage[0]
        self.usage_output += latest_usage[1]
        if latest_usage[0]:
            self.last_prompt_tokens = latest_usage[0]
        text = "".join(received).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
        return text

    def _generate_palm(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        # PaLM Chat used a different, now-retired API shape. Keeping this request
        # lets accounts with an available legacy endpoint attempt the original model.
        url = (self._base() + "/v1beta3/models/{}:generateMessage").format(
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


LMSTUDIO_PREFIX = "lmstudio:"
LMSTUDIO_GROUP = "LM Studio (local)"
LMSTUDIO_OFFLINE_ENTRY = ModelSpec(
    "__lmstudio_offline__",
    "LM Studio is not running",
    LMSTUDIO_GROUP,
    "lmstudio",
    "Start LM Studio's local server (Developer tab, Start Server) or pass --lmstudio-url",
)
REASONING_TAGS = ("think", "thinking", "reasoning", "thought")
REASONING_BLOCK = re.compile(r"<({})>.*?</\1>".format("|".join(REASONING_TAGS)), re.DOTALL)
REASONING_OPENERS = tuple("<{}>".format(tag) for tag in REASONING_TAGS) + ("<|channel|>",)
REASONING_CLOSERS = tuple("</{}>".format(tag) for tag in REASONING_TAGS)
HARMONY_FINAL = "<|channel|>final<|message|>"
HARMONY_TOKEN = re.compile(r"<\|(?:end|return|start|call)\|>.*", re.DOTALL)
# Errors that suggest the model's chat template rejected the system role or the message order.
ROLE_ERROR = re.compile(r"\brole|system|alternat|template|jinja", re.IGNORECASE)


def lmstudio_model_spec(model_id: str, note: str = "") -> ModelSpec:
    """A picker entry for one model LM Studio reported (model_id is LM Studio's own ID)."""
    return ModelSpec(
        LMSTUDIO_PREFIX + model_id,
        model_id,
        LMSTUDIO_GROUP,
        "lmstudio",
        note or "Local model served by LM Studio; free to run",
    )


def display_model_name(identifier: str) -> str:
    """How a model ID is shown in the header, chat, and stats."""
    if identifier.startswith(LMSTUDIO_PREFIX):
        return identifier[len(LMSTUDIO_PREFIX) :] + " (LM Studio)"
    return identifier


def visible_reasoning_free(raw: str, final: bool = False) -> str:
    """Remove reasoning from local-model output so only the answer remains.

    Handles <think>/<thinking>/<reasoning>/<thought> blocks, a closing tag with no opener (some chat
    templates put the opener in the prompt), and gpt-oss "harmony" channels. While streaming
    (final=False) an unfinished block and a trailing partial tag are held back."""
    text = REASONING_BLOCK.sub("", raw)
    if "<|channel|>" in text:
        if HARMONY_FINAL in text:
            text = HARMONY_TOKEN.sub("", text.split(HARMONY_FINAL)[-1])
        else:
            text = text[: text.find("<|channel|>")]
    first_open = min((text.find(opener) for opener in REASONING_OPENERS if opener in text), default=-1)
    if first_open != -1:
        text = text[:first_open]
    last_close = max((text.rfind(closer) + len(closer) for closer in REASONING_CLOSERS if closer in text), default=-1)
    if last_close != -1:
        text = text[last_close:]
    if not final:
        for size in range(max(map(len, REASONING_OPENERS)) - 1, 0, -1):
            if len(text) >= size and any(opener.startswith(text[-size:]) for opener in REASONING_OPENERS):
                text = text[:-size]
                break
    return text


def validate_lmstudio_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ApiError("The LM Studio URL must look like http://127.0.0.1:1234")
    if parsed.path.rstrip("/") not in ("", "/v1"):
        raise ApiError("The LM Studio URL should be just the server address, like http://127.0.0.1:1234")
    return "{}://{}".format(parsed.scheme, parsed.netloc)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters each) for servers that do not report usage."""
    return max(1, len(text) // 4) if text else 0


class LMStudioClient(GoogleApiClient):
    """LM Studio's local OpenAI-compatible server (chat completions, models, streaming).

    The documented /v1/chat/completions parameters are used. Two quirks are handled by retrying: models whose
    chat template rejects a system message (it is merged into the first user message), and servers that
    reject stream_options (usage is then estimated). Both are remembered in merge_system / include_usage."""

    def __init__(
        self,
        base_url: str,
        generation: Optional[dict] = None,
        timeout: int = 600,
        merge_system: bool = False,
        include_usage: bool = True,
    ) -> None:
        super().__init__("", generation, base_url)
        self.timeout = timeout
        self.merge_system = merge_system
        self.include_usage = include_usage
        self.lmstudio_key = os.environ.get("LMSTUDIO_API_KEY", "").strip()
        self.finish_reason = ""
        self.model_info: Dict[str, dict] = {}  # model id -> {"loaded": bool, "context": Optional[int]} when known
        self._reported_usage = False
        self._native_checked = False

    def _auth_headers(self) -> dict:
        headers = {"User-Agent": USER_AGENT}
        if self.lmstudio_key:
            headers["Authorization"] = "Bearer " + self.lmstudio_key
        return headers

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
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            # LM Studio's error body is not documented: accept {"error": "text"} and {"error": {"message": ...}}.
            details = parsed.get("error", parsed.get("message"))
            if isinstance(details, str) and details.strip():
                message = details.strip()
            elif isinstance(details, dict) and isinstance(details.get("message"), str):
                message = details["message"]
        if not message:
            message = str(error.reason or "request failed")
        if self.lmstudio_key:
            message = message.replace(self.lmstudio_key, "[redacted]")
        return message + (" [truncated]" if truncated else "")

    @staticmethod
    def _bare_id(identifier: str) -> str:
        return identifier[len(LMSTUDIO_PREFIX) :] if identifier.startswith(LMSTUDIO_PREFIX) else identifier

    # ---- model discovery -------------------------------------------------------------------------------------

    def _native_models(self) -> Dict[str, dict]:
        """Details from LM Studio's native API (v1, then v0): type, loaded state, and context lengths."""
        if self._native_checked:
            return self.model_info
        self._native_checked = True
        info: Dict[str, dict] = {}
        try:
            response = self._request(self._base() + "/api/v1/models")
            for item in response.get("models", []):
                if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                    continue
                instances = [i for i in item.get("loaded_instances", []) if isinstance(i, dict)]
                loaded_context = next(
                    (i["config"]["context_length"] for i in instances
                     if isinstance(i.get("config"), dict) and isinstance(i["config"].get("context_length"), int)),
                    None,
                )
                entry = {
                    "type": str(item.get("type", "")),
                    "loaded": bool(instances),
                    "context": loaded_context or item.get("max_context_length"),
                }
                info[item["key"]] = entry
                for instance in instances:
                    if isinstance(instance.get("id"), str):
                        info[instance["id"]] = entry
        except (ApiError, TypeError, KeyError):
            info = {}
        if not info:
            try:
                response = self._request(self._base() + "/api/v0/models")
                for item in response.get("data", []):
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        info[item["id"]] = {
                            "type": str(item.get("type", "")),
                            "loaded": item.get("state") == "loaded",
                            "context": item.get("loaded_context_length") or item.get("max_context_length"),
                        }
            except (ApiError, TypeError):
                info = {}
        for entry in info.values():
            if not isinstance(entry["context"], int) or isinstance(entry["context"], bool) or entry["context"] <= 0:
                entry["context"] = None
        self.model_info = info
        return info

    def list_models(self) -> List[str]:
        response = self._request(self._base() + "/v1/models")
        data = response.get("data", [])
        if not isinstance(data, list):
            raise ApiError("LM Studio returned an unexpected model list.")
        ids = [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]
        native = self._native_models()
        usable = []
        for name in ids:
            kind = native.get(name, {}).get("type", "")
            if kind.startswith("embed") or (not kind and "embed" in name.lower()):
                continue  # embedding models cannot chat
            if 0 < len(name) <= MAX_MODEL_ID_CHARS - len(LMSTUDIO_PREFIX):
                usable.append(name)
        return usable

    def model_input_limit(self, model_id: str) -> Optional[int]:
        bare = self._bare_id(model_id)
        entry = self._native_models().get(bare)
        if entry and entry["context"]:
            return entry["context"]
        try:
            response = self._request(self._base() + "/api/v0/models/" + urllib.parse.quote(bare, safe="/"))
        except ApiError:
            return None
        for field in ("loaded_context_length", "max_context_length"):
            value = response.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    # ---- chat ------------------------------------------------------------------------------------------------

    def _chat_body(
        self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str, stream: bool,
        merge_system: bool, include_usage: bool,
    ) -> dict:
        system_text = system_instruction.strip()
        messages: List[Dict[str, str]] = []
        for item in history:
            if item["role"] not in ("user", "model") or not item["content"]:
                continue
            role = "user" if item["role"] == "user" else "assistant"
            if messages and messages[-1]["role"] == role:
                # Several chat templates insist that roles alternate, so merge back-to-back messages.
                messages[-1]["content"] += "\n\n" + item["content"]
            else:
                messages.append({"role": role, "content": item["content"]})
        if system_text:
            if merge_system:
                if messages and messages[0]["role"] == "user":
                    messages[0]["content"] = system_text + "\n\n" + messages[0]["content"]
                else:
                    messages.insert(0, {"role": "user", "content": system_text})
            else:
                messages.insert(0, {"role": "system", "content": system_text})
        body: dict = {"model": self._bare_id(model.identifier), "messages": messages, "stream": stream}
        names = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_output_tokens": "max_tokens",
            "seed": "seed",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }
        for key, api_name in names.items():
            if key in self.generation:
                body[api_name] = self.generation[key]
        if self.generation.get("stop_sequences"):
            body["stop"] = list(self.generation["stop_sequences"])
        if stream and include_usage:
            body["stream_options"] = {"include_usage": True}
        return body

    def _with_fallbacks(self, system_instruction: str, run: Callable[[bool, bool], object]) -> object:
        merge, usage_option = self.merge_system, self.include_usage
        for _ in range(3):
            try:
                result = run(merge, usage_option)
            except ApiError as error:
                text = str(error)
                if not text.startswith(("HTTP 400", "HTTP 422", "HTTP 500")):
                    raise
                if usage_option and "stream_options" in text.lower():
                    usage_option = False
                    continue
                if not merge and system_instruction.strip() and ROLE_ERROR.search(text):
                    merge = True
                    continue
                raise
            self.merge_system, self.include_usage = merge, usage_option
            return result
        raise ApiError("LM Studio rejected the request.")

    def _record_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0:
            self.usage_input += prompt_tokens
            self.last_prompt_tokens = prompt_tokens
            self._reported_usage = True
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens > 0:
            self.usage_output += completion_tokens
            self._reported_usage = True

    def _estimate_usage(self, body: dict, answer: str) -> None:
        """Some servers send no token counts, so fall back to about 4 characters per token."""
        if self._reported_usage:
            return
        prompt_tokens = estimate_tokens("".join(message["content"] for message in body["messages"]))
        self.usage_input += prompt_tokens
        self.usage_output += estimate_tokens(answer)
        self.last_prompt_tokens = prompt_tokens

    def generate(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        self.model_version = model.identifier

        def run(merge: bool, usage_option: bool) -> str:
            body = self._chat_body(model, history, system_instruction, False, merge, usage_option)
            response = self._request(self._base() + "/v1/chat/completions", body)
            self._reported_usage = False
            self._record_usage(response.get("usage"))
            choices = response.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):  # content parts
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str):
                raise ApiError("LM Studio returned no text.")
            text = visible_reasoning_free(content, final=True).strip()
            if not text:
                raise ApiError("The model returned no text (it may have spent its whole token budget thinking).")
            if len(text) > MAX_STREAM_TEXT_CHARS:
                raise ApiError("The response exceeded the safety limit.")
            self.finish_reason = str(choice.get("finish_reason") or "")
            self._estimate_usage(body, text)
            return text

        return self._with_fallbacks(system_instruction, run)  # type: ignore[return-value]

    def generate_stream(
        self,
        model: ModelSpec,
        history: Sequence[Dict[str, str]],
        system_instruction: str,
        on_text: Callable[[str], None],
    ) -> str:
        self.model_version = model.identifier

        def run(merge: bool, usage_option: bool) -> str:
            body = self._chat_body(model, history, system_instruction, True, merge, usage_option)
            data = json.dumps(body).encode("utf-8")
            if len(data) > MAX_API_REQUEST_BYTES:
                raise ApiError("Conversation context exceeded the request safety limit. Use /clear and try again.")
            headers = self._auth_headers()
            headers.update({"Content-Type": "application/json", "Accept": "text/event-stream"})
            request = urllib.request.Request(
                self._base() + "/v1/chat/completions", data=data, headers=headers, method="POST"
            )
            raw_text = ""
            emitted = ""
            latest_usage: object = None
            finish = ""
            try:
                with GOOGLE_API_OPENER.open(request, timeout=self.timeout) as response:
                    while True:
                        raw_line = response.readline(MAX_STREAM_EVENT_BYTES + 1)
                        if not raw_line:
                            break
                        if len(raw_line) > MAX_STREAM_EVENT_BYTES:
                            raise ApiError("A streaming event exceeded the safety limit.")
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError as error:
                            raise ApiError("LM Studio returned an invalid streaming event.") from error
                        if not isinstance(event, dict):
                            raise ApiError("LM Studio returned an unexpected streaming event.")
                        if "error" in event:
                            details = event["error"]
                            message = details.get("message") if isinstance(details, dict) else details
                            raise ApiError("LM Studio stopped the reply: {}".format(message))
                        if event.get("usage"):
                            latest_usage = event["usage"]  # running totals, so keep only the latest
                        choices = event.get("choices")
                        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                        if choice.get("finish_reason"):
                            finish = str(choice["finish_reason"])
                        delta = choice.get("delta")
                        chunk = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(chunk, str) and chunk:
                            raw_text += chunk
                            if len(raw_text) > MAX_STREAM_TEXT_CHARS:
                                raise ApiError("The streamed response exceeded the safety limit.")
                            visible = visible_reasoning_free(raw_text)
                            if visible.startswith(emitted):
                                if len(visible) > len(emitted):
                                    on_text(visible[len(emitted) :])
                                    emitted = visible
                            else:
                                emitted = visible  # a stray closing tag removed text already shown; the final reply replaces it
            except urllib.error.HTTPError as error:
                raise ApiError("HTTP {}: {}".format(error.code, self._http_error_message(error))) from error
            except urllib.error.URLError as error:
                raise ApiError("Network error: {}".format(error.reason)) from error
            except TimeoutError as error:
                raise ApiError("Request timed out. Check LM Studio and try again.") from error
            except OSError as error:
                raise ApiError("Network error: {}".format(error)) from error
            final = visible_reasoning_free(raw_text, final=True)
            if final.startswith(emitted) and len(final) > len(emitted):
                on_text(final[len(emitted) :])
            text = final.strip()
            if not text:
                raise ApiError("The model returned no text (it may have spent its whole token budget thinking).")
            self._reported_usage = False
            self._record_usage(latest_usage)
            self.finish_reason = finish
            self._estimate_usage(body, text)
            return text

        return self._with_fallbacks(system_instruction, run)  # type: ignore[return-value]

    def generate_with_tools(self, *_args, **_kwargs):
        raise ApiError("Terminal access is not available for LM Studio models.")


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
        raw_usage = data.get("usage")
        usage = {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False}
        if isinstance(raw_usage, dict):
            for name in ("input", "output", "context"):
                value = raw_usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10**12:
                    usage[name] = value
            cost = raw_usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and 0 <= cost < 10**9:
                usage["cost"] = float(cost)
            usage["unpriced"] = raw_usage.get("unpriced") is True
        settings = data.get("settings", {})
        streaming = settings.get("streaming", True) if isinstance(settings, dict) else True
        terminal_tools = settings.get("terminal_tools", False) if isinstance(settings, dict) else False
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
            "usage": usage,
            "lifetime": normalize_lifetime(data.get("lifetime")),
            "settings": {
                "streaming": streaming if isinstance(streaming, bool) else True,
                "generation": normalize_generation(settings.get("generation") if isinstance(settings, dict) else None),
                "terminal_tools": terminal_tools if isinstance(terminal_tools, bool) else False,
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
        self.data.setdefault("usage", {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False})
        self.data.setdefault("lifetime", normalize_lifetime(None))
        self.data["settings"].setdefault("generation", {})
        self.data["settings"].setdefault("streaming", True)
        self.data["settings"].setdefault("terminal_tools", False)
        self.data["settings"].setdefault("system_instruction_configured", False)
        self.data["settings"].setdefault("theme", DEFAULT_THEME_ID)
        for item in self.data["history"]:
            if item["role"] == "model":
                item["content"] = self.plain_terminal_response(item["content"])
        self.status = "Ready" if self.api_key else "No API key. Quit and relaunch with GEMINI_API_KEY set or --api-key-file. The mock and LM Studio models work without one."
        self.input_rows = 1
        self.available_models: Optional[set] = None
        self.running = True
        self.restart_requested = False
        self.colors: Dict[str, int] = {}
        self.scroll_offset = 0
        self.prompt_history = [
            item["content"] for item in self.data["history"] if item["role"] == "user"
        ]
        self.last_failed_prompt: Optional[str] = None
        self.context_limits: Dict[str, int] = {}
        self.lmstudio_url = LMSTUDIO_URL
        self.lmstudio_quirks: Dict[str, dict] = {}  # per model: workarounds that LM Studio needed
        self.command_permissions = "ask"  # ask, safe, or all; resets on restart
        self.stop_requested = False

    @property
    def model(self) -> ModelSpec:
        identifier = self.data["model"]
        if identifier in MODEL_BY_ID:
            return MODEL_BY_ID[identifier]
        if identifier.startswith(LMSTUDIO_PREFIX):
            return lmstudio_model_spec(identifier[len(LMSTUDIO_PREFIX) :])
        return ModelSpec(identifier, identifier, "Custom Gemini Model", "gemini", "Custom model ID")

    @property
    def streaming_enabled(self) -> bool:
        return bool(self.data["settings"].get("streaming", True))

    @property
    def terminal_tools_setting(self) -> bool:
        return bool(self.data["settings"].get("terminal_tools", False))

    @property
    def terminal_tools_active(self) -> bool:
        return (
            self.terminal_tools_setting
            and self.model.protocol == "gemini"
            and model_supports_terminal_tools(self.model.identifier)
        )

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

    @staticmethod
    def split_terminal_lines(content: str) -> Tuple[str, str]:
        """Separate the leading "[terminal] ..." log lines of a reply from the reply text itself."""
        lines = content.split("\n")
        count = 0
        while count < len(lines) and lines[count].startswith("[terminal]"):
            count += 1
        return "\n".join(lines[:count]), "\n".join(lines[count:]).lstrip("\n")

    def render_history(self, width: int, compact: bool = False) -> List[Tuple[str, int]]:
        lines: List[Tuple[str, int]] = []
        content_width = max(width - (5 if compact else 8), 8)
        indent = "   " if compact else "       "
        for item in self.data["history"]:
            is_user = item["role"] == "user"
            label = "YOU" if is_user else display_model_name(item.get("model_id", "UNKNOWN MODEL")).upper()
            role_style = self.style("user" if is_user else "assistant", curses.A_BOLD)
            terminal_text, reply_text = ("", item["content"]) if is_user else self.split_terminal_lines(item["content"])
            # Terminal results are dimmed so they read differently from the model's own words.
            body: List[Tuple[str, int]] = []
            if terminal_text:
                body.extend((line, self.style("muted")) for line in self.wrap_content(terminal_text, content_width))
                if reply_text:
                    body.append(("", 0))
            if reply_text or not terminal_text:
                body.extend((line, self.style("text")) for line in self.wrap_content(reply_text, content_width))
            if compact:
                compact_label = "You" if is_user else display_model_name(item.get("model_id", "Unknown model"))
                lines.append((compact_label + ": " + body[0][0], role_style if is_user else body[0][1]))
                lines.extend((indent + line, style) for line, style in body[1:])
            elif is_user:
                lines.append(("  YOU  " + body[0][0], role_style))
                lines.extend((indent + line, style) for line, style in body[1:])
            else:
                lines.append(("  " + label, role_style))
                lines.extend((indent + line, style) for line, style in body)
            lines.append(("", 0))
        return lines

    def transcript_lines(self) -> List[str]:
        lines: List[str] = []
        for item in self.data["history"]:
            if item["role"] == "user":
                label = "YOU"
            else:
                label = display_model_name(item.get("model_id", "UNKNOWN MODEL")).upper()
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
        curses.nonl()  # Enter arrives as "\r" and Ctrl+J as "\n", so Shift+Enter can mean a new line
        while self.running:
            self.input_rows = 1
            self.draw_chat(screen)
            prompt = self.read_line(screen, "You > ")
            self.input_rows = 1
            if prompt is None:
                continue
            self.handle_input(screen, prompt)

    @staticmethod
    def format_tokens(value: int) -> str:
        return "{:,}".format(value)

    def context_window(self) -> Optional[int]:
        identifier = self.model.identifier
        if identifier in self.context_limits:
            return self.context_limits[identifier]
        return DEFAULT_CONTEXT_WINDOW if identifier.startswith("gemini-") else None

    def learn_context_window(self, client: "GoogleApiClient", model: ModelSpec) -> None:
        if model.protocol not in ("gemini", "lmstudio"):
            return
        # An LM Studio model can be reloaded with a different context length, so ask again each time.
        if model.protocol == "gemini" and model.identifier in self.context_limits:
            return
        try:
            client.timeout = 5
            limit = client.model_input_limit(model.identifier)
        except ApiError:
            return
        if limit:
            self.context_limits[model.identifier] = limit

    @staticmethod
    def format_cost(usage: dict) -> str:
        """Estimated spend; a trailing + means some usage came from a model with no known price."""
        cost = usage.get("cost", 0.0)
        if cost == 0 and usage.get("unpriced"):
            return "cost unknown"
        text = "~${:.4f}".format(cost) if cost < 0.1 else "~${:.2f}".format(cost)
        return text + ("+" if usage.get("unpriced") else "")

    def header_parts(self) -> Tuple[str, str]:
        """The one-line header: (model id, the rest)."""
        turns = sum(1 for item in self.data["history"] if item["role"] == "user")
        usage = self.data["usage"]
        window = self.context_window()
        context = self.format_tokens(usage.get("context", 0)) + ("/" + self.format_tokens(window) if window else "")
        rest = "   {} turn{}   {} in / {} out tokens   context {}   {}".format(
            turns,
            "" if turns == 1 else "s",
            self.format_tokens(usage["input"]),
            self.format_tokens(usage["output"]),
            context,
            self.format_cost(usage),
        )
        return display_model_name(self.model.identifier), rest

    def draw_header(self, screen: "curses._CursesWindow", width: int, compact: bool = False) -> None:
        model_id, rest = self.header_parts()
        if compact:
            self.add(screen, 0, 0, " " * max(width - 1, 0), curses.A_REVERSE)
            self.add(screen, 0, 0, (" " + model_id + rest)[: max(width - 1, 0)], curses.A_REVERSE)
            return
        self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("base"))
        self.add(screen, 0, 1, model_id[: max(width - 2, 0)], self.style("accent", curses.A_BOLD))
        self.add(screen, 0, 1 + len(model_id), rest[: max(width - len(model_id) - 2, 0)], self.style("muted"))

    def draw_chat(self, screen: "curses._CursesWindow") -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 10 or width < 42:
            self.draw_compact_chat(screen)
            return

        self.draw_header(screen, width)
        self.add(screen, 1, 0, "-" * max(width - 1, 0), self.style("border"))

        rows = self.input_rows
        input_top = height - 1 - rows
        lower_border = input_top - 1
        upper_border = lower_border - 3
        history_height = max(upper_border - 2, 1)
        lines = self.render_history(width)

        if not lines:
            empty_row = max(4, (upper_border - 2) // 2)
            self.add(screen, empty_row, 4, "Start a conversation", self.style("text", curses.A_BOLD))
            self.add(screen, empty_row + 2, 4, "Tab opens the menu. / opens commands.", self.style("muted"))
        visible, self.scroll_offset, _ = self.history_window(lines, history_height, self.scroll_offset)
        for row, (line, style) in enumerate(visible, start=2):
            self.add(screen, row, 0, line[: max(width - 1, 0)], style)
        self.add(screen, upper_border, 0, "-" * max(width - 1, 0), self.style("border"))
        if self.status.endswith(STOP_HINT):
            main = self.status[: -len(STOP_HINT)].rstrip()
            room = max(width - len(STOP_HINT) - 7, 10)
            if len(main) > room:
                main = main[: room - 1] + "..."
            self.add(screen, upper_border + 1, 1, main, self.style("text"))
            self.add(screen, upper_border + 1, 4 + len(main), STOP_HINT, self.style("muted"))
        else:
            for index, line in enumerate(self.status_lines(width), start=upper_border + 1):
                self.add(screen, index, 1, line, self.status_style())
        self.add(screen, lower_border, 0, "-" * max(width - 1, 0), self.style("border"))
        self.add(screen, input_top, 1, ">", self.style("accent", curses.A_BOLD))
        if self.scroll_offset:
            footer = "{} lines above latest".format(self.scroll_offset)
        else:
            footer = "Tab Menu   / Commands"
        self.add(screen, height - 1, 1, footer[: width - 2], self.style("footer"))
        screen.refresh()

    def draw_compact_chat(self, screen: "curses._CursesWindow") -> None:
        height, width = screen.getmaxyx()
        self.draw_header(screen, width, compact=True)
        history_height = max(height - 4, 1)
        lines = self.render_history(width, compact=True)
        visible, self.scroll_offset, _ = self.history_window(lines, history_height, self.scroll_offset)
        for row, (line, style) in enumerate(visible, start=1):
            self.add(screen, row, 0, line[: max(width - 1, 0)], style)
        self.add(screen, height - 3, 0, "-" * max(width - 1, 0))
        self.add(screen, height - 2, 0, self.status[: max(width - 1, 0)], curses.A_DIM)
        self.add(screen, height - 1, 0, "You > ")
        screen.refresh()

    def read_escape_sequence(self, screen: "curses._CursesWindow") -> str:
        """After an Esc key, collect whatever else the terminal sent with it (without waiting)."""
        collected: List[str] = []
        screen.nodelay(True)
        try:
            while len(collected) < 16:
                try:
                    key = screen.get_wch()
                except curses.error:
                    break
                if not isinstance(key, str):
                    break
                collected.append(key)
        finally:
            screen.nodelay(False)
        return "".join(collected)

    @staticmethod
    def is_newline_sequence(sequence: str) -> bool:
        """The escape codes terminals send for Shift+Enter (Esc has already been read)."""
        return sequence.startswith(("[13;2u", "[27;2;13~", "[13;2~"))

    def read_line(self, screen: "curses._CursesWindow", label: str) -> Optional[str]:
        buffer: List[str] = []
        cursor = 0
        scroll = 0
        history_index = len(self.prompt_history)
        draft: List[str] = []
        while True:
            height, width = screen.getmaxyx()
            compact = height < 10 or width < 42
            text_column = len(label) if compact else 3
            max_rows = 1 if compact else max(1, min(MAX_INPUT_ROWS, height - 10))
            wrap = max(width - text_column - 1, 1)
            text = "".join(buffer)
            rows = self.editor_rows(text, wrap)
            row, column = self.editor_cursor(rows, cursor)
            shown_rows = min(len(rows), max_rows)
            if not compact and shown_rows != self.input_rows:
                self.input_rows = shown_rows
                self.draw_chat(screen)
            scroll = min(max(scroll, row - shown_rows + 1), row, max(len(rows) - shown_rows, 0))
            input_top = height - 1 if compact else height - 1 - self.input_rows
            for offset in range(shown_rows):
                self.add(screen, input_top + offset, 0, " " * max(width - 1, 0))
                index = scroll + offset
                if index < len(rows):
                    start, end = rows[index]
                    self.add(screen, input_top + offset, text_column, text[start:end], self.style("input"))
            if compact:
                self.add(screen, input_top, 0, label, self.style("input"))
            else:
                self.add(screen, input_top, 1, ">", self.style("accent", curses.A_BOLD))
            try:
                screen.move(input_top + row - scroll, min(text_column + column, width - 1))
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
            if key == "\r":
                return "".join(buffer).strip()
            if key == "\x1b":
                if self.is_newline_sequence(self.read_escape_sequence(screen)) and len(buffer) < MAX_INPUT_CHARS:
                    buffer.insert(cursor, "\n")
                    cursor += 1
                continue
            if key == "\n":
                # Ctrl+J is deliberately not a shortcut; only Shift+Enter adds a new line.
                continue
            if key == "\x03":
                self.running = False
                return None
            if key == curses.KEY_F1:
                self.show_help(screen)
                return None
            if key in ("\t", "\x0b"):
                self.open_action_menu(screen)
                return None
            if key in (curses.KEY_F2, "\x0c"):
                self.select_model(screen)
                return None
            if key == curses.KEY_F3:
                self.edit_system_instruction(screen)
                return None
            if key == curses.KEY_F5:
                self.check_availability(screen)
                return None
            if key in (curses.KEY_F6, "\x14"):
                self.show_transcript(screen)
                return None
            if key == curses.KEY_UP:
                if row > 0:
                    start, end = rows[row - 1]
                    cursor = start + min(column, end - start)
                else:
                    self.scroll_offset += 1
                    self.draw_chat(screen)
                continue
            if key == curses.KEY_DOWN:
                if row < len(rows) - 1:
                    start, end = rows[row + 1]
                    cursor = start + min(column, end - start)
                else:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                    self.draw_chat(screen)
                continue
            if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                if cursor:
                    del buffer[cursor - 1]
                    cursor -= 1
                continue
            if key == curses.KEY_DC:
                if cursor < len(buffer):
                    del buffer[cursor]
                continue
            if key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
                continue
            if key == curses.KEY_RIGHT:
                cursor = min(len(buffer), cursor + 1)
                continue
            if key == "\x10" and self.prompt_history:
                if history_index == len(self.prompt_history):
                    draft = list(buffer)
                history_index = max(0, history_index - 1)
                buffer = list(self.prompt_history[history_index])
                cursor = len(buffer)
                continue
            if key == "\x0f" and history_index < len(self.prompt_history):
                history_index += 1
                buffer = list(draft if history_index == len(self.prompt_history) else self.prompt_history[history_index])
                cursor = len(buffer)
                continue
            if key in (curses.KEY_HOME, "\x01"):
                cursor = rows[row][0]
                continue
            if key in (curses.KEY_END, "\x05"):
                cursor = rows[row][1]
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
                    self.draw_chat(screen)

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
        use_tools = self.terminal_tools_active
        self.stop_requested = False
        generation = self.data["settings"].get("generation", {})
        if selected_model.protocol == "lmstudio":
            quirks = self.lmstudio_quirks.get(selected_model.identifier, {})
            client = LMStudioClient(
                self.lmstudio_url,
                generation,
                merge_system=quirks.get("merge_system", False),
                include_usage=quirks.get("include_usage", True),
            )
        elif selected_model.local_mock:
            client = GoogleApiClient(MOCK_API_KEY, generation, MOCK_BASE_URL)
        else:
            client = GoogleApiClient(self.api_key, generation)
        if use_tools:
            self.status = "Waiting for {}...   {}".format(display_model_name(selected_model.identifier), STOP_HINT)
        elif self.streaming_enabled:
            self.status = "Streaming {}...".format(display_model_name(selected_model.identifier))
        else:
            self.status = "Waiting for {}...".format(display_model_name(selected_model.identifier))
        try:
            if selected_model.local_mock:
                start_mock_server()
            if use_tools:
                self.draw_chat(screen)
                answer, tool_log = client.generate_with_tools(
                    selected_model,
                    self.data["history"],
                    self.terminal_system_instruction(),
                    lambda command: self.run_approved_command(screen, command),
                    lambda: self.poll_stop_request(screen),
                )
                content = self.plain_terminal_response(answer)
                if tool_log:
                    content = "\n".join(tool_log) + "\n\n" + content
                self.data["history"].append(
                    {"role": "model", "content": content, "model_id": client.model_version or selected_model.identifier}
                )
            elif selected_model.protocol in ("gemini", "lmstudio") and self.streaming_enabled:
                reply = {"role": "model", "content": "", "model_id": selected_model.identifier}
                self.data["history"].append(reply)
                self.draw_chat(screen)

                def append_chunk(chunk: str) -> None:
                    fragments = self.display_fragments(self.plain_terminal_response(chunk))
                    for fragment in fragments:
                        reply["content"] += fragment
                        self.status = "Streaming {}... {} chars".format(display_model_name(selected_model.identifier), len(reply["content"]))
                        self.draw_chat(screen)
                        if len(fragments) > 1:
                            try:
                                curses.napms(STREAM_RENDER_DELAY_MS)
                            except curses.error:
                                pass

                answer = client.generate_stream(
                    selected_model, self.data["history"][:-1], self.data.get("system_instruction", ""), append_chunk
                )
                reply["content"] = self.plain_terminal_response(answer)
                reply["model_id"] = client.model_version or selected_model.identifier
            else:
                self.draw_chat(screen)
                answer = client.generate(
                    selected_model, self.data["history"], self.data.get("system_instruction", "")
                )
                self.data["history"].append(
                    {
                        "role": "model",
                        "content": self.plain_terminal_response(answer),
                        "model_id": client.model_version or selected_model.identifier,
                    }
                )
            self.status = "Response complete."
            if isinstance(client, LMStudioClient):
                self.lmstudio_quirks[selected_model.identifier] = {
                    "merge_system": client.merge_system,
                    "include_usage": client.include_usage,
                }
                if client.finish_reason == "length":
                    self.status = "Response stopped at the model's token limit. Raise Max output tokens in /thinking or LM Studio's context."
            self.last_failed_prompt = None
            self.learn_context_window(client, selected_model)
        except ApiError as error:
            if self.data["history"] and self.data["history"][-1].get("role") == "model":
                self.data["history"].pop()
            if self.data["history"] and self.data["history"][-1].get("role") == "user":
                self.data["history"].pop()
            self.last_failed_prompt = text
            self.status = "{} Use /retry or Up to recall the prompt.".format(
                self.friendly_error(error, selected_model)
            )
        finally:
            self.add_usage(client, selected_model.identifier)
        self.store.save(self.data)

    def add_usage(self, client: "GoogleApiClient", model_id: str = "") -> None:
        if not (client.usage_input or client.usage_output):
            return
        name = client.model_version or model_id
        cost = estimate_cost(name, client.usage_input, client.usage_output)
        usage = self.data["usage"]
        if client.last_prompt_tokens:
            usage["context"] = client.last_prompt_tokens
        lifetime = self.data["lifetime"]
        per_model = lifetime["models"].setdefault(name[:MAX_MODEL_ID_CHARS] or "unknown", empty_usage_totals())
        # The conversation totals reset on /clear; the lifetime totals never do.
        for totals in (usage, lifetime, per_model):
            totals["input"] += client.usage_input
            totals["output"] += client.usage_output
            if cost is None:
                totals["unpriced"] = True
            else:
                totals["cost"] += cost

    def poll_stop_request(self, screen: "curses._CursesWindow", running: str = "") -> bool:
        """Non-blocking check for Esc while a terminal turn is in progress. Other keys are discarded."""
        if self.stop_requested:
            return True
        screen.nodelay(True)
        try:
            while True:
                try:
                    key = screen.get_wch()
                except curses.error:
                    break
                if key == "\x1b":
                    self.stop_requested = True
        finally:
            screen.nodelay(False)
        if self.stop_requested:
            self.status = "Stopping..."
            self.draw_chat(screen)
        return self.stop_requested

    def terminal_system_instruction(self) -> str:
        note = (
            "You can run shell commands on the user's computer with the {} tool. The user may deny a command or "
            "stop the turn; if a command is denied, do not retry it and ask how to proceed. Platform: {}. Working "
            "directory: {}. Run one purposeful command at a time and inspect with read-only commands before changing "
            "anything. Use non-interactive commands with small output (head, grep, or counts instead of dumping whole "
            "files); if output ends with a truncation note, narrow the command instead of guessing. Do not use sudo, "
            "delete files, or touch anything outside the working directory unless the user asked for it. You get at "
            "most {limit} terminal results per turn. When you finish, summarize what happened in plain text and do not "
            "claim success unless the output shows it."
        ).format(TERMINAL_TOOL_NAME, sys.platform, Path.cwd(), limit=MAX_TERMINAL_RESULTS)
        base = self.data.get("system_instruction", "").strip()
        return (base + "\n\n" + note) if base else note

    def run_approved_command(self, screen: "curses._CursesWindow", command: str) -> Tuple[dict, str]:
        shown = " ".join(command.split())
        if len(shown) > 120:
            shown = shown[:117] + "..."
        review = classify_command(command)
        # "Allow all" never covers risky commands; those ask every time.
        skip_prompt = self.command_permissions == "all" or (
            self.command_permissions == "safe" and not review.needs_confirmation
        )
        if not skip_prompt:
            decision = self.confirm_command(screen, command, review)
            if decision == "all" and not review.needs_confirmation:
                self.command_permissions = "safe"
            elif decision not in ("yes", "all"):
                return (
                    {"error": "The user denied this command. Do not retry it; ask the user how to proceed."},
                    "[terminal] denied [{}]: {}".format(review.primary, shown),
                )
        self.status = "Running: {}   {}".format(shown, STOP_HINT)
        self.draw_chat(screen)
        result = run_shell_command(command, should_stop=lambda: self.poll_stop_request(screen))
        if "error" in result:
            return result, "[terminal] failed to start [{}]: {}".format(review.primary, shown)
        if result.get("stopped_by_user"):
            suffix = "stopped by you"
            result["output"] = (result.get("output", "") + "\n[The user stopped this command.]").lstrip("\n")
        else:
            suffix = "timed out" if result.get("timed_out") else "exit {}".format(result.get("exit_code"))
        return result, "[terminal] ran [{}]: {} ({})".format(review.primary, shown, suffix)

    def confirm_command(
        self, screen: "curses._CursesWindow", command: str, review: Optional["CommandReview"] = None
    ) -> str:
        """Ask before running a model-requested command. Returns yes, all, or no."""
        review = review or classify_command(command)
        height, width = screen.getmaxyx()
        window_width = max(min(width - 4, 100), 24)
        inner = window_width - 6
        body: List[str] = []
        for line in command.splitlines() or [command]:
            body.extend(textwrap.wrap(line, inner, replace_whitespace=False, drop_whitespace=False) or [""])
        # Title, blank, command, blank, type line, footer, plus the box border.
        fixed_rows = 6
        if len(body) > max(height - fixed_rows - 2, 1):
            # Never approve a command whose full text cannot be shown.
            return "no"
        category_style = {
            "READ": self.style("ok"),
            "CREATION": self.style("accent"),
            "WRITE": self.style("warning"),
            "REMOVAL": self.style("error"),
        }
        type_line = "Type: {}".format(review.primary)
        extra: List[Tuple[str, int]] = []
        for reason in review.risks:
            for wrapped in textwrap.wrap("RISKY: " + reason, inner) or [""]:
                extra.append((wrapped, self.style("error", curses.A_BOLD)))
        for note in review.notes:
            for wrapped in textwrap.wrap(note, inner) or [""]:
                extra.append((wrapped, self.style("muted")))
        extra = extra[: max(height - fixed_rows - 2 - len(body), 0)]
        window_height = min(len(body) + len(extra) + fixed_rows + 2, height)
        window = curses.newwin(window_height, window_width, max((height - window_height) // 2, 0), max((width - window_width) // 2, 0))
        window.keypad(True)
        footer = "Y run   N/Esc deny" if review.needs_confirmation else "Y run   N/Esc deny   A allow all safe until restart"
        if review.needs_confirmation:
            footer += "   (always asks)"
        while True:
            window.erase()
            window.box()
            self.add(window, 1, 2, "Run this command?"[: window_width - 4], curses.A_BOLD)
            for index, line in enumerate(body):
                self.add(window, 3 + index, 3, self.sanitize_display(line)[:inner])
            row = 4 + len(body)
            self.add(window, row, 3, type_line[:inner], category_style[review.primary] | curses.A_BOLD)
            for offset, (text, style) in enumerate(extra, start=1):
                self.add(window, row + offset, 3, text[:inner], style)
            self.add(window, window_height - 2, 2, footer[: window_width - 4], curses.A_DIM)
            window.refresh()
            key = window.get_wch()
            if isinstance(key, str) and key.lower() == "y":
                return "yes"
            if isinstance(key, str) and key.lower() == "a" and not review.needs_confirmation:
                return "all"
            if (isinstance(key, str) and key.lower() == "n") or self.is_cancel_key(key):
                return "no"

    def friendly_error(self, error: ApiError, model: ModelSpec) -> str:
        raw = str(error)
        if model.protocol == "lmstudio":
            lowered = raw.lower()
            if raw.startswith("Network error"):
                return "Could not reach LM Studio at {}. Start its local server (Developer tab, Start Server).".format(self.lmstudio_url)
            if raw.startswith(("HTTP 401", "HTTP 403")):
                return "LM Studio needs an API token. Create one in its Server Settings (Manage Tokens), set LMSTUDIO_API_KEY, and relaunch."
            if "context" in lowered and any(word in lowered for word in ("length", "exceed", "overflow", "too long", "n_keep")):
                return "The conversation no longer fits this model's context in LM Studio ({}). Use /clear or reload the model with a larger context.".format(raw[:120])
            if raw.startswith("HTTP 404") or ("model" in lowered and ("not found" in lowered or "no such" in lowered)):
                return "LM Studio could not find or load {}. Check that it is downloaded (and Just-In-Time loading is on): {}".format(display_model_name(model.identifier), raw[:120])
            return "LM Studio: " + raw
        if model.local_mock and raw.startswith("Network error"):
            return "Could not reach the mock server on 127.0.0.1:{}. Run python3 tools/mock_gemini_server.py in another terminal.".format(MOCK_PORT)
        if raw.startswith("HTTP 429:") and "quota" in raw.lower():
            return "Quota unavailable for {}. Check plan/billing or choose another model.".format(model.identifier)
        if raw.startswith("HTTP 404:") and ("not found" in raw.lower() or "not supported" in raw.lower()):
            return "{} is unavailable for this API key or endpoint. Choose another model with /model.".format(model.identifier)
        return raw

    def clear_context(self) -> None:
        self.data["history"] = []
        self.data["usage"] = {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False}
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
        if command in ("/restart", "/reopen"):
            self.restart_requested = True
            self.running = False
        elif command == "/clear":
            self.clear_context()
        elif command == "/model":
            self.select_model(screen)
        elif command == "/retry":
            self.retry_last(screen)
        elif command == "/thinking":
            self.open_generation_settings(screen)
        elif command == "/permissions":
            if self.terminal_tools_setting:
                self.select_command_permissions(screen)
            else:
                self.status = "Turn on Terminal access in Settings to use /permissions."
        else:
            self.status = "Unknown command. Type / to see the commands."

    def open_action_menu(self, screen: "curses._CursesWindow") -> None:
        actions = (
            ("Settings", "Streaming, terminal access, themes, and system instruction", "settings"),
            ("View transcript", "Read the complete rewrapped conversation", "transcript"),
            ("Check availability", "See which catalog models this API key can use", "check"),
            ("Help", "Show shortcuts and slash commands", "help"),
            ("Usage stats", "All-time tokens, estimated cost, and models used", "stats"),
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
            if self.is_cancel_key(key) or key in ("\t", "q", "Q"):
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
            if action == "settings":
                self.open_settings(screen)
            elif action == "transcript":
                self.show_transcript(screen)
            elif action == "check":
                self.check_availability(screen)
            elif action == "stats":
                self.show_usage_stats(screen)
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
            ("Terminal access", "Let supported models run shell commands (you approve each)", "tools"),
            ("Theme", "Choose the terminal color palette", "theme"),
            ("System instruction", "Set or clear the instruction for this chat", "system"),
            ("Back", "Return to chat", "back"),
        )
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        while True:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
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
                elif label == "Terminal access":
                    value = "[ON]" if self.terminal_tools_setting else "[OFF]"
                    if not model_supports_terminal_tools(self.model.identifier):
                        value += " (not supported by {})".format(display_model_name(self.model.identifier))
                elif label == "Theme":
                    value = self.theme.label
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
            if self.is_cancel_key(key) or key in ("q", "Q"):
                break
            if key in (curses.KEY_UP, "k"):
                selected = max(0, selected - 1)
                continue
            if key in (curses.KEY_DOWN, "j"):
                selected = min(len(settings) - 1, selected + 1)
                continue
            action = settings[selected][2]
            if key == " " and action not in ("stream", "tools"):
                continue
            if key not in (" ", "\n", "\r"):
                continue
            if action == "stream":
                self.data["settings"]["streaming"] = not self.streaming_enabled
                self.store.save(self.data)
                self.status = "Streaming {}.".format("enabled" if self.streaming_enabled else "disabled")
            elif action == "tools":
                self.data["settings"]["terminal_tools"] = not self.terminal_tools_setting
                self.command_permissions = "ask"
                self.store.save(self.data)
                if self.terminal_tools_setting:
                    self.status = "Terminal access on for gemini-flash-latest, Gemini 4+, and Gemini 3.8 Flash+. Each command needs your approval. Streaming is off while it is used."
                else:
                    self.status = "Terminal access disabled."
            elif action == "theme":
                self.select_theme(screen)
            elif action == "system":
                self.edit_system_instruction(screen)
            elif action == "back":
                break
        self.show_cursor()

    def select_theme(self, screen: "curses._CursesWindow") -> None:
        original_theme_id = self.theme.identifier
        selected_id = original_theme_id
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
                    self.add(screen, row + 2, 5, "YOU", self.style("user", curses.A_BOLD))
                    self.add(screen, row + 2, 10, "sample prompt", self.style("text"))
                    self.add(screen, row + 2, 28, "GEMINI", self.style("assistant", curses.A_BOLD))
                    self.add(screen, row + 2, 36, "sample response", self.style("text"))
            self.add(screen, height - 1, 2, "Up/Down select   Enter apply   Esc close", self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if self.is_cancel_key(key) or key in ("q", "Q"):
                if self.data["settings"]["theme"] != original_theme_id:
                    self.data["settings"]["theme"] = original_theme_id
                    self.initialize_theme(screen)
                return
            if key in (curses.KEY_UP, "k"):
                selected_id = THEMES[max(0, selected - 1)].identifier
                self.preview_theme(screen, selected_id)
                continue
            if key in (curses.KEY_DOWN, "j"):
                selected_id = THEMES[min(len(THEMES) - 1, selected + 1)].identifier
                self.preview_theme(screen, selected_id)
                continue
            if key in ("\n", "\r"):
                selected_theme = THEME_BY_ID[selected_id]
                self.preview_theme(screen, selected_theme.identifier)
                self.store.save(self.data)
                self.status = "Theme set to {}.".format(selected_theme.label)
                return

    def preview_theme(self, screen: "curses._CursesWindow", identifier: str) -> None:
        self.data["settings"]["theme"] = identifier
        self.initialize_theme(screen)

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
                if (command[0] != "/permissions" or self.terminal_tools_setting)
                and (command[0] != "/retry" or self.last_failed_prompt)
                and query.lower() in "{} {}".format(command[0], command[1]).lower()
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
            if self.is_cancel_key(key):
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

    def fetch_lmstudio_models(self) -> List[ModelSpec]:
        """Models LM Studio reports right now, or a single placeholder row when its server is not running."""
        client = LMStudioClient(self.lmstudio_url, timeout=2)
        try:
            names = client.list_models()
        except ApiError:
            return [LMSTUDIO_OFFLINE_ENTRY]
        entries = []
        for name in names:
            info = client.model_info.get(name)
            if info is None:
                note = ""
            elif info["loaded"]:
                note = "Loaded in LM Studio; free to run"
            else:
                note = "Downloaded but not loaded; LM Studio loads it on first use if Just-In-Time loading is on"
            if info and info["context"]:
                self.context_limits[LMSTUDIO_PREFIX + name] = info["context"]
            entries.append(lmstudio_model_spec(name, note))
        return entries or [LMSTUDIO_OFFLINE_ENTRY]

    def select_model(self, screen: "curses._CursesWindow") -> None:
        lm_entries = self.fetch_lmstudio_models()
        entries = (
            CUSTOM_MODEL_PICKER_ENTRY,
            *(model for model in MODEL_CATALOG if not model.retired),
            *lm_entries,
            *(model for model in MODEL_CATALOG if model.retired),
        )
        selected_id = (
            self.model.identifier
            if self.data["model"] in MODEL_BY_ID or self.data["model"] in {model.identifier for model in lm_entries}
            else CUSTOM_MODEL_PICKER_ENTRY.identifier
        )
        query = ""
        while True:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            screen.erase()
            height, width = screen.getmaxyx()
            matches = [
                model
                for model in entries
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
                if model.identifier == selected_id:
                    line_style = self.style("selected", curses.A_DIM if model.retired else 0)
                else:
                    line_style = self.style("muted") if model.retired else self.style("text")
                self.add(screen, displayed, 0, line[: width - 1], line_style)
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
                selected_model = matches[selected]
                if selected_model.identifier == LMSTUDIO_OFFLINE_ENTRY.identifier:
                    self.status = "LM Studio is not reachable at {}. Start its local server (Developer tab, Start Server) or relaunch with --lmstudio-url.".format(self.lmstudio_url)
                    self.show_cursor()
                    return
                if selected_model.identifier == CUSTOM_MODEL_PICKER_ENTRY.identifier:
                    if self.select_custom_model(screen):
                        self.show_cursor()
                        return
                    continue
                self.data["model"] = selected_model.identifier
                self.store.save(self.data)
                if selected_model.retired:
                    self.status = "Selected {}, which is likely unavailable. Shared context is retained.".format(selected_model.label)
                else:
                    self.status = "Selected {}. Shared context is retained.".format(selected_model.label)
                self.show_cursor()
                return
            if self.is_cancel_key(key):
                self.show_cursor()
                return
            if key == curses.KEY_UP:
                if matches:
                    selected_id = matches[max(0, selected - 1)].identifier
            elif key == curses.KEY_DOWN:
                if matches:
                    selected_id = matches[min(len(matches) - 1, selected + 1)].identifier
            elif key == curses.KEY_F5:
                self.check_availability(screen)
            elif key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                query = query[:-1]
            elif key == "\x15":
                query = ""
            elif isinstance(key, str) and key.isprintable() and len(query) < 128:
                query += key

    def select_custom_model(self, screen: "curses._CursesWindow") -> bool:
        initial = self.data["model"] if self.data["model"] not in MODEL_BY_ID else ""
        value = self.prompt_dialog(
            screen,
            "Custom Gemini model ID",
            initial,
            max_length=MAX_MODEL_ID_CHARS,
        )
        if value is None:
            return False
        if not value:
            self.status = "Custom model ID was not changed."
            return False
        try:
            self.data["model"] = validate_model_id(value)
            self.store.save(self.data)
            self.status = "Selected custom model {}. Shared context is retained.".format(value)
            return True
        except ApiError as error:
            self.status = str(error)
            return False

    def availability_marker(self, model: ModelSpec) -> str:
        if self.available_models is None or model.protocol == "lmstudio":
            return model.availability_note
        return "available" if model.identifier in self.available_models else "not listed for this key"

    def check_availability(self, screen: "curses._CursesWindow") -> None:
        self.status = "Checking the models visible to this API key..."
        self.draw_chat(screen)
        try:
            self.available_models = set(GoogleApiClient(self.api_key).list_models())
            report = self.availability_report()
            self.status = report[0]
            self.show_message(screen, "Model availability", report)
        except ApiError as error:
            self.status = str(error)
            self.show_message(
                screen,
                "Model availability",
                ["The availability check could not complete.", "", str(error)],
            )

    def availability_report(self) -> List[str]:
        available = self.available_models or set()
        listed = [model for model in MODEL_CATALOG if model.identifier in available]
        unavailable = [model for model in MODEL_CATALOG if model.identifier not in available]
        lines = [
            "{} of {} configured model(s) are listed for this API key.".format(len(listed), len(MODEL_CATALOG)),
            "",
        ]
        if listed:
            lines.append("Listed by Google:")
            lines.extend("  {} ({})".format(model.label, model.identifier) for model in listed)
        else:
            lines.append("None of this app's configured models were listed by Google.")
        if unavailable:
            lines.extend(["", "Not listed for this key:"])
            lines.extend("  {} ({})".format(model.label, model.identifier) for model in unavailable)
        return lines

    @staticmethod
    def editor_rows(text: str, wrap: int) -> List[Tuple[int, int]]:
        """Visual rows of the text editor as (start, end) offsets; long lines wrap at a fixed width."""
        rows: List[Tuple[int, int]] = []
        offset = 0
        for line in text.split("\n"):
            for chunk in range(len(line) // wrap + 1):
                rows.append((offset + chunk * wrap, min(offset + len(line), offset + (chunk + 1) * wrap)))
            offset += len(line) + 1
        return rows

    @staticmethod
    def editor_cursor(rows: List[Tuple[int, int]], cursor: int) -> Tuple[int, int]:
        row = 0
        for index, (start, end) in enumerate(rows):
            if start <= cursor <= end:
                row = index
        return row, cursor - rows[row][0]

    def edit_text_dialog(
        self, screen: "curses._CursesWindow", title: str, initial: str, max_length: int = MAX_DIALOG_CHARS
    ) -> Optional[str]:
        """A full-screen multi-line editor. Ctrl+D saves, Esc cancels; Enter inserts a new line."""
        text = initial.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")[:max_length]
        cursor = len(text)
        top = 0
        previous_enter = False
        while True:
            height, width = screen.getmaxyx()
            if height < 8 or width < 24:
                self.status = "Terminal is too small for the editor. Resize it and try again."
                return None
            wrap = width - 4
            rows = self.editor_rows(text, wrap)
            row, column = self.editor_cursor(rows, cursor)
            visible = height - 4
            top = min(max(top, row - visible + 1), row)
            screen.erase()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("header"))
            self.add(screen, 0, 1, title[: width - 2], self.style("header", curses.A_BOLD))
            for screen_row, (start, end) in enumerate(rows[top : top + visible], start=2):
                printable = "".join(character if character.isprintable() else " " for character in text[start:end])
                self.add(screen, screen_row, 2, printable, self.style("text"))
            footer = "Ctrl+D save   Esc cancel   Enter new line   Ctrl+U clear all   {} chars".format(len(text))
            self.add(screen, height - 1, 1, footer[: width - 2], self.style("footer"))
            try:
                curses.curs_set(1)
                screen.move(2 + row - top, min(2 + column, width - 2))
            except curses.error:
                pass
            screen.refresh()
            key = screen.get_wch()
            if key == "\x04":
                self.show_cursor()
                return text.strip()
            if self.is_cancel_key(key):
                self.show_cursor()
                return None
            if key in ("\n", "\r"):
                # A pasted CRLF arrives as two keys; count it as one new line.
                if not (key == "\n" and previous_enter) and len(text) < max_length:
                    text = text[:cursor] + "\n" + text[cursor:]
                    cursor += 1
                previous_enter = key == "\r"
                continue
            previous_enter = False
            if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
                if cursor:
                    text = text[: cursor - 1] + text[cursor:]
                    cursor -= 1
            elif key == curses.KEY_DC:
                text = text[:cursor] + text[cursor + 1 :]
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(text), cursor + 1)
            elif key in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_PPAGE, curses.KEY_NPAGE):
                step = {curses.KEY_UP: -1, curses.KEY_DOWN: 1, curses.KEY_PPAGE: -visible, curses.KEY_NPAGE: visible}[key]
                target = min(max(row + step, 0), len(rows) - 1)
                start, end = rows[target]
                cursor = start + min(column, end - start)
            elif key in (curses.KEY_HOME, "\x01"):
                cursor = rows[row][0]
            elif key in (curses.KEY_END, "\x05"):
                cursor = rows[row][1]
            elif key == "\x15":
                text = ""
                cursor = 0
            elif isinstance(key, str) and key.isprintable() and len(text) < max_length:
                text = text[:cursor] + key + text[cursor:]
                cursor += 1

    def confirm_dialog(self, screen: "curses._CursesWindow", title: str, lines: Sequence[str]) -> bool:
        height, width = screen.getmaxyx()
        window_width = max(min(width - 4, 80), 24)
        body: List[str] = []
        for line in lines:
            body.extend(textwrap.wrap(line, window_width - 6) or [""])
        body = body[: max(height - 6, 1)]
        window_height = min(len(body) + 6, height)
        window = curses.newwin(window_height, window_width, max((height - window_height) // 2, 0), max((width - window_width) // 2, 0))
        window.keypad(True)
        while True:
            window.erase()
            window.box()
            self.add(window, 1, 2, title[: window_width - 4], curses.A_BOLD)
            for index, line in enumerate(body):
                self.add(window, 3 + index, 3, line[: window_width - 6])
            self.add(window, window_height - 2, 2, "Y confirm   N/Esc cancel"[: window_width - 4], curses.A_DIM)
            window.refresh()
            key = window.get_wch()
            if isinstance(key, str) and key.lower() == "y":
                return True
            if (isinstance(key, str) and key.lower() == "n") or self.is_cancel_key(key):
                return False

    def open_generation_settings(self, screen: "curses._CursesWindow") -> None:
        """/thinking: thinking level plus the other generation parameters the API accepts."""
        selected = 0
        message = ""
        while True:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            generation = self.data["settings"].setdefault("generation", {})
            mode, levels = thinking_support(self.model.identifier)
            rows: List[Tuple[str, str, str, str]] = []
            if mode == "level":
                level = generation.get("thinking_level")
                value = level or "default"
                if level and level not in levels:
                    value += "   (not supported by {}; ignored)".format(display_model_name(self.model.identifier))
                rows.append(("thinking_level", "Thinking level", value, "How much the model reasons before answering. This model accepts: {}.".format(", ".join(levels))))
            elif mode == "budget":
                budget = generation.get("thinking_budget")
                value = "default" if budget is None else "dynamic" if budget == -1 else "off" if budget == 0 else "{} tokens".format(budget)
                rows.append(("thinking_budget", "Thinking budget", value, "Tokens the model may spend thinking: -1 dynamic, 0 off, up to {}.".format(MAX_THINKING_BUDGET)))
            else:
                rows.append(("thinking_none", "Thinking", "not supported by this model", "Choose a Gemini model with /model to set thinking."))
            for parameter in GENERATION_PARAMETERS:
                raw = generation.get(parameter.key)
                value = "default" if raw is None else ", ".join(raw) if parameter.kind == "list" else str(raw)
                rows.append((parameter.key, parameter.label, value, parameter.description))
            rows.append(("reset", "Reset all to defaults", "", "Clear every value above so the model's own defaults apply."))
            selected = min(selected, len(rows) - 1)
            screen.erase()
            height, width = screen.getmaxyx()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("header"))
            self.add(screen, 0, 1, "THINKING AND GENERATION SETTINGS", self.style("header", curses.A_BOLD))
            self.add(screen, 2, 2, "Applies to {}. Only values you set are sent; the rest use the model's defaults.".format(self.model.identifier)[: width - 4], self.style("muted"))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            visible = max((height - 8) // 2, 1)
            offset = min(max(selected - visible + 1, 0), max(len(rows) - visible, 0))
            for displayed, index in enumerate(range(offset, min(offset + visible, len(rows)))):
                key, label, value, description = rows[index]
                row = 5 + displayed * 2
                style = self.style("selected") if index == selected else self.style("text")
                self.add(screen, row, 2, "{}  {: <22} {}".format(">" if index == selected else " ", label, value)[: width - 4], style)
                self.add(screen, row + 1, 5, description[: width - 7], self.style("muted"))
            if message:
                self.add(screen, height - 2, 2, message[: width - 4], self.style("warning"))
            self.add(screen, height - 1, 2, "Up/Down select   Enter change   R reset this one   Esc back"[: width - 4], self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            message = ""
            if self.is_cancel_key(key) or key in ("q", "Q"):
                break
            if key in (curses.KEY_UP, "k"):
                selected = max(0, selected - 1)
                continue
            if key in (curses.KEY_DOWN, "j"):
                selected = min(len(rows) - 1, selected + 1)
                continue
            name = rows[selected][0]
            if key in ("r", "R") and name not in ("reset", "thinking_none"):
                generation.pop(name, None)
            elif key in ("\n", "\r", " "):
                if name == "reset":
                    generation.clear()
                elif name == "thinking_level":
                    options = [None, *levels]
                    current = generation.get("thinking_level")
                    following = options[(options.index(current) + 1) % len(options)] if current in options else options[1]
                    if following is None:
                        generation.pop("thinking_level", None)
                    else:
                        generation["thinking_level"] = following
                elif name != "thinking_none":
                    parameter = GENERATION_PARAMETER_BY_KEY.get(name)
                    label = parameter.label if parameter else "Thinking budget"
                    current = generation.get(name)
                    initial = "" if current is None else ", ".join(current) if isinstance(current, list) else str(current)
                    typed = self.prompt_dialog(screen, "{} (blank = default)".format(label), initial, max_length=96)
                    if typed is None:
                        continue
                    if not typed:
                        generation.pop(name, None)
                    else:
                        try:
                            generation[name] = parse_generation_value(name, typed)
                        except ApiError as error:
                            message = str(error)
                            continue
                    if generation.get(name) == []:
                        generation.pop(name, None)
            else:
                continue
            self.store.save(self.data)
        self.show_cursor()

    def select_command_permissions(self, screen: "curses._CursesWindow") -> None:
        options = (
            ("ask", "Ask approval each time", "Every command shows a dialog and waits for Y or N."),
            ("safe", "Allow all safe", "Run read, creation and write commands without asking. Removals and risky commands still ask."),
            ("all", "Allow literally everything", "Run every command with no prompts, including removals, sudo, installs, and git push."),
        )
        selected = next((index for index, option in enumerate(options) if option[0] == self.command_permissions), 0)
        while True:
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            screen.erase()
            height, width = screen.getmaxyx()
            self.add(screen, 0, 0, " " * max(width - 1, 0), self.style("header"))
            self.add(screen, 0, 1, "TERMINAL PERMISSIONS", self.style("header", curses.A_BOLD))
            self.add(screen, 2, 2, "Applies to commands the model asks to run. Resets to 'ask' when you restart.", self.style("muted"))
            self.add(screen, 3, 0, "-" * max(width - 1, 0), self.style("border"))
            for index, (identifier, label, description) in enumerate(options):
                row = 5 + index * 3
                if row + 1 >= height - 1:
                    break
                current = "(current)" if identifier == self.command_permissions else ""
                style = self.style("selected") if index == selected else self.style("text")
                if identifier == "all" and index != selected:
                    style = self.style("error")
                self.add(screen, row, 2, "{}  {: <28} {}".format(">" if index == selected else " ", label, current)[: width - 4], style)
                self.add(screen, row + 1, 5, description[: width - 7], self.style("muted"))
            self.add(screen, height - 1, 2, "Up/Down select   Enter choose   Esc close", self.style("footer"))
            screen.refresh()
            key = screen.get_wch()
            if self.is_cancel_key(key) or key in ("q", "Q"):
                break
            if key in (curses.KEY_UP, "k"):
                selected = max(0, selected - 1)
            elif key in (curses.KEY_DOWN, "j"):
                selected = min(len(options) - 1, selected + 1)
            elif key in ("\n", "\r"):
                identifier, label = options[selected][0], options[selected][1]
                if identifier == "all" and not self.confirm_dialog(
                    screen,
                    "Allow literally everything?",
                    ["Every command the model requests will run immediately with no prompt, including deletions, sudo, and installs, until you restart or turn Terminal access off."],
                ):
                    continue
                self.command_permissions = identifier
                self.status = "Terminal permissions: {}.".format(label)
                break
        self.show_cursor()

    def edit_system_instruction(self, screen: "curses._CursesWindow") -> None:
        value = self.edit_text_dialog(
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
        try:
            curses.curs_set(1)
        except curses.error:
            pass
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
            if self.is_cancel_key(key):
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
                "Tab opens the main menu (settings, help, transcript, exit). Type / for the searchable command palette.",
                "Enter sends; Shift+Enter adds a new line (see the README for terminal setup). Up/Down or Page Up/Page Down scroll the chat. Ctrl+P/Ctrl+O recall submitted prompts.",
                "F6 opens the complete conversation reader with Up/Down scrolling.",
                "Settings contains streaming, terminal access, themes, and the system instruction. The Tab menu also has availability checks and all-time usage stats.",
                "/clear erases context; /retry repeats a failed prompt; /restart reopens the UI; /thinking sets thinking and generation parameters; /permissions sets terminal approval (when Terminal access is on).",
                "The API key comes from GEMINI_API_KEY or --api-key-file when you launch, and is never written to disk. Transcripts live in the selected state folder.",
            ],
        )

    def show_usage_stats(self, screen: "curses._CursesWindow") -> None:
        lifetime = self.data["lifetime"]
        lines = [
            "All-time usage across every conversation (/clear does not reset this)",
            "",
            "  Tokens in:       {}".format(self.format_tokens(lifetime["input"])),
            "  Tokens out:      {}".format(self.format_tokens(lifetime["output"])),
            "  Estimated cost:  {}".format(self.format_cost(lifetime)),
            "",
            "Models used",
        ]
        models = sorted(lifetime["models"].items(), key=lambda item: item[1]["input"] + item[1]["output"], reverse=True)
        for name, totals in models:
            lines.append(
                "  {}   {} in / {} out   {}".format(
                    display_model_name(name), self.format_tokens(totals["input"]), self.format_tokens(totals["output"]), self.format_cost(totals)
                )
            )
        if not models:
            lines.append("  none yet")
        lines += ["", "Costs are estimates from Google's published prices; a trailing + means some usage had no known price."]
        self.show_message(screen, "Usage stats", lines)

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
            if self.is_cancel_key(key) or key in ("\n", "\r", "q", "Q"):
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
    def plain_terminal_response(content: str) -> str:
        """Keep assistant output readable in a terminal even when Markdown slips through."""
        cleaned_lines: List[str] = []
        for line in content.splitlines():
            line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
            line = re.sub(r"^\s{0,3}>\s?", "", line)
            if re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", line):
                continue
            line = re.sub(r"`([^`]*)`", r"\1", line)
            line = line.replace("`", "")
            line = line.replace("**", "").replace("__", "")
            line = re.sub(r"\*([^*\n]+)\*", r"\1", line)
            line = re.sub(r"_([^_\n]+)_", r"\1", line)
            cleaned_lines.append(line.replace("*", "").replace("|", "  "))
        return "\n".join(cleaned_lines)

    @staticmethod
    def is_cancel_key(key: object) -> bool:
        return key in ("\x1b", 27, curses.KEY_EXIT)

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
            curses.init_pair(1, theme.header, theme.background)
            curses.init_pair(2, theme.accent, theme.background)
            curses.init_pair(3, theme.text, theme.background)
            curses.init_pair(4, theme.user, theme.background)
            curses.init_pair(5, theme.warning, theme.background)
            curses.init_pair(6, theme.error, theme.background)
            curses.init_pair(8, theme.selected_text, theme.selected_background)
            curses.init_pair(9, theme.text, theme.background)
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
    parser = argparse.ArgumentParser(description="Terminal UI for Gemini models and legacy PaLM chat.")
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
    parser.add_argument(
        "--lmstudio-url",
        help="Address of LM Studio's local server (default http://127.0.0.1:1234, or the LMSTUDIO_BASE_URL environment variable).",
    )
    parser.add_argument(
        "--api-base-url",
        help="Send API requests to a local test server such as tools/mock_gemini_server.py (loopback http:// URLs only).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.list_models:
        for model in MODEL_CATALOG:
            print("{}\t{}\t{}".format(model.identifier, model.group, model.availability_note))
        return 0
    if arguments.api_base_url:
        global API_BASE_URL
        parsed_base = urllib.parse.urlparse(arguments.api_base_url)
        if parsed_base.scheme != "http" or parsed_base.hostname not in ("127.0.0.1", "localhost", "::1") or parsed_base.path not in ("", "/"):
            print("--api-base-url must be a loopback address like http://127.0.0.1:8765", file=sys.stderr)
            return 2
        API_BASE_URL = "http://{}".format(parsed_base.netloc)
    lmstudio_address = arguments.lmstudio_url or os.environ.get("LMSTUDIO_BASE_URL")
    if lmstudio_address:
        global LMSTUDIO_URL
        try:
            LMSTUDIO_URL = validate_lmstudio_url(lmstudio_address)
        except ApiError as error:
            print(str(error), file=sys.stderr)
            return 2
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
        os.environ.setdefault("ESCDELAY", "25")  # make a lone Esc respond immediately
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
