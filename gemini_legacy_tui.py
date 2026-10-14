#!/usr/bin/env python3
"""A lightweight terminal chat client for Gemini models and legacy PaLM Chat."""

import argparse
import json
import os
import re
import shlex
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
MAX_TERMINAL_RESULTS = 10
STOP_HINT = ">>> PRESS ESC TO STOP <<<"
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
    "You are replying inside a terminal chat interface. Use plain text only. Never use Markdown "
    "emphasis or styling markers such as asterisks, underscores, backticks, heading hashes, block "
    "quote markers, pipe tables, or decorative separators. Do not attempt emphasis in any form. "
    "Blank lines between distinct blocks are allowed when they make the response easier to read."
)
SLASH_COMMANDS: Sequence[Tuple[str, str]] = (
    ("/clear", "Clear the shared conversation context"),
    ("/model", "Choose a model"),
    ("/retry", "Retry the last failed prompt"),
    ("/permissions", "Choose how terminal commands are approved (needs Terminal access)"),
    ("/key", "Enter an API key for this run"),
    ("/check", "Check model availability"),
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
    if not all(
        character.isascii() and (character.isalnum() or character in ("-", "_", "."))
        for character in identifier
    ):
        raise ApiError("Model IDs can contain only letters, numbers, hyphens, underscores, and periods.")
    return identifier


def model_supports_terminal_tools(identifier: str) -> bool:
    """Terminal access is limited to gemini-flash-latest, Gemini 4 and later, and Gemini 3.8 Flash and later."""
    if identifier == "gemini-flash-latest":
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
    "mdfind", "xxd", "hexdump", "strings",
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


class GoogleApiClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = validate_api_key(api_key)
        self.usage_input = 0  # prompt tokens billed by the requests this client made
        self.usage_output = 0  # response tokens, including thinking tokens
        self.model_version = ""  # the model Google says answered (resolves aliases like gemini-flash-latest)

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
        self.usage_input += prompt_tokens
        self.usage_output += output_tokens
        version = response.get("modelVersion")
        if isinstance(version, str) and version.strip():
            self.model_version = version.strip()[:MAX_MODEL_ID_CHARS]

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

    def list_models(self) -> List[str]:
        url = API_BASE_URL + "/v1beta/models"
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
        url = (API_BASE_URL + "/v1beta/models/{}:generateContent").format(
            urllib.parse.quote(model.identifier, safe="")
        )
        response = self._request(url, self._gemini_body(history, system_instruction))
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
            if isinstance(part, dict) and isinstance(part.get("text"), str)
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
        url = (API_BASE_URL + "/v1beta/models/{}:generateContent").format(
            urllib.parse.quote(model.identifier, safe="")
        )
        body = self._gemini_body(history, system_instruction)
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
            raise ApiError("No API key. Set GEMINI_API_KEY or press F4 to enter one.")
        url = (API_BASE_URL + "/v1beta/models/{}:streamGenerateContent?alt=sse").format(
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
        self.usage_input += latest_usage[0]
        self.usage_output += latest_usage[1]
        text = "".join(received).strip()
        if not text:
            raise ApiError("The selected model returned no text.")
        return text

    def _generate_palm(self, model: ModelSpec, history: Sequence[Dict[str, str]], system_instruction: str) -> str:
        # PaLM Chat used a different, now-retired API shape. Keeping this request
        # lets accounts with an available legacy endpoint attempt the original model.
        url = (API_BASE_URL + "/v1beta3/models/{}:generateMessage").format(
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
        raw_usage = data.get("usage")
        usage = {"input": 0, "output": 0}
        if isinstance(raw_usage, dict):
            for name in usage:
                value = raw_usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10**12:
                    usage[name] = value
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
            "settings": {
                "streaming": streaming if isinstance(streaming, bool) else True,
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
        self.data.setdefault("usage", {"input": 0, "output": 0})
        self.data["settings"].setdefault("streaming", True)
        self.data["settings"].setdefault("terminal_tools", False)
        self.data["settings"].setdefault("system_instruction_configured", False)
        self.data["settings"].setdefault("theme", DEFAULT_THEME_ID)
        for item in self.data["history"]:
            if item["role"] == "model":
                item["content"] = self.plain_terminal_response(item["content"])
        self.status = "Ready" if self.api_key else "No API key. Press F4 or open Settings from the Tab menu to enter one."
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
        self.command_permissions = "ask"  # ask, safe, or all; resets on restart
        self.stop_requested = False

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

    def header_parts(self) -> Tuple[str, str]:
        """The one-line header: (model id, the rest)."""
        turns = sum(1 for item in self.data["history"] if item["role"] == "user")
        usage = self.data["usage"]
        rest = "   {} turn{}   {} in / {} out tokens".format(
            turns,
            "" if turns == 1 else "s",
            self.format_tokens(usage["input"]),
            self.format_tokens(usage["output"]),
        )
        return self.model.identifier, rest

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
            if key == curses.KEY_F4:
                self.configure_api_key(screen)
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
        client = GoogleApiClient(self.api_key)
        if use_tools:
            self.status = "{}   Waiting for {}...".format(STOP_HINT, selected_model.identifier)
        elif self.streaming_enabled:
            self.status = "Streaming {}...".format(selected_model.identifier)
        else:
            self.status = "Waiting for {}...".format(selected_model.identifier)
        try:
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
            elif selected_model.protocol == "gemini" and self.streaming_enabled:
                reply = {"role": "model", "content": "", "model_id": selected_model.identifier}
                self.data["history"].append(reply)
                self.draw_chat(screen)

                def append_chunk(chunk: str) -> None:
                    fragments = self.display_fragments(self.plain_terminal_response(chunk))
                    for fragment in fragments:
                        reply["content"] += fragment
                        self.status = "Streaming {}... {} chars".format(selected_model.identifier, len(reply["content"]))
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
        finally:
            self.add_usage(client)
        self.store.save(self.data)

    def add_usage(self, client: "GoogleApiClient") -> None:
        usage = self.data["usage"]
        usage["input"] += client.usage_input
        usage["output"] += client.usage_output

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
            "You can run shell commands on the user's computer with the {} tool. The user may deny a command "
            "or stop the turn; do not retry a denied command. Platform: {}. Working directory: {}. Use non-interactive commands, "
            "keep output small, and explain what you did in plain text. You get at most {limit} terminal results per turn."
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
        self.status = "{}   Running: {}".format(STOP_HINT, shown)
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
        if len(review.categories) > 1:
            type_line += "   (also: {})".format(", ".join(review.categories[:-1][::-1]))
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
        if raw.startswith("HTTP 429:") and "quota" in raw.lower():
            return "Quota unavailable for {}. Check plan/billing or choose another model.".format(model.identifier)
        if raw.startswith("HTTP 404:") and ("not found" in raw.lower() or "not supported" in raw.lower()):
            return "{} is unavailable for this API key or endpoint. Choose another model with Ctrl+L.".format(model.identifier)
        return raw

    def clear_context(self) -> None:
        self.data["history"] = []
        self.data["usage"] = {"input": 0, "output": 0}
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
        elif command == "/permissions":
            if self.terminal_tools_setting:
                self.select_command_permissions(screen)
            else:
                self.status = "Turn on Terminal access in Settings to use /permissions."
        elif command == "/key":
            self.configure_api_key(screen)
        elif command == "/check":
            self.check_availability(screen)
        else:
            self.status = "Unknown command. Type / to see the commands."

    def open_action_menu(self, screen: "curses._CursesWindow") -> None:
        actions = (
            ("Settings", "Themes, streaming, system instruction, API key, and availability", "settings"),
            ("View transcript", "Read the complete rewrapped conversation", "transcript"),
            ("Clear context", "Remove the shared conversation history", "clear"),
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
            if action == "clear":
                self.clear_context()
            elif action == "settings":
                self.open_settings(screen)
            elif action == "transcript":
                self.show_transcript(screen)
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
            ("API key", "Enter an API key for this run only", "key"),
            ("Model availability", "Check models visible to this API key", "check"),
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
                        value += " (not supported by {})".format(self.model.identifier)
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
            elif action == "key":
                self.configure_api_key(screen)
            elif action == "check":
                self.check_availability(screen)
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

    def select_model(self, screen: "curses._CursesWindow") -> None:
        selected_id = (
            self.model.identifier
            if self.data["model"] in MODEL_BY_ID
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
                for model in (CUSTOM_MODEL_PICKER_ENTRY, *MODEL_CATALOG)
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
            if self.is_cancel_key(key) or key in ("q", "Q"):
                self.show_cursor()
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
        if self.available_models is None:
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
                "Settings contains streaming, terminal access, themes, system instruction, API key, and model availability.",
                "/clear erases context; /retry repeats a failed prompt; /restart reopens the UI; /permissions sets terminal approval (when Terminal access is on).",
                "API keys are never written to disk. Transcripts live in the selected state folder.",
            ],
        )

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
        if STOP_HINT in self.status:
            return self.style("warning", curses.A_BOLD | curses.A_REVERSE)
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
