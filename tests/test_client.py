import io
import json
import os
import tempfile
import unittest
import urllib.error
import curses
from datetime import date
from pathlib import Path
from unittest.mock import patch

import gemini_legacy_tui as app
from gemini_legacy_tui import (
    ApiError,
    DEFAULT_SYSTEM_INSTRUCTION,
    DEFAULT_THEME_ID,
    GoogleApiClient,
    MAX_API_RESPONSE_BYTES,
    MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK,
    MAX_STREAM_EVENT_BYTES,
    MODEL_BY_ID,
    MODEL_CATALOG,
    THEME_BY_ID,
    THEMES,
    NoRedirectHandler,
    SLASH_COMMANDS,
    SessionStore,
    Tui,
    build_generation_config,
    estimate_cost,
    parse_generation_value,
    thinking_support,
    classify_command,
    model_supports_terminal_tools,
    run_shell_command,
    validate_model_id,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self, amount=None):
        return self.payload if amount is None else self.payload[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeStreamResponse:
    def __init__(self, events):
        self.events = ["data: {}\n".format(json.dumps(event)).encode("utf-8") for event in events]
        self.index = 0

    def readline(self, amount=None):
        if self.index >= len(self.events):
            return b""
        event = self.events[self.index]
        self.index += 1
        return event if amount is None else event[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeScreen:
    def __init__(self, keys):
        self.keys = iter(keys)

    def getmaxyx(self):
        return (24, 80)

    def addstr(self, *_args):
        pass

    def erase(self):
        pass

    def move(self, *_args):
        pass

    def refresh(self):
        pass

    def nodelay(self, _flag):
        pass

    def get_wch(self):
        key = next(self.keys)
        if key is None:  # nothing is waiting in the terminal's input buffer
            raise curses.error("no input")
        return key


class ClientTests(unittest.TestCase):
    def test_catalog_has_requested_models_and_greys_out_retired_ones(self):
        identifiers = [model.identifier for model in MODEL_CATALOG]
        self.assertEqual(identifiers[0], "gemini-4-argon")
        for wanted in ("gemini-3.8-flash", "gemini-flash-latest"):
            self.assertFalse(MODEL_BY_ID[wanted].retired)
        for old in ("gemini-3.5-flash", "gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002",
                    "gemini-2.0-flash", "gemini-2.0-flash-001", "gemini-2.0-flash-exp", "chat-bison-001"):
            self.assertTrue(MODEL_BY_ID[old].retired)
        palm = [model.identifier for model in MODEL_CATALOG if model.protocol == "palm"]
        self.assertEqual(palm, ["chat-bison-001"])
        self.assertFalse(any("lite" in model or "pro" in model for model in identifiers))
        self.assertFalse(any("image" in model for model in identifiers))

    def test_theme_catalog_includes_matrix_green_and_default_theme(self):
        self.assertIn("matrix", THEME_BY_ID)
        self.assertEqual(THEME_BY_ID["matrix"].label, "Matrix Green")
        self.assertEqual(THEME_BY_ID["matrix"].background, curses.COLOR_BLACK)
        self.assertEqual(THEME_BY_ID["arctic"].background, curses.COLOR_BLUE)
        self.assertEqual([theme.identifier for theme in THEMES][0], DEFAULT_THEME_ID)

    def test_session_theme_selection_persists_and_invalid_values_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save({"settings": {"theme": "matrix"}})
            tui = Tui("", store)
        self.assertEqual(tui.theme.identifier, "matrix")
        self.assertEqual(SessionStore.normalize({"settings": {"theme": "unknown"}})["settings"]["theme"], DEFAULT_THEME_ID)

    def test_availability_report_keeps_the_result_visible_and_names_models(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
        tui.available_models = {"gemini-2.5-flash"}
        report = tui.availability_report()
        self.assertEqual(report[0], "1 of {} configured model(s) are listed for this API key.".format(len(MODEL_CATALOG)))
        self.assertIn("  Gemini 2.5 Flash (gemini-2.5-flash)", report)
        self.assertIn("Not listed for this key:", report)

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_gemini_uses_generate_content_schema(self, mocked_open):
        mocked_open.return_value = FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
        )
        client = GoogleApiClient("test-key")
        answer = client.generate(
            MODEL_BY_ID["gemini-2.5-flash"],
            [{"role": "user", "content": "Hi"}],
            "Be concise.",
        )
        request = mocked_open.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(answer, "hello")
        self.assertIn(":generateContent", request.full_url)
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "Be concise.")

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_palm_uses_legacy_chat_schema(self, mocked_open):
        mocked_open.return_value = FakeResponse({"candidates": [{"content": "hello from PaLM"}]})
        client = GoogleApiClient("test-key")
        answer = client.generate(
            MODEL_BY_ID["chat-bison-001"],
            [{"role": "user", "content": "Hi"}, {"role": "model", "content": "Hello"}],
            "Be concise.",
        )
        request = mocked_open.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(answer, "hello from PaLM")
        self.assertIn("v1beta3/models/chat-bison-001:generateMessage", request.full_url)
        self.assertEqual(body["prompt"]["messages"][0]["author"], "0")
        self.assertEqual(body["prompt"]["messages"][1]["author"], "1")
        self.assertEqual(body["prompt"]["context"], "Be concise.")

    def test_palm_retirement_error_is_explained(self):
        client = GoogleApiClient("test-key")
        with patch.object(client, "_request", side_effect=ApiError("HTTP 501: unavailable")):
            with self.assertRaisesRegex(ApiError, "retired"):
                client.generate(MODEL_BY_ID["chat-bison-001"], [{"role": "user", "content": "Hi"}], "")

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_gemini_stream_yields_sse_text_chunks(self, mocked_open):
        mocked_open.return_value = FakeStreamResponse(
            [
                {"candidates": [{"content": {"parts": [{"text": "hel"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
            ]
        )
        chunks = []
        answer = GoogleApiClient("test-key").generate_stream(
            MODEL_BY_ID["gemini-2.5-flash"], [{"role": "user", "content": "Hi"}], "", chunks.append
        )
        request = mocked_open.call_args.args[0]
        self.assertEqual(chunks, ["hel", "lo"])
        self.assertEqual(answer, "hello")
        self.assertIn(":streamGenerateContent?alt=sse", request.full_url)
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_model_list_uses_api_key_header(self, mocked_open):
        mocked_open.return_value = FakeResponse({"models": [{"name": "models/gemini-test"}]})
        models = GoogleApiClient("test-key").list_models()
        request = mocked_open.call_args.args[0]
        self.assertEqual(models, ["gemini-test"])
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")

    def test_api_key_rejects_header_control_characters(self):
        with self.assertRaisesRegex(ApiError, "invalid characters"):
            GoogleApiClient("key\r\nInjected: value")

    def test_api_key_is_redacted_from_http_error_messages(self):
        client = GoogleApiClient("secret-key")
        error = urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"bad secret-key"}}'),
        )
        message = client._http_error_message(error)
        self.assertEqual(message, "bad [redacted]")

    def test_custom_model_id_validation_rejects_path_and_control_characters(self):
        self.assertEqual(validate_model_id("gemini-custom_1.0"), "gemini-custom_1.0")
        for identifier in ("models/gemini-test", "gemini test", "gemini\ud800"):
            with self.subTest(identifier=repr(identifier)):
                with self.assertRaises(ApiError):
                    validate_model_id(identifier)

    def test_invalid_session_model_id_falls_back_to_default(self):
        normalized = SessionStore.normalize({"model": "models/gemini-test"})
        self.assertEqual(normalized["model"], "gemini-2.5-flash")

    def test_api_redirects_are_disabled(self):
        handler = NoRedirectHandler()
        self.assertIsNone(handler.redirect_request(None, None, 302, "Found", {}, "https://example.com"))

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_oversized_request_context_is_rejected_before_network_io(self, mocked_open):
        with patch.object(app, "MAX_API_REQUEST_BYTES", 100):
            with self.assertRaisesRegex(ApiError, "request safety limit"):
                GoogleApiClient("test-key").generate(
                    MODEL_BY_ID["gemini-2.5-flash"],
                    [{"role": "user", "content": "x" * 200}],
                    "",
                )
        mocked_open.assert_not_called()

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_non_object_api_response_is_reported_cleanly(self, mocked_open):
        mocked_open.return_value = FakeResponse(["unexpected"])
        with self.assertRaisesRegex(ApiError, "unexpected JSON"):
            GoogleApiClient("test-key").list_models()

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_oversized_api_response_is_rejected(self, mocked_open):
        mocked_open.return_value = FakeResponse("x" * MAX_API_RESPONSE_BYTES)
        with self.assertRaisesRegex(ApiError, "safety limit"):
            GoogleApiClient("test-key").list_models()

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_oversized_stream_event_is_rejected(self, mocked_open):
        mocked_open.return_value = FakeStreamResponse(
            [{"candidates": [{"content": {"parts": [{"text": "x" * MAX_STREAM_EVENT_BYTES}]}}]}]
        )
        with self.assertRaisesRegex(ApiError, "streaming event exceeded"):
            GoogleApiClient("test-key").generate_stream(
                MODEL_BY_ID["gemini-2.5-flash"],
                [{"role": "user", "content": "Hi"}],
                "",
                lambda _chunk: None,
            )

    def test_model_not_found_error_is_concise(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            message = tui.friendly_error(
                ApiError("HTTP 404: models/gemini-3-flash-preview is not found for API version v1beta"),
                MODEL_BY_ID["gemini-3-flash-preview"],
            )
        self.assertIn("gemini-3-flash-preview is unavailable", message)
        self.assertIn("/model", message)

    def test_large_stream_chunk_is_split_for_progressive_display(self):
        self.assertEqual(Tui.display_fragments("abcdefghijklmnopq", width=8), ["abcdefgh", "ijklmnop", "q"])

    def test_normal_stream_chunks_render_character_by_character(self):
        self.assertEqual(Tui.display_fragments("stream"), ["s", "t", "r", "e", "a", "m"])

    def test_terminal_response_removes_markdown_styling(self):
        response = Tui.plain_terminal_response("# Title\n\n**Bold** and *italic* with `code`.\n\n> note")
        self.assertEqual(response, "Title\n\nBold and italic with code.\n\nnote")

    def test_escape_and_curses_exit_close_menus(self):
        self.assertTrue(Tui.is_cancel_key("\x1b"))
        self.assertTrue(Tui.is_cancel_key(curses.KEY_EXIT))

    def test_stream_chunk_repaint_count_is_bounded(self):
        fragments = Tui.display_fragments("x" * 10000)
        self.assertLessEqual(len(fragments), MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK)

    def test_streaming_setting_defaults_to_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
        self.assertTrue(tui.streaming_enabled)

    def test_custom_model_id_uses_gemini_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save({"model": "gemini-my-custom-id", "system_instruction": "", "history": []})
            tui = Tui("", store)
        self.assertEqual(tui.model.identifier, "gemini-my-custom-id")
        self.assertEqual(tui.model.protocol, "gemini")

    def test_custom_model_selection_validates_and_persists_the_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "prompt_dialog", return_value="gemini-3.7-flash"):
                selected = tui.select_custom_model(object())
        self.assertTrue(selected)
        self.assertEqual(tui.data["model"], "gemini-3.7-flash")

    def test_session_store_repairs_invalid_fields_and_history_items(self):
        normalized = SessionStore.normalize(
            {
                "model": " ",
                "system_instruction": 123,
                "settings": {"streaming": "yes"},
                "history": [
                    {"role": "user", "content": "hello", "model_id": "ignored"},
                    {"role": "model", "content": "hi", "model_id": "gemini-test"},
                    {"role": "assistant", "content": "wrong role"},
                    {"role": "user", "content": 42},
                    "not a message",
                ],
            }
        )
        self.assertEqual(normalized["model"], "gemini-2.5-flash")
        self.assertEqual(normalized["system_instruction"], DEFAULT_SYSTEM_INSTRUCTION)
        self.assertTrue(normalized["settings"]["streaming"])
        self.assertFalse(normalized["settings"]["system_instruction_configured"])
        self.assertEqual(normalized["settings"]["theme"], DEFAULT_THEME_ID)
        self.assertEqual(
            normalized["history"],
            [
                {"role": "user", "content": "hello"},
                {"role": "model", "content": "hi", "model_id": "gemini-test"},
            ],
        )

    def test_explicitly_cleared_system_instruction_stays_cleared(self):
        normalized = SessionStore.normalize(
            {"system_instruction": "", "settings": {"system_instruction_configured": True}}
        )
        self.assertEqual(normalized["system_instruction"], "")
        self.assertTrue(normalized["settings"]["system_instruction_configured"])

    @unittest.skipIf(os.name == "nt", "Windows chmod has different permission semantics")
    def test_session_store_uses_private_file_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save(SessionStore.normalize({}))
            mode = store.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    @unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "Symlink behavior is platform-specific")
    def test_session_store_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps({"model": "attacker-model"}), encoding="utf-8")
            state_dir = root / "state"
            state_dir.mkdir()
            (state_dir / "session.json").symlink_to(target)
            loaded = SessionStore(state_dir).load()
        self.assertEqual(loaded["model"], "gemini-2.5-flash")

    @unittest.skipIf(os.name == "nt" or not hasattr(os, "symlink"), "Symlink behavior is platform-specific")
    def test_session_save_does_not_follow_predictable_temp_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            target.write_text("do not overwrite", encoding="utf-8")
            store = SessionStore(root / "state")
            store.path.parent.mkdir()
            store.path.with_suffix(".tmp").symlink_to(target)
            store.save(SessionStore.normalize({"model": "gemini-test"}))
            target_contents = target.read_text(encoding="utf-8")
        self.assertEqual(target_contents, "do not overwrite")

    def test_oversized_session_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            with patch.object(app, "MAX_SESSION_BYTES", 64):
                store.path.write_text(json.dumps({"model": "x" * 100}), encoding="utf-8")
                loaded = store.load()
        self.assertEqual(loaded["model"], "gemini-2.5-flash")

    def test_terminal_control_characters_are_removed(self):
        unsafe = "before\x1b[31mred\x1b[0m\u202eafter\x00\ud800"
        cleaned = Tui.sanitize_display(unsafe)
        self.assertNotIn("\x1b", cleaned)
        self.assertNotIn("\u202e", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\ud800", cleaned)
        self.assertIn("red", cleaned)

    def test_history_window_clamps_and_scrolls_from_latest(self):
        lines = [(str(index), index) for index in range(10)]
        visible, offset, maximum = Tui.history_window(lines, height=4, offset=3)
        self.assertEqual([line for line, _ in visible], ["3", "4", "5", "6"])
        self.assertEqual(offset, 3)
        self.assertEqual(maximum, 6)

    def test_transcript_reader_preserves_full_message_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.data["history"] = [
                {"role": "user", "content": "first line\nsecond line"},
                {"role": "model", "content": "reply", "model_id": "gemini-2.5-flash"},
            ]
        self.assertEqual(
            tui.transcript_lines(),
            ["YOU", "first line", "second line", "", "GEMINI-2.5-FLASH", "reply", ""],
        )

    def test_existing_assistant_messages_are_normalized_for_terminal_display(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            store.save({"history": [{"role": "model", "content": "**Old bold reply**"}]})
            tui = Tui("", store)
        self.assertEqual(tui.data["history"][0]["content"], "Old bold reply")

    def test_wrap_content_preserves_explicit_blank_lines(self):
        self.assertEqual(Tui.wrap_content("first\n\nsecond", 20), ["first", "", "second"])

    def test_prompt_history_avoids_adjacent_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.remember_prompt("hello")
            tui.remember_prompt("hello")
            tui.remember_prompt("different")
        self.assertEqual(tui.prompt_history, ["hello", "different"])

    def test_up_arrow_scrolls_chat_and_ctrl_p_recalls_a_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.data["history"] = [
                {"role": "user", "content": "line {}".format(index)} for index in range(30)
            ]
            tui.prompt_history = ["earlier prompt"]
            tui.read_line(FakeScreen([curses.KEY_UP, "\x03"]), "You > ")
            self.assertGreater(tui.scroll_offset, 0)
            recalled = tui.read_line(FakeScreen(["\x10", "\r"]), "You > ")
        self.assertEqual(recalled, "earlier prompt")

    def test_retry_reuses_the_last_failed_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.last_failed_prompt = "try this again"
            with patch.object(tui, "handle_input") as handle_input:
                tui.retry_last(object())
        handle_input.assert_called_once_with(unittest.mock.ANY, "try this again")

    def test_command_palette_lists_every_supported_user_command(self):
        commands = {command for command, _ in SLASH_COMMANDS}
        self.assertTrue(
            {"/clear", "/model", "/thinking", "/retry", "/permissions", "/restart"}.issubset(commands)
        )
        self.assertFalse({"/new", "/models", "/transcript", "/stop", "/quit", "/help", "/system", "/settings"} & commands)

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open", side_effect=TimeoutError())
    def test_timeout_error_is_concise(self, _mocked_open):
        client = GoogleApiClient("test-key")
        with self.assertRaisesRegex(ApiError, "Request timed out"):
            client.generate(MODEL_BY_ID["gemini-2.5-flash"], [{"role": "user", "content": "Hi"}], "")

    def test_terminal_tools_are_limited_to_supported_models(self):
        for identifier in ("gemini-flash-latest", "gemini-4-argon", "gemini-4-flash", "gemini-5.0-flash",
                           "gemini-3.8-flash", "gemini-3.9-flash", "gemini-3.10-flash"):
            self.assertTrue(model_supports_terminal_tools(identifier), identifier)
        for identifier in ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash",
                           "gemini-1.5-flash", "chat-bison-001", "gemini-3.8-flash-tts", "gemini-4-flash-lite"):
            self.assertFalse(model_supports_terminal_tools(identifier), identifier)

    def test_terminal_access_defaults_off_and_requires_a_supported_model(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(SessionStore.normalize({})["settings"]["terminal_tools"])
            self.assertFalse(SessionStore.normalize({"settings": {"terminal_tools": "yes"}})["settings"]["terminal_tools"])
            tui = Tui("", SessionStore(Path(directory)))
            tui.data["settings"]["terminal_tools"] = True
            tui.data["model"] = "gemini-3.7-flash"
            self.assertFalse(tui.terminal_tools_active)
            tui.data["model"] = "gemini-4-argon"
            self.assertTrue(tui.terminal_tools_active)
            tui.data["settings"]["terminal_tools"] = False
            self.assertFalse(tui.terminal_tools_active)

    def test_shell_command_captures_output_and_hides_api_keys(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret-key", "GOOGLE_API_KEY": "secret-key"}):
            result = run_shell_command('echo "hi $GEMINI_API_KEY$GOOGLE_API_KEY"')
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["output"].strip(), "hi")
        self.assertNotIn("secret-key", result["output"])

    def test_shell_command_times_out_and_truncates_output(self):
        slow = run_shell_command("sleep 5", timeout=1)
        self.assertTrue(slow["timed_out"])
        big = run_shell_command("head -c 100000 /dev/zero | tr '\\0' 'a'")
        self.assertTrue(big["output_truncated"])
        self.assertEqual(big["output"][: app.MAX_TOOL_OUTPUT_CHARS], "a" * app.MAX_TOOL_OUTPUT_CHARS)
        self.assertEqual(
            big["output"][app.MAX_TOOL_OUTPUT_CHARS :],
            "\n[truncated at 8000 characters; about {} characters omitted]".format(100000 - app.MAX_TOOL_OUTPUT_CHARS),
        )
        self.assertNotIn("truncated", run_shell_command("echo short")["output"])

    def test_shell_command_can_be_stopped_while_running(self):
        stopped = run_shell_command("sleep 5", timeout=30, should_stop=lambda: True)
        self.assertTrue(stopped["stopped_by_user"])
        self.assertNotIn("timed_out", stopped)

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_tool_loop_runs_command_and_returns_final_text(self, mocked_open):
        call_part = {"functionCall": {"name": "run_terminal_command", "args": {"command": "ls"}, "id": "c1"}, "thoughtSignature": "sig"}
        mocked_open.side_effect = [
            FakeResponse({"candidates": [{"content": {"role": "model", "parts": [call_part]}}]}),
            FakeResponse({"candidates": [{"content": {"parts": [{"text": "Two files."}]}}]}),
        ]
        seen = []

        def run(command):
            seen.append(command)
            return {"exit_code": 0, "output": "a\nb"}, "[terminal] ran: ls (exit 0)"

        text, log = GoogleApiClient("test-key").generate_with_tools(
            MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "list files"}], "", run
        )
        self.assertEqual(text, "Two files.")
        self.assertEqual(seen, ["ls"])
        self.assertEqual(log, ["[terminal] ran: ls (exit 0)"])
        first = json.loads(mocked_open.call_args_list[0].args[0].data)
        second = json.loads(mocked_open.call_args_list[1].args[0].data)
        self.assertEqual(first["tools"][0]["functionDeclarations"][0]["name"], "run_terminal_command")
        self.assertEqual(second["contents"][1]["parts"][0]["thoughtSignature"], "sig")
        response = second["contents"][2]["parts"][0]["functionResponse"]
        self.assertEqual(response["id"], "c1")
        self.assertEqual(response["response"]["output"], "a\nb")

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_tool_loop_allows_ten_results_warns_on_eighth_and_ninth_then_forces_text(self, mocked_open):
        call = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "run_terminal_command", "args": {"command": "ls"}}}]}}]}
        final = {"candidates": [{"content": {"parts": [{"text": "Done."}]}}]}
        mocked_open.side_effect = [FakeResponse(call)] * 10 + [FakeResponse(final)]
        ran = []

        def run(command):
            ran.append(command)
            return {"exit_code": 0, "output": "out"}, "[terminal] ran"

        text, log = GoogleApiClient("test-key").generate_with_tools(
            MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "loop"}], "", run
        )
        self.assertEqual(text, "Done.")
        self.assertEqual(len(ran), 10)
        bodies = [json.loads(call.args[0].data) for call in mocked_open.call_args_list]
        outputs = [
            body["contents"][-1]["parts"][0]["functionResponse"]["response"]["output"] for body in bodies[1:]
        ]
        for index, output in enumerate(outputs, start=1):
            if index == 8:
                self.assertTrue(output.endswith("[You have 2 more terminal accesses in the current turn.]"))
            elif index == 9:
                self.assertTrue(output.endswith("[You have 1 more terminal access in the current turn.]"))
            elif index == 10:
                self.assertTrue(output.endswith("[That was your last terminal access in the current turn.]"))
            else:
                self.assertEqual(output, "out")
        self.assertNotIn("toolConfig", bodies[9])
        self.assertEqual(bodies[10]["toolConfig"]["functionCallingConfig"]["mode"], "NONE")

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_tool_loop_errors_if_model_ignores_the_limit(self, mocked_open):
        call = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "run_terminal_command", "args": {"command": "ls"}}}]}}]}
        mocked_open.side_effect = lambda *_a, **_k: FakeResponse(call)
        with self.assertRaisesRegex(ApiError, "kept asking for terminal access"):
            GoogleApiClient("test-key").generate_with_tools(
                MODEL_BY_ID["gemini-4-argon"],
                [{"role": "user", "content": "loop"}],
                "",
                lambda command: ({"exit_code": 0, "output": ""}, "[terminal] ran"),
            )

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_tool_loop_stops_when_the_user_stops(self, mocked_open):
        call = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "run_terminal_command", "args": {"command": "ls"}}}]}}]}
        mocked_open.side_effect = lambda *_a, **_k: FakeResponse(call)
        stop = {"value": False}

        def run(command):
            stop["value"] = True
            return {"exit_code": 0, "output": ""}, "[terminal] ran"

        text, log = GoogleApiClient("test-key").generate_with_tools(
            MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "go"}], "", run, lambda: stop["value"]
        )
        self.assertIn("Stopped by you", text)
        self.assertEqual(mocked_open.call_count, 1)

    def test_esc_stops_a_terminal_turn_and_other_keys_are_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "draw_chat"):
                self.assertFalse(tui.poll_stop_request(FakeScreen(["a", "/", "\r", None])))
                self.assertTrue(tui.poll_stop_request(FakeScreen(["x", "\x1b", None])))
            self.assertTrue(tui.stop_requested)

    def test_denied_command_is_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "confirm_command", return_value="no"), patch.object(app, "run_shell_command") as runner:
                result, line = tui.run_approved_command(FakeScreen([]), "rm -rf /tmp/x")
            runner.assert_not_called()
            self.assertIn("denied", result["error"])
            self.assertTrue(line.startswith("[terminal] denied"))

    def test_commands_are_labeled_read_creation_write_or_removal(self):
        expected = {
            "ls -la": "READ",
            "cat a.txt | grep foo": "READ",
            'grep -E "a|b" notes.txt': "READ",
            "git status": "READ",
            "sed s/a/b/ notes.txt": "READ",
            "mkdir build-output": "CREATION",
            "curl -o archive.zip https://example.com/a.zip": "CREATION",
            "pip install requests": "CREATION",
            "mv a b": "WRITE",
            "sed -i s/a/b/ notes.txt": "WRITE",
            "git commit -m x": "WRITE",
            "python3 -m unittest": "WRITE",
            "rm notes.txt": "REMOVAL",
            "find . -name '*.pyc' -delete": "REMOVAL",
            "echo $(rm x)": "REMOVAL",
            "bash -c 'rm -rf x'": "REMOVAL",
            "pwd; ls && rm x": "REMOVAL",
        }
        for command, label in expected.items():
            self.assertEqual(classify_command(command).primary, label, command)

    def test_redirects_distinguish_creation_from_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                Path("existing.txt").write_text("x")
                self.assertEqual(classify_command("echo hi > brand-new.txt").primary, "CREATION")
                self.assertEqual(classify_command("echo hi > existing.txt").primary, "WRITE")
                self.assertEqual(classify_command("ls 2>&1 > /dev/null").primary, "READ")
            finally:
                os.chdir(old)

    def test_risky_commands_are_flagged(self):
        for command in (
            "sudo ls",
            "git push origin main",
            "git reset --hard HEAD~1",
            "curl -s https://example.com/install | sh",
            "python3 -c 'print(1)'",
            "echo hi > /etc/hosts",
            "chmod -R 777 .",
            "pip install requests",
            "kill 123",
        ):
            self.assertTrue(classify_command(command).risks, command)
        for command in ("ls -la", "git status", "mkdir build-output", "python3 -m unittest", "mv a b"):
            self.assertFalse(classify_command(command).risks, command)

    def test_allow_all_safe_skips_prompts_except_for_risky_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.command_permissions = "safe"
            with patch.object(tui, "confirm_command", return_value="no") as confirm, patch.object(
                app, "run_shell_command", return_value={"exit_code": 0, "output": ""}
            ):
                _result, line = tui.run_approved_command(FakeScreen([]), "ls")
                confirm.assert_not_called()
                self.assertIn("[READ]", line)
                result, line = tui.run_approved_command(FakeScreen([]), "rm notes.txt")
                confirm.assert_called_once()
                self.assertIn("denied", result["error"])
                self.assertIn("[REMOVAL]", line)

    def test_allow_all_answer_on_a_risky_command_does_not_enable_auto_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "confirm_command", return_value="all"), patch.object(
                app, "run_shell_command", return_value={"exit_code": 0, "output": ""}
            ):
                tui.run_approved_command(FakeScreen([]), "rm notes.txt")
            self.assertEqual(tui.command_permissions, "ask")

    def test_removal_asks_every_time_without_warning_text(self):
        review = classify_command("rm notes.txt")
        self.assertEqual(review.primary, "REMOVAL")
        self.assertTrue(review.needs_confirmation)
        self.assertEqual(review.risks, [])
        self.assertFalse(classify_command("ls").needs_confirmation)

    def test_permission_modes_control_prompting(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            fake = {"exit_code": 0, "output": ""}
            with patch.object(tui, "confirm_command", return_value="yes") as confirm, patch.object(
                app, "run_shell_command", return_value=fake
            ):
                tui.command_permissions = "ask"
                tui.run_approved_command(FakeScreen([]), "ls")
                self.assertEqual(confirm.call_count, 1)
                tui.command_permissions = "safe"
                tui.run_approved_command(FakeScreen([]), "ls")
                self.assertEqual(confirm.call_count, 1)
                tui.run_approved_command(FakeScreen([]), "rm notes.txt")
                self.assertEqual(confirm.call_count, 2)
                tui.command_permissions = "all"
                tui.run_approved_command(FakeScreen([]), "sudo rm -rf x")
                self.assertEqual(confirm.call_count, 2)

    def test_allow_all_answer_on_a_safe_command_switches_to_allow_all_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "confirm_command", return_value="all"), patch.object(
                app, "run_shell_command", return_value={"exit_code": 0, "output": ""}
            ):
                tui.run_approved_command(FakeScreen([]), "ls")
            self.assertEqual(tui.command_permissions, "safe")

    def test_permissions_command_requires_terminal_access(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "select_command_permissions") as picker:
                tui.handle_command(FakeScreen([]), "/permissions")
                picker.assert_not_called()
                self.assertIn("Terminal access", tui.status)
                tui.data["settings"]["terminal_tools"] = True
                tui.handle_command(FakeScreen([]), "/permissions")
                picker.assert_called_once()

    def test_editor_wraps_long_lines_and_tracks_the_cursor(self):
        rows = Tui.editor_rows("abcdefghij\nxy", 4)
        self.assertEqual(rows, [(0, 4), (4, 8), (8, 10), (11, 13)])
        self.assertEqual(Tui.editor_cursor(rows, 0), (0, 0))
        self.assertEqual(Tui.editor_cursor(rows, 4), (1, 0))
        self.assertEqual(Tui.editor_cursor(rows, 10), (2, 2))
        self.assertEqual(Tui.editor_cursor(rows, 13), (3, 2))
        self.assertEqual(Tui.editor_rows("", 4), [(0, 0)])

    def test_multiline_editor_inserts_newlines_and_saves_with_ctrl_d(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            keys = list("ab") + ["\n"] + list("cd") + ["\x04"]
            with patch.object(Tui, "show_cursor"):
                result = tui.edit_text_dialog(FakeScreen(keys), "t", "")
            self.assertEqual(result, "ab\ncd")
            with patch.object(Tui, "show_cursor"):
                self.assertIsNone(tui.edit_text_dialog(FakeScreen(["x", "\x1b"]), "t", "keep"))

    def test_header_shows_model_turns_and_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.data["model"] = "gemini-4-argon"
            tui.data["history"] = [
                {"role": "user", "content": "a"},
                {"role": "model", "content": "b", "model_id": "gemini-4-argon"},
                {"role": "user", "content": "c"},
            ]
            tui.data["usage"] = {"input": 1234, "output": 56, "cost": 0.00432, "unpriced": False}
            model_id, rest = tui.header_parts()
            self.assertEqual(model_id, "gemini-4-argon")
            self.assertIn("2 turns", rest)
            self.assertIn("1,234 in / 56 out tokens", rest)
            self.assertTrue(rest.endswith("~$0.0043"))
            for old in ("GEMINI LEGACY", "READY", "local session"):
                self.assertNotIn(old, model_id + rest)

    def test_usage_is_normalized_persisted_and_cleared(self):
        self.assertEqual(SessionStore.normalize({})["usage"], {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False})
        bad = SessionStore.normalize({"usage": {"input": -5, "output": "x"}})["usage"]
        self.assertEqual(bad, {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False})
        self.assertEqual(
            SessionStore.normalize({"usage": {"input": 7, "output": 9, "cost": 0.5, "unpriced": True}})["usage"],
            {"input": 7, "output": 9, "context": 0, "cost": 0.5, "unpriced": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.data["usage"] = {"input": 7, "output": 9, "context": 5, "cost": 1.0, "unpriced": True}
            tui.clear_context()
            self.assertEqual(tui.data["usage"], {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False})

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_client_records_tokens_and_the_model_version_the_api_reports(self, mocked_open):
        mocked_open.return_value = FakeResponse(
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "thoughtsTokenCount": 6},
                "modelVersion": "gemini-3.8-flash",
            }
        )
        client = GoogleApiClient("test-key")
        client.generate(MODEL_BY_ID["gemini-flash-latest"], [{"role": "user", "content": "Hi"}], "")
        self.assertEqual((client.usage_input, client.usage_output), (10, 10))
        self.assertEqual(client.model_version, "gemini-3.8-flash")

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_stream_uses_the_latest_running_token_totals(self, mocked_open):
        events = [
            {"candidates": [{"content": {"parts": [{"text": "a"}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1}, "modelVersion": "m-1"},
            {"candidates": [{"content": {"parts": [{"text": "b"}]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}, "modelVersion": "m-1"},
        ]
        mocked_open.return_value = FakeStreamResponse(events)
        client = GoogleApiClient("test-key")
        client.generate_stream(MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "Hi"}], "", lambda _c: None)
        self.assertEqual((client.usage_input, client.usage_output, client.model_version), (10, 2, "m-1"))

    def test_slash_commands_that_moved_to_the_menu_are_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            for command in ("/quit", "/help", "/system", "/settings", "/stop", "/new", "/models", "/transcript"):
                tui.handle_command(FakeScreen([]), command)
                self.assertIn("Unknown command", tui.status, command)
                self.assertTrue(tui.running)

    def test_action_menu_has_no_model_retry_or_restart_entries(self):
        source = Path(app.__file__).read_text()
        menu = source[source.index("def open_action_menu") : source.index("def open_settings")]
        for removed in ("Choose model", "Retry failed prompt", "Restart interface", "Clear context"):
            self.assertNotIn(removed, menu)
        for kept in ("Settings", "View transcript", "Help", "Exit"):
            self.assertIn(kept, menu)

    SHIFT_ENTER = ["\x1b", "[", "1", "3", ";", "2", "u", None]

    def test_prompt_grows_with_newlines_and_scrolls_after_five_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            keys = []
            for letter in "abcdefg":
                keys += [letter] + (self.SHIFT_ENTER if letter != "g" else [])
            keys += ["\r"]
            result = tui.read_line(FakeScreen(keys), "You > ")
            self.assertEqual(result, "a\nb\nc\nd\ne\nf\ng")
            self.assertEqual(tui.input_rows, 5)

    def test_only_shift_enter_adds_a_new_line(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            self.assertEqual(tui.read_line(FakeScreen(["x"] + self.SHIFT_ENTER + ["y", "\r"]), "You > "), "x\ny")
            modify_other_keys = ["x", "\x1b", "[", "2", "7", ";", "2", ";", "1", "3", "~", None, "y", "\r"]
            self.assertEqual(tui.read_line(FakeScreen(modify_other_keys), "You > "), "x\ny")
            # Ctrl+J (a bare newline byte), Alt+Enter, and a lone Esc do nothing.
            self.assertEqual(tui.read_line(FakeScreen(["x", "\n", "y", "\r"]), "You > "), "xy")
            self.assertEqual(tui.read_line(FakeScreen(["x", "\x1b", "\r", None, "y", "\r"]), "You > "), "xy")
            self.assertEqual(tui.read_line(FakeScreen(["x", "\x1b", None, "y", "\r"]), "You > "), "xy")

    def test_cancelling_the_slash_menu_redraws_the_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            with patch.object(tui, "open_command_palette", return_value=None), patch.object(tui, "draw_chat") as draw:
                tui.read_line(FakeScreen(["/", "a", "\r"]), "You > ")
            self.assertTrue(draw.called)

    def test_cost_estimates_use_dated_prices_and_unknown_models_have_none(self):
        before, after = date(2026, 10, 14), date(2026, 12, 1)
        self.assertAlmostEqual(estimate_cost("gemini-4-argon", 1_000_000, 1_000_000, before), 12.0)
        self.assertAlmostEqual(estimate_cost("gemini-4-argon", 1_000_000, 1_000_000, after), 24.0)
        self.assertAlmostEqual(estimate_cost("gemini-3.8-flash", 2_000_000, 1_000_000, before), 5.25)
        self.assertAlmostEqual(estimate_cost("gemini-2.5-flash", 1_000_000, 0, before), 0.30)
        self.assertAlmostEqual(estimate_cost("gemini-3.8-flash-001", 1_000_000, 0, before), 0.75)
        self.assertEqual(estimate_cost("gemini-mock-model", 100, 100), 0.0)
        self.assertIsNone(estimate_cost("gemini-3.7-flash", 10, 10, before))
        self.assertIsNone(estimate_cost("my-custom-model", 10, 10, before))

    def test_usage_cost_accumulates_and_flags_unpriced_models(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            client = GoogleApiClient("k")
            client.usage_input, client.usage_output = 1_000_000, 100_000
            client.model_version = "gemini-2.5-flash"
            tui.add_usage(client, "gemini-flash-latest")
            self.assertAlmostEqual(tui.data["usage"]["cost"], 0.55)
            self.assertFalse(tui.data["usage"]["unpriced"])
            self.assertTrue(tui.format_cost(tui.data["usage"]).startswith("~$0.55"))
            client.model_version = ""
            tui.add_usage(client, "gemini-3.7-flash")
            self.assertTrue(tui.data["usage"]["unpriced"])
            self.assertTrue(tui.format_cost(tui.data["usage"]).endswith("+"))
            self.assertEqual(Tui.format_cost({"cost": 0.0, "unpriced": True}), "cost unknown")

    def test_generation_config_only_sends_set_values_and_maps_thinking_per_model(self):
        self.assertEqual(build_generation_config({}, "gemini-4-argon"), {})
        settings = {"temperature": 0.4, "top_p": 0.9, "top_k": 20, "max_output_tokens": 500, "seed": 7,
                    "presence_penalty": 0.5, "frequency_penalty": -0.5, "stop_sequences": ["END"],
                    "thinking_level": "high", "thinking_budget": 2048}
        argon = build_generation_config(settings, "gemini-4-argon")
        self.assertEqual(argon["thinkingConfig"], {"thinkingLevel": "HIGH"})
        self.assertEqual((argon["temperature"], argon["topP"], argon["topK"], argon["maxOutputTokens"], argon["seed"]), (0.4, 0.9, 20, 500, 7))
        self.assertEqual((argon["presencePenalty"], argon["frequencyPenalty"], argon["stopSequences"]), (0.5, -0.5, ["END"]))
        self.assertEqual(build_generation_config(settings, "gemini-2.5-flash")["thinkingConfig"], {"thinkingBudget": 2048})
        # A level the model does not accept is left out rather than causing an API error.
        self.assertNotIn("thinkingConfig", build_generation_config({"thinking_level": "minimal"}, "gemini-4-argon"))
        self.assertEqual(build_generation_config({"thinking_level": "minimal"}, "gemini-3.6-flash")["thinkingConfig"], {"thinkingLevel": "MINIMAL"})
        self.assertEqual(thinking_support("chat-bison-001")[0], "none")

    def test_generation_values_are_validated(self):
        self.assertEqual(parse_generation_value("temperature", "1.5"), 1.5)
        self.assertEqual(parse_generation_value("top_k", "40"), 40)
        self.assertEqual(parse_generation_value("stop_sequences", "a, b ,,c"), ["a", "b", "c"])
        self.assertEqual(parse_generation_value("thinking_budget", "-1"), -1)
        for key, text in (("temperature", "3"), ("temperature", "hot"), ("top_p", "-0.1"), ("top_k", "1.5"),
                          ("max_output_tokens", "0"), ("stop_sequences", "1,2,3,4,5,6"), ("thinking_budget", "99999")):
            with self.assertRaises(ApiError, msg=(key, text)):
                parse_generation_value(key, text)
        cleaned = SessionStore.normalize({"settings": {"generation": {
            "temperature": 9, "top_p": 0.5, "thinking_level": "extreme", "bogus": 1, "seed": True, "stop_sequences": []}}})
        self.assertEqual(cleaned["settings"]["generation"], {"top_p": 0.5})

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_request_body_carries_generation_settings_and_skips_thought_parts(self, mocked_open):
        mocked_open.return_value = FakeResponse({"candidates": [{"content": {"parts": [
            {"text": "secret reasoning", "thought": True}, {"text": "the answer"}]}}]})
        client = GoogleApiClient("k", {"temperature": 0.2, "thinking_level": "low"})
        answer = client.generate(MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "Hi"}], "")
        body = json.loads(mocked_open.call_args.args[0].data)
        self.assertEqual(answer, "the answer")
        self.assertEqual(body["generationConfig"], {"temperature": 0.2, "thinkingConfig": {"thinkingLevel": "LOW"}})
        mocked_open.return_value = FakeResponse({"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
        GoogleApiClient("k").generate(MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "Hi"}], "")
        self.assertNotIn("generationConfig", json.loads(mocked_open.call_args.args[0].data))

    def test_thinking_menu_cycles_levels_edits_values_and_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.data["model"] = "gemini-4-argon"
            with patch.object(Tui, "show_cursor"):
                tui.open_generation_settings(FakeScreen(["\r", "\x1b"]))
                self.assertEqual(tui.data["settings"]["generation"], {"thinking_level": "low"})
                tui.open_generation_settings(FakeScreen(["\r", "\r", "\r", "\x1b"]))  # medium, high, back to default
                self.assertEqual(tui.data["settings"]["generation"], {})
                with patch.object(tui, "prompt_dialog", return_value="0.3"):
                    tui.open_generation_settings(FakeScreen([curses.KEY_DOWN, "\r", "\x1b"]))
                self.assertEqual(tui.data["settings"]["generation"], {"temperature": 0.3})
                tui.open_generation_settings(FakeScreen([curses.KEY_DOWN, "r", "\x1b"]))
                self.assertEqual(tui.data["settings"]["generation"], {})
                tui.data["settings"]["generation"] = {"seed": 5}
                tui.open_generation_settings(FakeScreen([curses.KEY_UP] + [curses.KEY_DOWN] * 20 + ["\r", "\x1b"]))
                self.assertEqual(tui.data["settings"]["generation"], {})

    def test_mock_model_is_in_the_catalog_and_talks_to_the_local_server(self):
        mock = MODEL_BY_ID["gemini-mock-model"]
        self.assertTrue(mock.local_mock)
        self.assertTrue(model_supports_terminal_tools(mock.identifier))
        self.assertFalse(any(model.local_mock for model in MODEL_CATALOG if model is not mock))
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("real-key-should-not-leak", SessionStore(Path(directory)))
            tui.data["model"] = "gemini-mock-model"
            captured = {}

            def fake_open(request, timeout=None):
                captured.setdefault("url", request.full_url)
                captured.setdefault("key", request.get_header("X-goog-api-key"))
                if request.full_url.endswith("/gemini-mock-model"):
                    return FakeResponse({"inputTokenLimit": 123456})
                return FakeResponse({"candidates": [{"content": {"parts": [{"text": "mock says hi"}]}}]})

            with patch.object(app, "start_mock_server") as starter, patch.object(
                app.GOOGLE_API_OPENER, "open", side_effect=fake_open
            ), patch.object(tui, "draw_chat"):
                tui.data["settings"]["streaming"] = False
                tui.handle_input(FakeScreen([]), "hello")
            starter.assert_called_once()
            self.assertTrue(captured["url"].startswith("http://127.0.0.1:8765/v1beta/models/gemini-mock-model:"))
            self.assertEqual(captured["key"], "mock")
            self.assertEqual(tui.context_limits["gemini-mock-model"], 123456)
            self.assertEqual(tui.data["history"][-1]["content"], "mock says hi")
            self.assertIn("mock server", tui.friendly_error(ApiError("Network error: refused"), mock))

    def test_waiting_status_is_plain_with_a_dim_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            tui.status = "Waiting for gemini-4-argon...   " + app.STOP_HINT
            drawn = []

            class Recorder(FakeScreen):
                def addstr(self_inner, row, column, text, style=0):
                    drawn.append((text, style))

            tui.draw_chat(Recorder([]))
            self.assertEqual(app.STOP_HINT, "press esc to stop turn.")
            texts = [text for text, _ in drawn]
            self.assertIn("Waiting for gemini-4-argon...", texts)
            self.assertIn("press esc to stop turn.", texts)
            for text, style in drawn:
                if text == "Waiting for gemini-4-argon...":
                    self.assertFalse(style & (curses.A_BOLD | curses.A_REVERSE))

    def test_header_cost_and_tokens_reset_when_context_is_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            tui.data["history"] = [{"role": "user", "content": "a"}, {"role": "model", "content": "b", "model_id": "m"}]
            tui.data["usage"] = {"input": 5000, "output": 900, "cost": 0.0191, "unpriced": True}
            self.assertTrue(tui.header_parts()[1].endswith("~$0.0191+"))
            tui.handle_command(FakeScreen([]), "/clear")
            rest = tui.header_parts()[1]
            self.assertIn("0 turns", rest)
            self.assertIn("0 in / 0 out tokens", rest)
            self.assertTrue(rest.endswith("~$0.0000"))
            self.assertIn("context 0/1,048,576", rest)
            reloaded = SessionStore(Path(directory)).load()
            self.assertEqual(reloaded["usage"], {"input": 0, "output": 0, "context": 0, "cost": 0.0, "unpriced": False})

    def test_retry_only_shows_in_the_slash_menu_after_a_failed_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            shown = []

            def run_palette():
                class Recorder(FakeScreen):
                    def addstr(self_inner, row, column, text, style=0):
                        shown.append(text)

                shown.clear()
                tui.open_command_palette(Recorder(["\x1b"]))
                return " ".join(shown)

            with patch.object(Tui, "show_cursor"):
                self.assertNotIn("/retry", run_palette())
                tui.last_failed_prompt = "try again"
                self.assertIn("/retry", run_palette())
            for removed in ("/key", "/check"):
                self.assertNotIn(removed, {command for command, _ in SLASH_COMMANDS})

    def test_header_shows_context_tokens_over_the_window_left_of_the_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            tui.data["usage"] = {"input": 9000, "output": 100, "context": 4321, "cost": 0.02, "unpriced": False}
            rest = tui.header_parts()[1]
            self.assertIn("context 4,321/1,048,576   ~$0.0200", rest)
            tui.context_limits[tui.model.identifier] = 200000
            self.assertIn("context 4,321/200,000", tui.header_parts()[1])
            tui.data["model"] = "my-custom-model"
            self.assertIn("context 4,321   ", tui.header_parts()[1])

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open")
    def test_client_reads_the_input_token_limit_and_tracks_the_current_context(self, mocked_open):
        mocked_open.return_value = FakeResponse({"name": "models/gemini-4-argon", "inputTokenLimit": 1048576})
        client = GoogleApiClient("k")
        self.assertEqual(client.model_input_limit("gemini-4-argon"), 1048576)
        self.assertTrue(mocked_open.call_args.args[0].full_url.endswith("/v1beta/models/gemini-4-argon"))
        mocked_open.return_value = FakeResponse({"candidates": [{"content": {"parts": [{"text": "a"}]}}], "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 1}})
        client.generate(MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "Hi"}], "")
        mocked_open.return_value = FakeResponse({"candidates": [{"content": {"parts": [{"text": "b"}]}}], "usageMetadata": {"promptTokenCount": 75, "candidatesTokenCount": 1}})
        client.generate(MODEL_BY_ID["gemini-4-argon"], [{"role": "user", "content": "Hi"}], "")
        self.assertEqual((client.last_prompt_tokens, client.usage_input), (75, 115))

    def test_usage_updates_context_and_lifetime_totals_and_clear_keeps_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            client = GoogleApiClient("k")
            client.usage_input, client.usage_output, client.last_prompt_tokens = 1_000_000, 100_000, 800
            client.model_version = "gemini-2.5-flash"
            tui.add_usage(client, "gemini-flash-latest")
            client.usage_input, client.usage_output, client.last_prompt_tokens, client.model_version = 10, 5, 99, ""
            tui.add_usage(client, "gemini-3.7-flash")
            self.assertEqual(tui.data["usage"]["context"], 99)
            life = tui.data["lifetime"]
            self.assertEqual((life["input"], life["output"]), (1_000_010, 100_005))
            self.assertTrue(life["unpriced"])
            self.assertAlmostEqual(life["models"]["gemini-2.5-flash"]["cost"], 0.55)
            self.assertTrue(life["models"]["gemini-3.7-flash"]["unpriced"])
            tui.clear_context()
            self.assertEqual(tui.data["usage"]["context"], 0)
            self.assertEqual(tui.data["lifetime"]["input"], 1_000_010)
            reloaded = SessionStore(Path(directory)).load()
            self.assertEqual(reloaded["lifetime"]["input"], 1_000_010)
            self.assertIn("gemini-2.5-flash", reloaded["lifetime"]["models"])

    def test_lifetime_stats_are_sanitized_and_shown_in_the_tab_menu(self):
        bad = SessionStore.normalize({"lifetime": {"input": -1, "output": "x", "models": {"": {}, "ok": {"input": 5}, 7: {}}}})
        self.assertEqual(bad["lifetime"]["input"], 0)
        self.assertEqual(list(bad["lifetime"]["models"]), ["ok"])
        self.assertEqual(bad["lifetime"]["models"]["ok"]["input"], 5)
        source = Path(app.__file__).read_text()
        menu = source[source.index("def open_action_menu") : source.index("def open_settings")]
        for item in ("Check availability", "Usage stats"):
            self.assertIn(item, menu)
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            tui.data["lifetime"] = {"input": 1200, "output": 300, "cost": 0.0123, "unpriced": False,
                                    "models": {"gemini-4-argon": {"input": 1200, "output": 300, "cost": 0.0123, "unpriced": False}}}
            with patch.object(tui, "show_message") as show:
                tui.show_usage_stats(FakeScreen([]))
            text = "\n".join(show.call_args.args[2])
            for expected in ("Tokens in:       1,200", "Tokens out:      300", "~$0.0123", "gemini-4-argon", "1,200 in / 300 out"):
                self.assertIn(expected, text)

    def test_api_key_entry_is_gone_and_context_does_not_depend_on_the_key(self):
        source = Path(app.__file__).read_text()
        self.assertNotIn("def configure_api_key", source)
        self.assertNotIn("KEY_F4", source)
        with tempfile.TemporaryDirectory() as directory:
            first = Tui("key-one", SessionStore(Path(directory)))
            first.data["history"] = [{"role": "user", "content": "remember me"}]
            first.data["usage"]["context"] = 10
            first.store.save(first.data)
            second = Tui("key-two", SessionStore(Path(directory)))
            self.assertEqual(second.data["history"], [{"role": "user", "content": "remember me"}])
            self.assertEqual(second.data["usage"]["context"], 10)

    def test_terminal_log_lines_are_dimmed_apart_from_the_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("k", SessionStore(Path(directory)))
            tui.initialize_theme = lambda screen: None
            tui.colors = {"muted": 111, "text": 222, "assistant": 333, "user": 444}
            tui.data["history"] = [
                {"role": "user", "content": "[terminal] looks like a log but is the user's"},
                {"role": "model", "model_id": "m",
                 "content": "[terminal] ran [READ]: ls (exit 0)\n[terminal] ran [WRITE]: mkdir x (exit 0)\n\nDone. [terminal] mentioned here"},
            ]
            lines = tui.render_history(80)
            styled = {text.strip(): style for text, style in lines}
            self.assertEqual(styled["[terminal] ran [READ]: ls (exit 0)"], 111)
            self.assertEqual(styled["[terminal] ran [WRITE]: mkdir x (exit 0)"], 111)
            self.assertEqual(styled["Done. [terminal] mentioned here"], 222)
            self.assertTrue(any(text.strip().startswith("YOU  [terminal] looks like") and style != 111 for text, style in lines))
            long_log = {"role": "model", "model_id": "m", "content": "[terminal] ran [READ]: " + "x" * 200 + " (exit 0)\n\nok"}
            tui.data["history"] = [long_log]
            wrapped = [style for text, style in tui.render_history(60) if text.strip() and text.strip() != "M" and text.strip() != "ok"]
            self.assertTrue(all(style == 111 for style in wrapped[1:]))
            self.assertEqual(Tui.split_terminal_lines("plain text"), ("", "plain text"))

    def test_command_type_is_only_the_most_severe_and_common_read_tools_are_read(self):
        review = classify_command("sw_vers && system_profiler SPHardwareDataType && df -h / && uptime")
        self.assertEqual(review.categories, ["READ"])
        mixed = classify_command("ls && mkdir out && mv a b")
        self.assertEqual(mixed.primary, "WRITE")
        source = Path(app.__file__).read_text()
        self.assertNotIn("(also:", source)
        unknown = classify_command("somecustomtool --go")
        self.assertEqual(unknown.primary, "WRITE")
        self.assertFalse(unknown.needs_confirmation)
        self.assertIn("Unrecognized command", " ".join(unknown.notes))


if __name__ == "__main__":
    unittest.main()
