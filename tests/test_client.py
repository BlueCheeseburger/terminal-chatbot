import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import gemini_legacy_tui as app
from gemini_legacy_tui import (
    ApiError,
    DEFAULT_SYSTEM_INSTRUCTION,
    GoogleApiClient,
    MAX_API_RESPONSE_BYTES,
    MAX_STREAM_DISPLAY_UPDATES_PER_CHUNK,
    MAX_STREAM_EVENT_BYTES,
    MODEL_BY_ID,
    MODEL_CATALOG,
    NoRedirectHandler,
    SLASH_COMMANDS,
    SessionStore,
    Tui,
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


class ClientTests(unittest.TestCase):
    def test_catalog_has_only_the_requested_palm_chat_model(self):
        palm = [model.identifier for model in MODEL_CATALOG if model.protocol == "palm"]
        self.assertEqual(palm, ["chat-bison-001"])
        identifiers = [model.identifier for model in MODEL_CATALOG]
        self.assertFalse(any("lite" in model or "pro" in model or "text-bison" in model for model in identifiers))
        self.assertFalse(any("image" in model for model in identifiers))

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
        self.assertNotIn("key=", request.full_url)
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
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
                ApiError("HTTP 404: models/gemini-1.5-flash is not found for API version v1beta"),
                MODEL_BY_ID["gemini-1.5-flash"],
            )
        self.assertIn("gemini-1.5-flash is unavailable", message)
        self.assertIn("Ctrl+L", message)

    def test_large_stream_chunk_is_split_for_progressive_display(self):
        self.assertEqual(Tui.display_fragments("abcdefghijklmnopq", width=8), ["abcdefgh", "ijklmnop", "q"])

    def test_normal_stream_chunks_render_character_by_character(self):
        self.assertEqual(Tui.display_fragments("stream"), ["s", "t", "r", "e", "a", "m"])

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

    def test_wrap_content_preserves_explicit_blank_lines(self):
        self.assertEqual(Tui.wrap_content("first\n\nsecond", 20), ["first", "", "second"])

    def test_prompt_history_avoids_adjacent_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            tui = Tui("", SessionStore(Path(directory)))
            tui.remember_prompt("hello")
            tui.remember_prompt("hello")
            tui.remember_prompt("different")
        self.assertEqual(tui.prompt_history, ["hello", "different"])

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
            {"/clear", "/new", "/model", "/models", "/settings", "/transcript", "/retry", "/system", "/key", "/check", "/restart", "/help", "/quit"}.issubset(commands)
        )

    @patch("gemini_legacy_tui.GOOGLE_API_OPENER.open", side_effect=TimeoutError())
    def test_timeout_error_is_concise(self, _mocked_open):
        client = GoogleApiClient("test-key")
        with self.assertRaisesRegex(ApiError, "Request timed out"):
            client.generate(MODEL_BY_ID["gemini-2.5-flash"], [{"role": "user", "content": "Hi"}], "")


if __name__ == "__main__":
    unittest.main()
