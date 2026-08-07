import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemini_legacy_tui import ApiError, GoogleApiClient, MODEL_BY_ID, MODEL_CATALOG, SessionStore, Tui


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeStreamResponse:
    def __init__(self, events):
        self.events = ["data: {}\n".format(json.dumps(event)).encode("utf-8") for event in events]

    def __iter__(self):
        return iter(self.events)

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

    @patch("urllib.request.urlopen")
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
        self.assertEqual(body["contents"][0]["role"], "user")
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "Be concise.")

    @patch("urllib.request.urlopen")
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

    @patch("urllib.request.urlopen")
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


if __name__ == "__main__":
    unittest.main()
