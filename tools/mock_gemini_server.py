#!/usr/bin/env python3
"""A fake Gemini API on localhost for testing the TUI without spending API credits.

Run it, then start the app against it:

    python3 tools/mock_gemini_server.py
    GEMINI_API_KEY=mock python3 gemini_legacy_tui.py --api-base-url http://127.0.0.1:8765

To try the LM Studio provider without LM Studio, launch the app with --lmstudio-url http://127.0.0.1:8765
and pick one of the "mock-" models under LM Studio (local) in /model.

Type "run <command>" (for example "run ls -la" or "run rm notes.txt") to make the fake model
ask to run that command through the terminal tool. Anything else gets an echo reply.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def last_user_text(contents):
    for item in reversed(contents):
        if item.get("role") == "user":
            for part in item.get("parts", []):
                if isinstance(part.get("text"), str):
                    return part["text"]
    return ""


def build_reply(body):
    contents = body.get("contents", [])
    last = contents[-1] if contents else {}
    for part in last.get("parts", []):
        response = part.get("functionResponse")
        if response:
            result = response.get("response", {})
            if "error" in result:
                return {"text": "Mock model: the command did not run ({}).".format(result["error"])}
            output = str(result.get("output", "")).strip()[:300]
            return {"text": "Mock model: the command exited with code {}. Output: {}".format(result.get("exit_code"), output or "(none)")}
    text = last_user_text(contents).strip()
    has_tool = bool(body.get("tools"))
    if has_tool and text.lower().startswith("run "):
        command = text[4:].strip()
        return {"call": {"name": "run_terminal_command", "args": {"command": command}, "id": "mock-call"}}
    if text.lower().startswith("run "):
        return {"text": "Mock model: terminal access is off or unsupported for this model, so I cannot run commands."}
    config = body.get("generationConfig")
    suffix = "  [generationConfig: {}]".format(json.dumps(config, sort_keys=True)) if config else ""
    return {"text": "Mock model reply to: {}{}".format(text or "(empty)", suffix)}


def candidate(reply, model="mock-model"):
    usage = {"promptTokenCount": 12, "candidatesTokenCount": 7, "totalTokenCount": 19}
    if "call" in reply:
        parts = [{"functionCall": reply["call"]}]
    else:
        parts = [{"text": reply["text"]}]
    return {"candidates": [{"content": {"role": "model", "parts": parts}}], "usageMetadata": usage, "modelVersion": model}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("mock:", fmt % args)

    def send_json(self, payload, status=200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        # LM Studio's local server (OpenAI-compatible); it does not use a Google key.
        if self.path.startswith("/api/v1/models"):
            return self.send_json({"models": [
                {"type": "llm", "key": "mock-llama-3", "max_context_length": 32768,
                 "loaded_instances": [{"id": "mock-llama-3", "config": {"context_length": 8192}}]},
                {"type": "llm", "key": "qwen/mock-think", "max_context_length": 32768, "loaded_instances": []},
                {"type": "llm", "key": "mock-nosystem", "max_context_length": 4096, "loaded_instances": []},
                {"type": "llm", "key": "mock-harmony", "max_context_length": 4096, "loaded_instances": []},
                {"type": "embedding", "key": "text-embedding-mock", "max_context_length": 2048, "loaded_instances": []},
            ]})
        if self.path.startswith("/v1/models"):
            names = ["mock-llama-3", "qwen/mock-think", "mock-nosystem", "mock-harmony", "text-embedding-mock"]
            return self.send_json({"data": [{"id": name, "object": "model"} for name in names]})
        if self.path.startswith("/api/v0/models/"):
            return self.send_json({"id": self.path.split("/api/v0/models/")[-1], "loaded_context_length": 8192, "max_context_length": 32768})
        if not self.headers.get("x-goog-api-key"):
            return self.send_json({"error": {"message": "missing key"}}, 401)
        if "/models/" in self.path:
            name = self.path.split("/models/")[-1].split("?")[0]
            return self.send_json({"name": "models/" + name, "inputTokenLimit": 1048576, "outputTokenLimit": 65536})
        names = ["gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-flash-latest"]
        self.send_json({"models": [{"name": "models/" + name} for name in names]})

    def lmstudio_chat(self, body):
        model = body.get("model", "mock-llama-3")
        if model == "mock-nosystem" and any(m.get("role") == "system" for m in body.get("messages", [])):
            return self.send_json({"error": "Only user and assistant roles are supported!"}, 400)
        messages = [m for m in body.get("messages", []) if m.get("role") == "user"]
        text = messages[-1].get("content", "") if messages else ""
        extras = {key: body[key] for key in ("temperature", "top_p", "top_k", "max_tokens", "seed", "stop") if key in body}
        suffix = "  [params: {}]".format(json.dumps(extras, sort_keys=True)) if extras else ""
        reply = "Mock LM Studio reply to: {}{}".format(text, suffix)
        if "think" in model:
            reply = "<think>pondering the question</think>" + reply
        if model == "mock-harmony":
            reply = "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>" + reply + "<|return|>"
        usage = {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(0, len(reply), 7):
                event = {"model": model, "choices": [{"index": 0, "delta": {"content": reply[i : i + 7]}}]}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode("utf-8"))
            final = {"model": model, "choices": [], "usage": usage}
            self.wfile.write(("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode("utf-8"))
            self.wfile.flush()
            return
        self.send_json({"model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}], "usage": usage})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if self.path.startswith("/v1/chat/completions"):
            return self.lmstudio_chat(json.loads(self.rfile.read(length) or b"{}"))
        if not self.headers.get("x-goog-api-key"):
            return self.send_json({"error": {"message": "missing key"}}, 401)
        body = json.loads(self.rfile.read(length) or b"{}")
        reply = build_reply(body)
        model = self.path.split("/models/")[-1].split(":")[0]
        # Pretend aliases resolve to a concrete model, like the real API does.
        model = "gemini-3.8-flash" if model == "gemini-flash-latest" else model
        if ":streamGenerateContent" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            text = reply.get("text", "")
            chunks = [text[i : i + 12] for i in range(0, len(text), 12)] or [""]
            for chunk in chunks:
                event = candidate({"text": chunk}, model)
                self.wfile.write(("data: " + json.dumps(event) + "\r\n\r\n").encode("utf-8"))
                self.wfile.flush()
            return
        self.send_json(candidate(reply, model))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler)
    print("Mock Gemini API on http://127.0.0.1:{} (Ctrl+C to stop)".format(arguments.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
