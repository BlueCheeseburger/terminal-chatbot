#!/usr/bin/env python3
"""A fake Gemini API on localhost for testing the TUI without spending API credits.

Run it, then start the app against it:

    python3 tools/mock_gemini_server.py
    GEMINI_API_KEY=mock python3 gemini_legacy_tui.py --api-base-url http://127.0.0.1:8765

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
        if not self.headers.get("x-goog-api-key"):
            return self.send_json({"error": {"message": "missing key"}}, 401)
        if "/models/" in self.path:
            name = self.path.split("/models/")[-1].split("?")[0]
            return self.send_json({"name": "models/" + name, "inputTokenLimit": 1048576, "outputTokenLimit": 65536})
        names = ["gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-flash-latest"]
        self.send_json({"models": [{"name": "models/" + name} for name in names]})

    def do_POST(self):
        if not self.headers.get("x-goog-api-key"):
            return self.send_json({"error": {"message": "missing key"}}, 401)
        length = int(self.headers.get("Content-Length", "0"))
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
