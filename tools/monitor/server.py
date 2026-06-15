"""只读训练看板 HTTP 服务(stdlib)。"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collector import list_jobs

_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")


def make_handler(output_root: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index.html"):
                try:
                    with open(_HTML_PATH, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    self._send(500, b"index.html missing", "text/plain")
            elif self.path.startswith("/api/jobs"):
                try:
                    payload = json.dumps(list_jobs(output_root)).encode("utf-8")
                except Exception as e:  # 端点永不 500
                    payload = json.dumps(
                        {"generated_at": None, "gpus": [], "jobs": [],
                         "error": str(e)}).encode("utf-8")
                self._send(200, payload, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):  # 静音默认访问日志
            pass

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="U-Bench 只读训练看板")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--output-root", default="./output")
    args = parser.parse_args(argv)

    handler = make_handler(args.output_root)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"训练看板: http://{args.host}:{args.port}  (output-root={args.output_root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        httpd.server_close()


if __name__ == "__main__":
    main()
