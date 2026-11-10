"""Lightweight MJPEG web stream for LIBERO evaluation (stdlib + imageio)."""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import imageio.v2 as imageio
import numpy as np


_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>LIBERO Live Stream</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    header {{ padding: 12px 16px; background: #1a1a1a; border-bottom: 1px solid #333; }}
    main {{ display: flex; justify-content: center; padding: 16px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #444; background: #000; }}
    #status {{ font-size: 14px; color: #aaa; margin-top: 6px; }}
  </style>
</head>
<body>
  <header>
    <div>LIBERO Evaluation — Live Stream</div>
    <div id="status">Connecting...</div>
  </header>
  <main>
    <img src="/video" alt="LIBERO stream">
  </main>
  <script>
    async function refreshStatus() {{
      try {{
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').textContent =
          `Task ${{data.task_id}} | Episode ${{data.episode}} | Step ${{data.step}} | ${{data.status}}`;
      }} catch (e) {{
        document.getElementById('status').textContent = 'Waiting for frames...';
      }}
    }}
    setInterval(refreshStatus, 1000);
    refreshStatus();
  </script>
</body>
</html>
"""


class WebStreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._meta = {
            "task_id": "-",
            "episode": "-",
            "step": 0,
            "status": "waiting",
        }
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def get_meta(self) -> dict:
        with self._lock:
            return dict(self._meta)

    def update(self, frame: np.ndarray, **meta) -> None:
        buf = io.BytesIO()
        imageio.imwrite(buf, frame.astype(np.uint8), format="JPEG", quality=80)
        with self._lock:
            self._jpeg = buf.getvalue()
            self._meta.update(meta)

    def start(self) -> None:
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/status":
                    body = json.dumps(streamer.get_meta()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/video":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    try:
                        while True:
                            jpeg = streamer.get_jpeg()
                            if jpeg is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                                self.wfile.write(jpeg)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                            time.sleep(0.033)
                    except (BrokenPipeError, ConnectionResetError):
                        return

                self.send_response(404)
                self.end_headers()

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
