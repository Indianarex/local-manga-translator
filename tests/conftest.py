"""Shared fixtures: a small fake manga site served locally, so the tests need no internet access."""
from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The fake site runs on 127.0.0.1, which the app normally refuses to download from
os.environ["MANGA_ALLOW_PRIVATE_HOSTS"] = "1"

PAGES = 12  # pages of each fake chapter
ICONS = '<img src="/ui/prev.png"><img src="/ui/next.png"><img src="/ui/logo.png">'


def _png(n: int, size=(600, 860)) -> bytes:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).rectangle([40, 40, size[0] - 40, size[1] - 40], outline="black", width=4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _token(n: int) -> str:
    """Per-page random-looking token: no numeric pattern to guess the other pages from."""
    return hashlib.sha1(f"page{n}".encode()).hexdigest()[:10]


def _page(img: str, prev: str | None, nxt: str | None, extra: str = "") -> bytes:
    nav = (f'<a id="prev" href="{prev}">Prev</a>' if prev else "") + (f'<a id="next" href="{nxt}">Next</a>' if nxt else "")
    return (f"<html><head><title>Fake reader</title></head><body>{ICONS}{nav}{extra}"
            f'<img id="img" src="{img}" style="width:600px;height:860px">{ICONS}{nav}</body></html>').encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype="text/html"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0].strip("/").split("/")
        if p[0] == "ui":
            return self._send(_png(0, (24, 24)), "image/png")
        if p[0] == "empty":  # a page without any manga image
            return self._send(b"<html><body><p>Nothing here</p></body></html>")
        if p[0] == "img":
            return self._send(_png(int(p[1])), "image/png")
        # 1. numbered URLs: /num/page/<n>, with links to the neighbours and to the last page
        if p[0] == "num" and len(p) == 3:
            n = int(p[2])
            if not 1 <= n <= PAGES:
                return self.send_error(404)
            prev = f"/num/page/{n - 1}" if n > 1 else None
            nxt = f"/num/page/{n + 1}" if n < PAGES else None
            return self._send(_page(f"/img/{n}", prev, nxt, f'<a href="/num/page/{PAGES}">Last</a>'))
        # 2. page menu: /sel/<token>
        if p[0] == "sel" and len(p) == 2:
            n = next(i for i in range(1, PAGES + 1) if _token(i) == p[1])
            opts = "".join(f'<option value="/sel/{_token(i)}">{i}</option>' for i in range(1, PAGES + 1))
            prev = f"/sel/{_token(n - 1)}" if n > 1 else None
            nxt = f"/sel/{_token(n + 1)}" if n < PAGES else None
            return self._send(_page(f"/img/{n}", prev, nxt, f"<select>{opts}</select>"))
        # 3. links only: /seq/<token>/<n> (the token changes on every page)
        if p[0] == "seq" and len(p) == 3:
            n = int(p[2])
            prev = f"/seq/{_token(n - 1)}/{n - 1}" if n > 1 else f"/seq/{_token(1)}/1"   # page 1 links to itself
            nxt = f"/seq/{_token(n + 1)}/{n + 1}" if n < PAGES else f"/seq/{_token(n)}/{n}"
            return self._send(_page(f"/img/{n}", prev, nxt))
        # 4. a classic chapter page with every image
        if p[0] == "chapter":
            imgs = "".join(f'<img data-src="/img/{i}">' for i in range(1, PAGES + 1))
            return self._send(f"<html><body>{ICONS}{imgs}<a class='next-chap' href='/chapter2'>Next</a></body></html>".encode())
        self.send_error(404)


@pytest.fixture(scope="session")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _token
    server.shutdown()
