"""Local web server: a manga reader that automatically translates pages into English."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

try:  # use the OS certificate store (antivirus software, corporate proxies...)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import requests
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from render import render_page
from scraper import (HEADERS, SCRAPER_VERSION, BlockedURL, Chapter, ScrapeError, extend_viewer, fetch,
                     resolve_viewer, scrape)
import translator
from translator import TranslationError, Translator

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("MANGA_CACHE_DIR") or ROOT / "cache")
CACHE.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")

# The server only listens on 127.0.0.1, but any web page open in the browser can still send it requests.
# Only accept requests addressed to localhost (blocks DNS rebinding), and only JSON for requests that act
# (browsers cannot send those cross-site without a CORS preflight, which this server never grants).
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


@app.before_request
def _local_only():
    host = (request.host or "").lower()
    if not host.startswith("["):
        host = host.split(":")[0]
    elif "]" in host:
        host = host[:host.index("]") + 1]
    if host not in _LOCAL_HOSTS:
        abort(403)
    if request.method == "POST" and not request.is_json:
        return jsonify({"error": "Only JSON requests are accepted"}), 415


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp

_chapters: dict[str, Chapter] = {}
_jobs: dict[tuple[str, int, str], Future] = {}
_jobs_lock = threading.Lock()
_img_locks: dict[tuple[str, int], threading.Lock] = {}
_pool = ThreadPoolExecutor(max_workers=3)
PREFETCH = 2  # number of pages translated ahead of the current one
# Maximum size of the cache folder (pages, translations, lettered pages); the chapters read longest
# ago are deleted beyond it. Override with the MANGA_CACHE_MB environment variable.
CACHE_LIMIT = int(os.environ.get("MANGA_CACHE_MB", "2048")) * 1024 * 1024
_prune_lock = threading.Lock()

_translator = Translator()


# --------------------------------------------------------------------------- chapters

def _chapter_id(url: str) -> str:
    return hashlib.sha1(url.strip().encode()).hexdigest()[:16]


_CACHE_ROOT = os.path.realpath(CACHE)
_CID_RE = re.compile(r"[0-9a-f]{16}")


def _chapter_dir(cid: str, create: bool = True) -> Path:
    """Cache folder of a chapter. The id comes from the request URL, so it is checked strictly:
    exactly 16 hexadecimal characters, and the resulting path must stay inside the cache folder."""
    if not _CID_RE.fullmatch(cid or ""):
        abort(404, "Unknown chapter")
    path = os.path.normpath(os.path.join(_CACHE_ROOT, cid))
    if not path.startswith(_CACHE_ROOT + os.sep):
        abort(404, "Unknown chapter")
    d = Path(path)
    if create:
        d.mkdir(exist_ok=True)
    return d


def _get_chapter(cid: str) -> Chapter:
    ch = _chapters.get(cid)
    if ch:
        return ch
    meta = _chapter_dir(cid, create=False) / "chapter.json"
    if meta.exists():
        ch = Chapter(**json.loads(meta.read_text(encoding="utf-8")))
        _chapters[cid] = ch
        return ch
    abort(404, "Unknown chapter")


def _save_chapter(cid: str, ch: Chapter):
    # Write to a temporary file, then swap it in: a crash can never leave a half-written chapter.json
    path = _chapter_dir(cid) / "chapter.json"
    tmp = path.with_name(f"chapter.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(ch.__dict__, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _prune_cache(keep: str | None = None):
    """Keep the cache under CACHE_LIMIT by deleting the least recently opened chapters."""
    with _prune_lock:
        dirs = [d for d in CACHE.iterdir() if d.is_dir()]
        sizes = {d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in dirs}
        total = sum(sizes.values())
        if total <= CACHE_LIMIT:
            return

        def last_opened(d: Path) -> float:
            meta = d / "chapter.json"
            return meta.stat().st_mtime if meta.exists() else 0.0

        for d in sorted(dirs, key=last_opened):
            if total <= CACHE_LIMIT * 0.8:  # free some margin so this does not run on every page
                break
            if d.name == keep:
                continue
            shutil.rmtree(d, ignore_errors=True)
            _chapters.pop(d.name, None)
            total -= sizes[d]


def _chapter_json(cid: str, ch: Chapter) -> dict:
    return {
        "id": cid, "title": ch.title, "url": ch.url, "page_count": len(ch.pages),
        "more": bool(ch.extra.get("open_end")),  # more pages exist beyond page_count (discovered while reading)
        "next_url": ch.next_url, "prev_url": ch.prev_url, "start_page": ch.extra.get("start_page", 0),
        # Fingerprint of this extraction, added to image URLs: if the chapter is extracted again,
        # the URLs change, so the browser can never show an image cached from an older extraction.
        "rev": hashlib.sha1(f"{ch.url}|{ch.extra.get('scraped_at', '')}".encode()).hexdigest()[:10],
    }


_extend_lock = threading.Lock()


def _ensure_page(cid: str, n: int) -> bool:
    """For chapters whose pages are discovered while reading: make sure page n is known."""
    ch = _get_chapter(cid)
    with _extend_lock:
        grew = False
        while n >= len(ch.pages) and ch.extra.get("open_end"):
            if not extend_viewer(ch):
                break
            grew = True
        if grew or not ch.extra.get("open_end"):
            _save_chapter(cid, ch)
    return 0 <= n < len(ch.pages)


def _download_image(url: str, headers: dict) -> requests.Response:
    r = fetch(url, headers=headers, timeout=60)  # public addresses only (see scraper.fetch)
    r.raise_for_status()
    if not r.headers.get("Content-Type", "image/").startswith("image/"):
        raise requests.HTTPError(f"Not an image ({r.headers.get('Content-Type')}) at {url}")
    return r


def _page_image(cid: str, n: int) -> tuple[bytes, str]:
    ch = _get_chapter(cid)
    if not 0 <= n < len(ch.pages):
        abort(404, "No such page")
    d = _chapter_dir(cid)
    with _img_locks.setdefault((cid, n), threading.Lock()):
        for p in d.glob(f"page{n:04d}.*"):
            if p.suffix != ".json":
                return p.read_bytes(), p.suffix
        headers = dict(HEADERS)
        headers["Accept"] = "image/avif,image/webp,image/*,*/*;q=0.8"
        if ch.extra.get("user_agent"):
            headers["User-Agent"] = ch.extra["user_agent"]
        if ch.referer:
            headers["Referer"] = ch.referer
        r, last_exc = None, None
        if ch.extra.get("viewer"):
            # The page URL is a reader page showing one image: find the image, then download it
            viewer = ch.pages[n]
            for _ in range(2):
                try:
                    r = _download_image(resolve_viewer(viewer), {**headers, "Referer": viewer})
                    break
                except ScrapeError as e:
                    last_exc = requests.RequestException(str(e))
                    break
                except requests.RequestException as e:
                    last_exc = e
        else:
            urls = [ch.pages[n]]
            alt = ch.extra.get("alt_pages") or []
            if n < len(alt):
                urls.append(alt[n])
            for u in urls * 2:  # two attempts per URL
                try:
                    r = _download_image(u, headers)
                    break
                except requests.RequestException as e:
                    last_exc = e
        if r is None:
            raise last_exc or requests.RequestException("Could not download the image")
        ctype = r.headers.get("Content-Type", "")
        ext = {"image/png": ".png", "image/webp": ".webp", "image/avif": ".avif", "image/gif": ".gif"}.get(
            ctype.split(";")[0], ".jpg"
        )
        (d / f"page{n:04d}{ext}").write_bytes(r.content)
        return r.content, ext


# --------------------------------------------------------------------------- translation jobs

# Known language codes; the code used in file names always comes from this table, never from the request
_LANG_CODES = {code: code for code in [*translator.LANGUAGES, "auto"]}


def _lang_param() -> str:
    return _LANG_CODES.get(request.args.get("lang", "auto"), "auto")


def _translation_path(cid: str, n: int, lang: str) -> Path:
    return _chapter_dir(cid) / f"page{n:04d}.{lang}.json"


def _rendered_path(cid: str, n: int, lang: str, hide_sfx: bool) -> Path:
    # Not named "page...": that prefix is reserved for the source images
    return _chapter_dir(cid) / f"rendered{n:04d}.{lang}.{'nosfx' if hide_sfx else 'sfx'}.webp"


def _clear_rendered(cid: str, n: int, lang: str):
    for hide in (False, True):
        _rendered_path(cid, n, lang, hide).unlink(missing_ok=True)


def _render(cid: str, n: int, lang: str, hide_sfx: bool) -> bytes:
    """The page with the translation lettered in (cached)."""
    path = _rendered_path(cid, n, lang, hide_sfx)
    if path.exists():
        return path.read_bytes()
    data = _do_translate(cid, n, lang)
    img, _ = _page_image(cid, n)
    out = render_page(img, data, hide_sfx=hide_sfx, fmt="WEBP")
    path.write_bytes(out)
    return out


def _do_translate(cid: str, n: int, lang: str, prerender: bool = False) -> dict:
    path = _translation_path(cid, n, lang)
    result = None
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("format", 1) < translator.FORMAT:  # cached by an older version: redo it
            result = None
            _clear_rendered(cid, n, lang)
    if result is None:
        img, _ = _page_image(cid, n)
        ch = _get_chapter(cid)
        # Likely language: announced by the site (MangaDex) or detected on an earlier page
        hint = ch.extra.get("lang") or ch.extra.get("detected_lang")
        result = _translator.translate(img, lang=lang, hint=hint)
        if lang == "auto" and result["items"] and not ch.extra.get("detected_lang"):
            ch.extra["detected_lang"] = result["source_language"]
            _save_chapter(cid, ch)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if prerender:  # letter the page now, so turning to it is instant
        _render(cid, n, lang, hide_sfx=False)
    return result


def _job(cid: str, n: int, lang: str) -> dict | None:
    if not _ensure_page(cid, n):  # pages discovered while reading: find page n first
        return None
    return _do_translate(cid, n, lang, prerender=True)


def _submit(cid: str, n: int, lang: str) -> Future | None:
    ch = _get_chapter(cid)
    if n < 0 or (n >= len(ch.pages) and not ch.extra.get("open_end")):
        return None
    with _jobs_lock:
        fut = _jobs.get((cid, n, lang))
        if fut is None or (fut.done() and (fut.exception() is not None or fut.result() is None)):
            fut = _pool.submit(_job, cid, n, lang)
            _jobs[(cid, n, lang)] = fut
        return fut


# --------------------------------------------------------------------------- errors

# The browser only gets these fixed messages; the details (exception text) go to the server window.
MSG_BLOCKED = "This address cannot be opened: only public http(s) web addresses are allowed."
MSG_NO_IMAGES = ("No manga images were found on this page. If the site loads its images with JavaScript, "
                 "install Playwright (see the README).")
MSG_PAGE = "Could not load the page. Check the address and your internet connection."
MSG_NEXT_PAGE = "Could not load the next page."
MSG_IMAGE = "Could not download the image of this page."
MSG_TRANSLATION = "The translation failed. Details are shown in the server window."
MSG_UNEXPECTED = "Unexpected error. Details are shown in the server window."


def _error(message: str, status: int, exc: BaseException):
    app.logger.warning("%s [%s: %s]", message, type(exc).__name__, exc)
    return jsonify({"error": message}), status


# --------------------------------------------------------------------------- routes

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def get_status():
    return jsonify(translator.status())


@app.get("/api/languages")
def get_languages():
    return jsonify(translator.language_options())


@app.post("/api/chapter")
def load_chapter():
    url = ((request.get_json(silent=True) or {}).get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing URL"}), 400
    cid = _chapter_id(url)
    meta = _chapter_dir(cid) / "chapter.json"
    try:
        ch = _get_chapter(cid) if meta.exists() else None
        if ch is None or request.args.get("refresh") or ch.extra.get("scraper_version") != SCRAPER_VERSION:
            # (Re)extract the chapter; drop images and translations cached from an older extraction
            for f in _chapter_dir(cid).iterdir():
                if f.is_file():
                    f.unlink()
            with _jobs_lock:
                for key in [k for k in _jobs if k[0] == cid]:
                    del _jobs[key]
            ch = scrape(url)
            _chapters[cid] = ch
            _save_chapter(cid, ch)
    except BlockedURL as e:
        return _error(MSG_BLOCKED, 422, e)
    except ScrapeError as e:
        return _error(MSG_NO_IMAGES, 422, e)
    except requests.RequestException as e:
        return _error(MSG_PAGE, 502, e)
    now = time.time()
    os.utime(meta, (now, now))  # "last opened" date, used by the cache clean-up
    threading.Thread(target=_prune_cache, args=(cid,), daemon=True).start()
    return jsonify(_chapter_json(cid, ch))


@app.get("/api/chapter/<cid>")
def chapter_info(cid):
    """Current state of a chapter (its page count grows on readers whose pages are discovered while reading)."""
    return jsonify(_chapter_json(cid, _get_chapter(cid)))


@app.post("/api/chapter/<cid>/extend")
def chapter_extend(cid):
    """Discover pages up to page n (0-based) when the chapter has more pages than known so far."""
    try:
        _ensure_page(cid, request.args.get("n", type=int, default=0))
    except requests.RequestException as e:
        return _error(MSG_NEXT_PAGE, 502, e)
    return jsonify(_chapter_json(cid, _get_chapter(cid)))


@app.get("/api/chapter/<cid>/page/<int:n>/image")
def page_image(cid, n):
    try:
        data, ext = _page_image(cid, n)
    except requests.RequestException as e:
        return _error(MSG_IMAGE, 502, e)
    mime ={".png": "image/png", ".webp": "image/webp", ".avif": "image/avif", ".gif": "image/gif"}.get(
        ext, "image/jpeg"
    )
    resp = send_file(io.BytesIO(data), mimetype=mime)
    # Long caching is safe only for versioned URLs (?rev=...); otherwise the browser must revalidate
    resp.headers["Cache-Control"] = "private, max-age=86400" if request.args.get("rev") else "no-cache"
    return resp


@app.get("/api/chapter/<cid>/page/<int:n>/translation")
def page_translation(cid, n):
    lang = _lang_param()
    fut = _submit(cid, n, lang)
    if fut is None:
        abort(404)
    # Translate the next pages in the background
    for k in range(1, PREFETCH + 1):
        _submit(cid, n + k, lang)
    try:
        result = fut.result()
        if result is None:
            abort(404)
        return jsonify(result)
    except TranslationError as e:
        return _error(MSG_TRANSLATION, 502, e)
    except requests.RequestException as e:
        return _error(MSG_IMAGE, 502, e)
    except Exception as e:  # noqa: BLE001
        app.logger.exception("Unexpected error while translating a page")
        return _error(MSG_UNEXPECTED, 500, e)


@app.get("/api/chapter/<cid>/page/<int:n>/rendered")
def page_rendered(cid, n):
    lang = _lang_param()
    if not 0 <= n < len(_get_chapter(cid).pages):
        abort(404)
    try:
        data = _render(cid, n, lang, hide_sfx=request.args.get("sfx") == "0")
    except TranslationError as e:
        return _error(MSG_TRANSLATION, 502, e)
    except requests.RequestException as e:
        return _error(MSG_IMAGE, 502, e)
    resp = send_file(io.BytesIO(data), mimetype="image/webp")
    # the URL changes when the page is translated again (?t=...), so it can be cached
    resp.headers["Cache-Control"] = "private, max-age=86400" if request.args.get("rev") else "no-cache"
    return resp


@app.post("/api/chapter/<cid>/page/<int:n>/retranslate")
def retranslate(cid, n):
    lang = _lang_param()
    _translation_path(cid, n, lang).unlink(missing_ok=True)
    _clear_rendered(cid, n, lang)
    with _jobs_lock:
        _jobs.pop((cid, n, lang), None)
    return page_translation(cid, n)


@app.get("/api/chapter/<cid>/page/<int:n>/download")
def download(cid, n):
    lang = _lang_param()
    if not _translation_path(cid, n, lang).exists():
        return jsonify({"error": "This page has not been translated yet"}), 409
    img, _ = _page_image(cid, n)
    png = render_page(img, _do_translate(cid, n, lang), hide_sfx=request.args.get("sfx") == "0", fmt="PNG")
    return send_file(io.BytesIO(png), mimetype="image/png", as_attachment=True,
                     download_name=f"page_{n + 1:03d}_en.png")


if __name__ == "__main__":
    # Load the models in the background at startup (downloaded on first run)
    threading.Thread(target=translator.preload, daemon=True).start()
    threading.Thread(target=_prune_cache, daemon=True).start()
    port = int(os.environ.get("PORT", "5000"))
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)
