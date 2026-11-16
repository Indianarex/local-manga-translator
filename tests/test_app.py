"""Server API against the fake site (chapters, images, page discovery, cache), without the AI models."""
import importlib
import os
import time

import pytest

from conftest import PAGES


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MANGA_CACHE_DIR", str(tmp_path / "cache"))
    import app as appmod
    appmod = importlib.reload(appmod)
    yield appmod.app.test_client(), appmod


def test_status_and_languages(client):
    c, _ = client
    assert c.get("/api/status").get_json()["ready"] in (True, False)
    assert c.get("/api/languages").get_json()[0]["code"] == "auto"


def test_unknown_or_crafted_chapter_ids(client):
    c, _ = client
    for cid in ("deadbeefdeadbeef", "%2e%2e", "..%5c..%5cWindows"):
        assert c.get(f"/api/chapter/{cid}/page/0/image").status_code == 404


def test_classic_chapter_and_cached_image(client, site):
    c, _ = client
    base, _ = site
    ch = c.post("/api/chapter", json={"url": f"{base}/chapter"}).get_json()
    assert ch["page_count"] == PAGES and not ch["more"]
    r = c.get(f"/api/chapter/{ch['id']}/page/0/image?rev={ch['rev']}")
    assert r.status_code == 200 and r.mimetype == "image/png" and "max-age" in r.headers["Cache-Control"]
    assert c.get(f"/api/chapter/{ch['id']}/page/0/image").headers["Cache-Control"] == "no-cache"


def test_links_only_reader_is_discovered_while_reading(client, site):
    c, _ = client
    base, token = site
    ch = c.post("/api/chapter", json={"url": f"{base}/seq/{token(3)}/3"}).get_json()
    assert ch["start_page"] == 2 and ch["more"] and ch["page_count"] == 3
    r = c.get(f"/api/chapter/{ch['id']}/page/2/image")           # image of the page we asked for
    assert r.status_code == 200 and r.mimetype == "image/png"
    ch = c.post(f"/api/chapter/{ch['id']}/extend?n=5").get_json()  # discover up to page 6
    assert ch["page_count"] == 6 and ch["more"]
    ch = c.post(f"/api/chapter/{ch['id']}/extend?n=999").get_json()
    assert ch["page_count"] == PAGES and not ch["more"]


def test_cache_is_pruned_oldest_first(client, site, monkeypatch):
    c, appmod = client
    base, token = site
    ids = []
    for path in ("/chapter", f"/sel/{token(1)}", f"/num/page/1"):
        ch = c.post("/api/chapter", json={"url": base + path}).get_json()
        c.get(f"/api/chapter/{ch['id']}/page/0/image")
        ids.append(ch["id"])
        time.sleep(0.05)
    sizes = {i: sum(f.stat().st_size for f in (appmod.CACHE / i).rglob("*") if f.is_file()) for i in ids}
    monkeypatch.setattr(appmod, "CACHE_LIMIT", sizes[ids[-1]] + sizes[ids[-2]])
    appmod._prune_cache(keep=ids[-1])
    left = {p.name for p in appmod.CACHE.iterdir()}
    assert ids[0] not in left and ids[-1] in left
