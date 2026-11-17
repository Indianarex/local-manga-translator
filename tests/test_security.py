"""Security checks: downloads limited to public addresses, safe cache paths, no exception details sent."""
import importlib

import pytest
import requests

import scraper


@pytest.fixture()
def public_only(monkeypatch):
    monkeypatch.delenv("MANGA_ALLOW_PRIVATE_HOSTS", raising=False)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:5000/api/status",       # this machine
    "http://localhost/",
    "http://10.0.0.1/", "http://192.168.1.1/admin", "http://172.16.0.5/",   # local network
    "http://169.254.169.254/latest/meta-data",  # cloud metadata
    "http://[::1]/", "http://0.0.0.0/",
])
def test_local_addresses_are_refused(public_only, url):
    with pytest.raises(scraper.BlockedURL):
        scraper.check_url(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)", "http:///nohost"])
def test_other_schemes_are_refused(url):
    with pytest.raises(scraper.BlockedURL):
        scraper.check_url(url)


def test_public_addresses_are_allowed(public_only):
    assert scraper.check_url("http://93.184.216.34/page") == "http://93.184.216.34/page"


def test_redirects_are_checked(site, monkeypatch):
    base, _ = site
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        r = requests.Response()
        r.status_code, r.url = 302, url
        r.headers["Location"] = "http://169.254.169.254/latest/meta-data"
        return r

    monkeypatch.setattr(scraper.requests, "get", fake_get)
    monkeypatch.delenv("MANGA_ALLOW_PRIVATE_HOSTS", raising=False)
    with pytest.raises(scraper.BlockedURL):
        scraper.fetch("http://93.184.216.34/redirect")
    assert calls == ["http://93.184.216.34/redirect"]   # the redirect target was never requested


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MANGA_CACHE_DIR", str(tmp_path / "cache"))
    import app as appmod
    appmod = importlib.reload(appmod)
    yield appmod.app.test_client(), appmod, tmp_path


@pytest.mark.parametrize("cid", ["..", "..%5c..%5cWindows", "ABCDEF0123456789", "0123456789abcdef0", "x" * 16])
def test_crafted_chapter_ids_never_touch_the_disk(client, cid):
    c, appmod, tmp_path = client
    before = {p for p in tmp_path.rglob("*")}
    assert c.get(f"/api/chapter/{cid}/page/0/image").status_code == 404
    assert c.post(f"/api/chapter/{cid}/page/0/retranslate", json={}).status_code == 404
    assert c.post(f"/api/chapter/{cid}/extend?n=3", json={}).status_code == 404
    assert {p for p in tmp_path.rglob("*")} == before     # nothing created or deleted


def test_unknown_language_codes_never_reach_file_names(client):
    _, appmod, _ = client
    with appmod.app.test_request_context("/?lang=../../evil"):
        assert appmod._lang_param() == "auto"


def test_errors_do_not_leak_exception_details(client, site, public_only):
    c, appmod, _ = client
    base, _ = site
    r = c.post("/api/chapter", json={"url": f"{base}/chapter"})        # 127.0.0.1: refused
    assert r.status_code == 422 and r.get_json()["error"] == appmod.MSG_BLOCKED


def test_page_without_images_gives_a_clear_message(client, site):
    c, appmod, _ = client
    base, _ = site
    r = c.post("/api/chapter", json={"url": f"{base}/empty"})
    assert r.status_code == 422 and r.get_json()["error"] == appmod.MSG_NO_IMAGES
