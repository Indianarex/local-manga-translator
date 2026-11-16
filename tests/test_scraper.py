from bs4 import BeautifulSoup

import scraper
from conftest import PAGES


def test_srcset_picks_largest():
    assert scraper._best_from_srcset("a.jpg 800w, b.jpg 1600w") == "b.jpg"


def test_lazy_images_and_icon_filtering():
    html = """<img src="/img/logo.png"><img src="data:," data-src="https://cdn.x/m/ch1/01.jpg">
    <img data-lazy-src="https://cdn.x/m/ch1/02.jpg"><img srcset="https://cdn.x/m/ch1/03.webp 1600w">
    <img src="https://ads.y/banner.jpg">"""
    imgs = scraper._images_from_html(BeautifulSoup(html, "html.parser"), "https://site/ch1")
    assert scraper._pick_main_group(imgs) == [
        "https://cdn.x/m/ch1/01.jpg", "https://cdn.x/m/ch1/02.jpg", "https://cdn.x/m/ch1/03.webp"]


def test_images_in_scripts():
    found = scraper._images_from_scripts(r'var p=["https:\/\/a.com\/x\/1.jpg","https:\/\/a.com\/x\/2.jpg"]')
    assert found == ["https://a.com/x/1.jpg", "https://a.com/x/2.jpg"]


def test_navigation_links():
    soup = BeautifulSoup('<a class="btn next-chap" href="/ch2">Chapter</a><a href="/ch0">« Previous</a>', "html.parser")
    assert scraper._find_nav(soup, "https://site/ch1", "next") == "https://site/ch2"
    assert scraper._find_nav(soup, "https://site/ch1", "prev") == "https://site/ch0"


def test_url_template():
    parts, i = scraper._url_template("https://s/read/page/7", "https://s/read/page/8")
    assert scraper._template_url(parts, i, 12) == "https://s/read/page/12"
    parts, i = scraper._url_template("https://s/c/005.html", "https://s/c/006.html")
    assert scraper._template_url(parts, i, 9) == "https://s/c/009.html"   # zero padding kept
    assert scraper._url_template("https://s/a/1", "https://s/b/2") is None  # two differences
    assert scraper._url_template("https://s/p/1", "https://s/p/5") is None  # not consecutive


def test_classic_chapter(site):
    base, _ = site
    ch = scraper.scrape(f"{base}/chapter")
    assert len(ch.pages) == PAGES and not ch.extra.get("viewer")
    assert ch.next_url == f"{base}/chapter2"


def test_numbered_reader(site):
    base, _ = site
    ch = scraper.scrape(f"{base}/num/page/3")
    assert ch.extra["viewer"] and len(ch.pages) == PAGES
    assert ch.pages[ch.extra["start_page"]] == f"{base}/num/page/3"
    assert scraper.resolve_viewer(ch.pages[0]) == f"{base}/img/1"   # the page image, not an icon


def test_page_menu_reader(site):
    base, token = site
    ch = scraper.scrape(f"{base}/sel/{token(5)}")
    assert len(ch.pages) == PAGES and ch.extra["start_page"] == 4 and not ch.extra.get("open_end")


def test_links_only_reader_walks_back_then_extends(site):
    base, token = site
    ch = scraper.scrape(f"{base}/seq/{token(4)}/4")
    assert ch.pages[0] == f"{base}/seq/{token(1)}/1"       # walked back to page 1
    assert ch.extra["start_page"] == 3 and ch.extra["open_end"]
    while scraper.extend_viewer(ch):                        # pages discovered while reading
        pass
    assert len(ch.pages) == PAGES and not ch.extra["open_end"]
    assert ch.pages[-1] == f"{base}/seq/{token(PAGES)}/{PAGES}"
