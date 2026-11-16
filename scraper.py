"""Fetch the list of page images of a manga chapter from a URL."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en;q=0.9,*;q=0.5"}

IMG_EXT_RE = re.compile(r"\.(?:jpe?g|png|webp|avif)(?:\?|#|$)", re.I)
SCRIPT_IMG_RE = re.compile(
    r"https?:(?:\\?/){2}[^\s\"'<>()]+?\.(?:jpe?g|png|webp|avif)(?:\?[^\s\"'<>()]*)?", re.I
)
JUNK_RE = re.compile(
    r"logo|icon|avatar|favicon|banner|sprite|emoji|badge|button|loading|placeholder|"
    r"spinner|/ads?/|advert|facebook|twitter|discord|patreon|cover|thumb",
    re.I,
)
LAZY_ATTRS = (
    "data-src", "data-lazy-src", "data-original", "data-url", "data-cfsrc",
    "data-echo", "data-lazy", "data-img", "data-srcset", "src",
)
NAV_TEXT_RE = {  # link text at the start of the label (after an optional arrow), or a lone arrow
    "next": re.compile(r"^\W{0,3}(next|suivant|siguiente|weiter|próximo|次|다음|下一)|^\s*[›»>→]{1,2}\s*$", re.I),
    "prev": re.compile(r"^\W{0,3}(previous|prev\b|précédent|precedent|anterior|zurück|前|이전|上一)|^\s*[‹«<←]{1,2}\s*$", re.I),
}
NAV_ATTR_RE = {"next": re.compile(r"next", re.I), "prev": re.compile(r"prev", re.I)}
MAX_WALK_BACK = 60  # pages followed backwards to find page 1 of a one-page-per-URL reader
# ids/classes commonly used by single-page readers for the page image
MAIN_IMG_RE = re.compile(r"^(img|image|main-?image|mainimage|comic|comic-?img|page-?image|manga-?page|reader-?image)$", re.I)
SIZE_RE = re.compile(r"(width|height)\s*:\s*(\d+)px", re.I)


@dataclass
class Chapter:
    url: str
    title: str
    pages: list[str]
    next_url: str | None = None
    prev_url: str | None = None
    referer: str | None = None
    extra: dict = field(default_factory=dict)


class ScrapeError(Exception):
    pass


# --------------------------------------------------------------------------- utils

def _best_from_srcset(srcset: str) -> str | None:
    best, best_w = None, -1.0
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0.0
        if len(bits) > 1:
            try:
                w = float(bits[1].rstrip("wx"))
            except ValueError:
                pass
        if w >= best_w:
            best, best_w = bits[0], w
    return best


def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _directory(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc}{p.path.rsplit('/', 1)[0]}"


def _pick_main_group(urls: list[str]) -> list[str]:
    """The pages of a chapter almost always share one folder: keep the largest group."""
    urls = [u for u in urls if not JUNK_RE.search(u)] or urls
    if len(urls) < 2:
        return urls
    counts = Counter(_directory(u) for u in urls)
    top_dir, top_n = counts.most_common(1)[0]
    if top_n >= 2:
        return [u for u in urls if _directory(u) == top_dir]
    return urls


def _declared_size(img) -> int:
    """Largest width/height declared in the tag (attributes or inline style), 0 when unknown."""
    sizes = [int(v) for v in (img.get("width"), img.get("height")) if v and str(v).isdigit()]
    sizes += [int(n) for _, n in SIZE_RE.findall(img.get("style") or "")]
    return max(sizes, default=0)


def _img_url(img, base: str) -> str | None:
    for attr in LAZY_ATTRS:
        v = (img.get(attr) or "").strip()
        if "srcset" in attr:
            v = _best_from_srcset(v) or ""
        if v and not v.startswith("data:"):
            return urljoin(base, v)
    if img.get("srcset"):
        v = _best_from_srcset(img["srcset"])
        return urljoin(base, v) if v else None
    return None


def main_image(soup: BeautifulSoup, base: str) -> str | None:
    """The page image of a single-page reader: a well-known id/class, else the only large image."""
    for img in soup.find_all("img"):
        names = [img.get("id", "")] + list(img.get("class", []))
        if any(MAIN_IMG_RE.match(n) for n in names if n):
            url = _img_url(img, base)
            if url:
                return url
    large = [img for img in soup.find_all("img") if _declared_size(img) >= 400]
    if len(large) == 1:
        return _img_url(large[0], base)
    return None


# --------------------------------------------------------------------------- one page per URL
#
# Many readers show a single page per address, with "previous"/"next" links. The chapter is then a list
# of reader-page URLs (the image of each is found when the page is opened). The list is built, in order
# of preference, from:
#   1. a page menu (<select> whose options are page URLs);
#   2. numbered URLs (…/page/3, ?p=3, -3.html): every number linked from the page gives the range;
#   3. otherwise the "previous" links are followed back to page 1, and later pages are discovered while
#      reading through the "next" links ("open-ended" chapter, extended by extend_viewer()).

_NUM = re.compile(r"(\d+)")


def _same_site(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


def _fetch_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _url_template(a: str, b: str):
    """If two URLs differ only by one number that differs by 1, return (parts, index) to build others."""
    pa, pb = _NUM.split(a), _NUM.split(b)
    if len(pa) != len(pb):
        return None
    diff = [i for i, (x, y) in enumerate(zip(pa, pb)) if x != y]
    if len(diff) != 1 or diff[0] % 2 == 0:  # odd indexes of the split are the numbers
        return None
    i = diff[0]
    if abs(int(pa[i]) - int(pb[i])) != 1:
        return None
    return pa, i


def _template_number(parts, i, url: str) -> int | None:
    p = _NUM.split(url)
    if len(p) != len(parts) or any(x != y for k, (x, y) in enumerate(zip(p, parts)) if k != i):
        return None
    return int(p[i])


def _template_url(parts, i, n: int) -> str:
    width = len(parts[i]) if parts[i].startswith("0") else 0  # keep zero padding (…/005.html)
    return "".join(parts[:i] + [str(n).zfill(width)] + parts[i + 1:])


def _pages_from_select(soup: BeautifulSoup, url: str) -> list[str] | None:
    for sel in soup.find_all("select"):
        opts = sel.find_all("option")
        if len(opts) < 2:
            continue
        urls = []
        for o in opts:
            v = (o.get("value") or o.get("data-url") or "").strip()
            if v and ("/" in v or "?" in v):
                urls.append(urljoin(url, v))
        if len(urls) == len(opts) and all(_same_site(u, url) for u in urls) and url.split("#")[0] in urls:
            return _dedupe(urls)
    return None


def _pages_from_template(soup: BeautifulSoup, url: str, neighbour: str | None) -> list[str] | None:
    tpl = _url_template(url, neighbour) if neighbour else None
    if not tpl:
        return None
    parts, i = tpl
    numbers = {_template_number(parts, i, url)}
    for a in soup.find_all("a", href=True):
        n = _template_number(parts, i, urljoin(url, a["href"]))
        if n is not None:
            numbers.add(n)
    numbers.discard(None)
    lo, hi = min(numbers), max(numbers)
    if hi - lo < 1 or hi - lo > 2000:
        return None
    # The lowest linked number is often just the previous page: go down while pages have a "previous" link
    while lo > 0 and hi - lo < 2000:
        try:
            first = _template_url(parts, i, lo)
            if not _find_nav(_fetch_soup(first), first, "prev"):
                break  # page lo is the first page
            if not main_image(_fetch_soup(_template_url(parts, i, lo - 1)), url):
                break
        except requests.RequestException:
            break
        lo -= 1
    return [_template_url(parts, i, n) for n in range(lo, hi + 1)]


def _single_page_reader(url: str, soup: BeautifulSoup, title: str) -> Chapter:
    next_url, prev_url = _find_nav(soup, url, "next"), _find_nav(soup, url, "prev")
    here = url.split("#")[0]
    extra = {"viewer": True}

    pages = _pages_from_select(soup, here)
    if not pages:
        pages = _pages_from_template(soup, here, next_url or prev_url)
        if pages:
            extra["open_end"] = True  # the highest linked number is not necessarily the last page
    if pages and here in pages:
        extra["start_page"] = pages.index(here)
        if extra.get("open_end"):
            extra["last_next"] = None
        return Chapter(url=url, title=title, pages=pages, referer=url, extra=extra)

    # Follow "previous" links back to page 1 (reader pages only, no images are downloaded)
    pages, seen, cur, cur_prev = [here], {here}, here, prev_url
    while cur_prev and len(pages) <= MAX_WALK_BACK:
        cur_prev = cur_prev.split("#")[0]
        if cur_prev in seen or not _same_site(cur_prev, here):
            break
        try:
            s = _fetch_soup(cur_prev)
        except requests.RequestException:
            break
        if not main_image(s, cur_prev):
            break  # not a page of this chapter any more
        pages.insert(0, cur_prev)
        seen.add(cur_prev)
        cur, cur_prev = cur_prev, _find_nav(s, cur_prev, "prev")
    extra.update(start_page=pages.index(here), open_end=True, last_next=next_url)
    # still more pages before the first one found: "previous" opens them as a new chapter
    first_prev = cur_prev if cur_prev and cur_prev.split("#")[0] not in seen else None
    return Chapter(url=url, title=title, pages=pages, prev_url=first_prev, referer=url, extra=extra)


def extend_viewer(ch: Chapter) -> bool:
    """Discover the page after the last known one of an open-ended chapter. False when there is none."""
    if not ch.extra.get("open_end"):
        return False
    last = ch.pages[-1]
    nxt = ch.extra.get("last_next")
    if nxt is None:
        try:
            nxt = _find_nav(_fetch_soup(last), last, "next")
        except requests.RequestException:
            return False  # network error: try again later
    nxt = nxt.split("#")[0] if nxt else None
    if nxt and nxt not in ch.pages and _same_site(nxt, last):
        try:
            s = _fetch_soup(nxt)
        except requests.RequestException:
            return False
        if main_image(s, nxt):
            ch.pages.append(nxt)
            ch.extra["last_next"] = _find_nav(s, nxt, "next")
            return True
    ch.extra["open_end"] = False  # last page reached
    ch.extra.pop("last_next", None)
    return False


def resolve_viewer(viewer_url: str) -> str:
    """The image URL shown on a one-page-per-URL reader page."""
    img = main_image(_fetch_soup(viewer_url), viewer_url)
    if not img:
        raise ScrapeError("No page image found on the reader page.")
    return img


# --------------------------------------------------------------------------- MangaDex

MANGADEX_RE = re.compile(r"mangadex\.org/chapter/([0-9a-f-]{36})", re.I)


def _mangadex(url: str, chapter_id: str) -> Chapter:
    api = "https://api.mangadex.org"
    s = requests.Session()
    s.headers.update({"User-Agent": "MangaTranslator/1.0"})
    home = s.get(f"{api}/at-home/server/{chapter_id}", timeout=20)
    home.raise_for_status()
    h = home.json()
    base, ch = h["baseUrl"], h["chapter"]
    pages = [f"{base}/data/{ch['hash']}/{f}" for f in ch["data"]]
    # Fallback server in case the MangaDex@Home node is unavailable
    alt_pages = [f"https://uploads.mangadex.org/data/{ch['hash']}/{f}" for f in ch["data"]]

    title, next_url, prev_url, lang = "MangaDex", None, None, None
    try:
        info = s.get(f"{api}/chapter/{chapter_id}", params={"includes[]": "manga"}, timeout=20).json()["data"]
        attrs = info["attributes"]
        lang = {
            "ja": "ja", "zh": "zh", "zh-hk": "zh-hant", "ko": "ko",
            "es": "es", "es-la": "es", "pt": "pt", "pt-br": "pt", "ru": "ru", "en": "en",
        }.get(attrs.get("translatedLanguage"))
        manga = next((r for r in info["relationships"] if r["type"] == "manga"), None)
        mtitle = ""
        if manga and "attributes" in manga:
            t = manga["attributes"]["title"]
            mtitle = t.get("en") or next(iter(t.values()), "")
        title = f"{mtitle} – Ch. {attrs.get('chapter') or '?'}".strip(" –")
        # Next / previous chapter in the same language
        if manga and attrs.get("chapter"):
            agg = s.get(
                f"{api}/manga/{manga['id']}/aggregate",
                params={"translatedLanguage[]": attrs["translatedLanguage"]},
                timeout=20,
            ).json()
            flat = []
            for vol in agg.get("volumes", {}).values():
                for c in vol.get("chapters", {}).values():
                    try:
                        flat.append((float(c["chapter"]), c["id"]))
                    except (TypeError, ValueError):
                        pass
            flat.sort()
            ids = [cid for _, cid in flat]
            if chapter_id in ids:
                i = ids.index(chapter_id)
                if i + 1 < len(ids):
                    next_url = f"https://mangadex.org/chapter/{ids[i + 1]}"
                if i > 0:
                    prev_url = f"https://mangadex.org/chapter/{ids[i - 1]}"
    except Exception:
        pass
    return Chapter(
        url=url, title=title, pages=pages, next_url=next_url, prev_url=prev_url, referer=None,
        extra={"alt_pages": alt_pages, "user_agent": "MangaTranslator/1.0", "lang": lang},
    )


# --------------------------------------------------------------------------- generic sites

def _find_nav(soup: BeautifulSoup, base: str, rel: str) -> str | None:
    """Link to the next/previous page or chapter: rel attribute, else link text, id, class or title."""
    here = base.split("#")[0].rstrip("/")
    tag = soup.find("a", rel=rel) or soup.find("link", rel=rel)
    if tag and tag.get("href"):
        full = urljoin(base, tag["href"])
        if full.split("#")[0].rstrip("/") != here:
            return full
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("#", "javascript:")):
            continue
        text = a.get_text(" ", strip=True)
        attrs = " ".join(
            [" ".join(a.get("class", [])), a.get("id", ""), a.get("title", ""), a.get("aria-label", "")]
        )
        if NAV_TEXT_RE[rel].search(text) or NAV_ATTR_RE[rel].search(attrs):
            full = urljoin(base, href)
            if full.split("#")[0].rstrip("/") != here:
                return full
    return None


def _images_from_html(soup: BeautifulSoup, base: str) -> list[str]:
    urls = []
    for img in soup.find_all(["img", "source"]):
        cand = None
        for attr in LAZY_ATTRS:
            v = img.get(attr)
            if not v:
                continue
            v = v.strip()
            if "srcset" in attr:
                v = _best_from_srcset(v) or ""
            if v and not v.startswith("data:"):
                cand = v
                break
        if not cand and img.get("srcset"):
            cand = _best_from_srcset(img["srcset"])
        if cand:
            urls.append(urljoin(base, cand))
    return _dedupe(u for u in urls if not u.lower().endswith((".svg", ".gif")))


def _images_from_scripts(html: str) -> list[str]:
    found = [m.group(0).replace("\\/", "/") for m in SCRIPT_IMG_RE.finditer(html)]
    return _dedupe(found)


def _images_with_browser(url: str) -> tuple[list[str], str]:
    """Render the page with JavaScript through Playwright (optional dependency)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return [], ""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        # Scroll down to trigger lazy loading
        for _ in range(40):
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(250)
        page.wait_for_timeout(1000)
        imgs = page.evaluate(
            """() => Array.from(document.images)
                 .filter(i => i.naturalWidth >= 300 && i.naturalHeight >= 300)
                 .map(i => i.currentSrc || i.src)"""
        )
        html = page.content()
        browser.close()
    return _dedupe(u for u in imgs if u and not u.startswith(("data:", "blob:"))), html


SCRAPER_VERSION = 3  # bump when extraction changes, so cached chapters are fetched again


def scrape(url: str) -> Chapter:
    ch = _scrape(url)
    ch.extra["scraper_version"] = SCRAPER_VERSION
    ch.extra["scraped_at"] = time.time()
    return ch


def _scrape(url: str) -> Chapter:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    m = MANGADEX_RE.search(url)
    if m:
        return _mangadex(url, m.group(1))

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if ctype.startswith("image/") or IMG_EXT_RE.search(urlparse(url).path):
        return Chapter(url=url, title=urlparse(url).path.rsplit("/", 1)[-1], pages=[url], referer=url)

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "") or urlparse(url).netloc

    pages = _pick_main_group(_images_from_html(soup, url))
    main = main_image(soup, url)
    if main and (main not in pages or (len(pages) == 1 and (_find_nav(soup, url, "next") or _find_nav(soup, url, "prev")))):
        # One page per URL (the other images, if any, are interface icons)
        return _single_page_reader(url, soup, title)
    if len(pages) < 2:
        from_scripts = _pick_main_group(_images_from_scripts(html))
        if len(from_scripts) > len(pages):
            pages = from_scripts
    if len(pages) < 2:
        rendered, rendered_html = _images_with_browser(url)
        rendered = _pick_main_group(rendered)
        if len(rendered) > len(pages):
            pages = rendered
            soup = BeautifulSoup(rendered_html, "html.parser")

    if not pages:
        raise ScrapeError(
            "No manga images found on this page. If the site loads its images with JavaScript, "
            "install Playwright (see README) and try again."
        )

    return Chapter(
        url=url,
        title=title,
        pages=pages,
        next_url=_find_nav(soup, url, "next"),
        prev_url=_find_nav(soup, url, "prev"),
        referer=url,
    )


if __name__ == "__main__":
    import sys

    ch = scrape(sys.argv[1])
    print(json.dumps(ch.__dict__, indent=2, ensure_ascii=False))
