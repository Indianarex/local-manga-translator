"""Typesetting: erase the original text and letter the English translation into the page.

For every text area the page itself is analysed to decide how the translation should look:
  - background: a flat bubble/box (repainted with its own color) or artwork (inpainted);
  - colors: the original ink color, with a contrasting outline when the text sits on artwork;
  - size: derived from the size of the original lettering, then shrunk until the text fits;
  - shape: lines follow the oval of a speech bubble, or fill a rectangular box;
  - font: dialogue, shouting, thoughts, narration, handwritten notes and sound effects get their own font.
Drawn sound effects (SFX) are kept and get a small translated label, since erasing them would damage the art.
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parent / "fonts"
STYLES = {
    # style: (font file, variable-font weight or None, line spacing)
    "speech": ("ComicNeue-Bold.ttf", None, 1.12),
    "thought": ("ComicNeue-BoldItalic.ttf", None, 1.12),
    "shout": ("Bangers-Regular.ttf", None, 1.05),
    "narration": ("RobotoCondensed-Variable.ttf", 600, 1.12),
    "note": ("PatrickHand-Regular.ttf", None, 1.05),
    "sfx": ("Bangers-Regular.ttf", None, 1.0),
}
MIN_SIZE = 9
CJK_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
# scanlation credits / watermarks: left as they are
WATERMARK = re.compile(r"https?://|www\.|\.(com|net|org|me|io|gg)\b|t\.me/|@\w{3,}|discord|[a-z0-9]_[a-z0-9]", re.I)


@lru_cache(maxsize=256)
def _font(style: str, size: int) -> ImageFont.FreeTypeFont:
    file, weight, _ = STYLES[style]
    try:
        f = ImageFont.truetype(str(FONTS / file), size)
        if weight:
            f.set_variation_by_axes([weight])
        return f
    except OSError:  # fonts folder missing: fall back to a system font
        for alt in ("C:/Windows/Fonts/comicbd.ttf", "C:/Windows/Fonts/arialbd.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if Path(alt).exists():
                return ImageFont.truetype(alt, size)
        return ImageFont.load_default(size)


# --------------------------------------------------------------------------- analysis

def _lum(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _px(box, W, H, pad=0):
    x0, y0, x1, y1 = box
    return [max(0, int(x0 * W) - pad), max(0, int(y0 * H) - pad), min(W, int(round(x1 * W)) + pad),
            min(H, int(round(y1 * H)) + pad)]


def _ellipse_mask(shape, box, scale=1.0):
    """Boolean mask of the ellipse inscribed in box (scaled around its center)."""
    h, w = shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = max(1, (x1 - x0) / 2 * scale), max(1, (y1 - y0) / 2 * scale)
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1


def _runs(profile: np.ndarray) -> list[int]:
    """Lengths of consecutive non-empty runs in a projection profile."""
    out, n = [], 0
    for v in profile:
        if v:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def _rect_mask(shape, box, inset=0):
    h, w = shape
    x0, y0, x1, y1 = box
    m = np.zeros(shape, bool)
    m[max(0, int(y0) + inset):max(0, min(h, int(y1) - inset)), max(0, int(x0) + inset):max(0, min(w, int(x1) - inset))] = True
    return m


def _is_oval(img: np.ndarray, bubble, tb, bg_lum: float) -> bool:
    """Oval or rectangular bubble? Walking from the center towards each corner, an oval's outline is
    crossed before the corner; a rectangle's is not. Oval unless the evidence for a rectangle is clear."""
    H, W = img.shape[:2]
    bx0, by0, bx1, by1 = bubble
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    clear_corners = 0
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        hit, seen = False, 0
        for t in np.linspace(0.55, 0.96, 40):
            x = int(cx + dx * t * (bx1 - bx0) / 2)
            y = int(cy + dy * t * (by1 - by0) / 2)
            if tb[0] - 3 <= x <= tb[2] + 3 and tb[1] - 3 <= y <= tb[3] + 3:
                continue  # ignore the text itself
            seen += 1
            if abs(float(_lum(img[min(y, H - 1), min(x, W - 1)].astype(float))) - bg_lum) > 80:
                hit = True
                break
        clear_corners += (not hit) and seen >= 8
    return clear_corners < 3


def _analyse(img: np.ndarray, it: dict) -> dict:
    """Describe the background, ink and size of one text area (all in pixels)."""
    H, W = img.shape[:2]
    tb = _px(it["text_box"], W, H)
    bubble = _px(it["bubble"], W, H) if it.get("bubble") else None

    # Background sample: inside the bubble but outside the text, or a band around the text
    if bubble:
        bx0, by0, bx1, by1 = bubble
        region = img[by0:by1, bx0:bx1]
        sample = _ellipse_mask(region.shape[:2], (0, 0, bx1 - bx0, by1 - by0), 0.85)
        sample[max(0, tb[1] - by0 - 3):max(0, tb[3] - by0 + 3), max(0, tb[0] - bx0 - 3):max(0, tb[2] - bx0 + 3)] = False
        pixels = region[sample]
    else:
        band = _px(it["text_box"], W, H, pad=max(4, (tb[3] - tb[1]) // 8))
        region = img[band[1]:band[3], band[0]:band[2]]
        sample = np.ones(region.shape[:2], bool)
        sample[tb[1] - band[1]:tb[3] - band[1], tb[0] - band[0]:tb[2] - band[0]] = False
        pixels = region[sample]
    if len(pixels) < 20:  # no room around the text: use the text box border
        crop = img[tb[1]:tb[3], tb[0]:tb[2]]
        pixels = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]) if crop.size else img.reshape(-1, 3)
    bg = np.median(pixels, axis=0)
    bg_lum = float(_lum(bg))
    texture = float(np.std(_lum(pixels.astype(np.float32))))
    flat = texture < 22

    # Ink: pixels of the text box that clearly differ from the background
    crop = img[tb[1]:tb[3], tb[0]:tb[2]].astype(np.float32)
    diff = np.abs(_lum(crop) - bg_lum) if crop.size else np.zeros((1, 1))
    ink = diff > 55
    if ink.sum() > 10:
        strong = diff >= np.percentile(diff[ink], 60)
        ink_color = np.median(crop[ink & strong], axis=0)
    else:
        ink_color = np.array([0, 0, 0] if bg_lum > 128 else [255, 255, 255], float)
    # Keep enough contrast between text and background
    if abs(float(_lum(ink_color)) - bg_lum) < 90:
        ink_color = np.array([20, 20, 20] if bg_lum > 128 else [245, 245, 245], float)

    # Size of the original lettering, from the text area and the number of characters read by the OCR.
    # (Projection profiles fail on tight leading and on Hangul, whose syllables are stacked pieces.)
    tw, th = max(1, tb[2] - tb[0]), max(1, tb[3] - tb[1])
    text = re.sub(r"\s", "", it["original"])
    n = max(1, len(text))
    if CJK_CHARS.search(text):
        char_px = np.sqrt(tw * th / (n * 1.45))            # square characters plus spacing
        char_px = min(char_px, tw if it.get("vertical") else th)
    else:
        char_px = np.sqrt(tw * th / (n * 1.2))             # Latin/Cyrillic: cap height
        char_px = min(char_px, th)

    # Ink that belongs to the bubble outline or panel borders (long strokes touching the text box edge)
    # must not be erased.
    ink_u8 = ink.astype(np.uint8)
    if ink_u8.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink_u8, connectivity=8)
        for i in range(1, count):
            x, y, w, h, _area = stats[i]
            touches = x == 0 or y == 0 or x + w >= tw or y + h >= th
            if touches and max(w, h) > 2.2 * char_px:
                ink[labels == i] = False
    # Inside a bubble, only erase within the bubble outline (the text box can stick out over the art)
    if bubble:
        bx0, by0, bx1, by1 = bubble
        local = (bx0 - tb[0], by0 - tb[1], bx1 - tb[0], by1 - tb[1])
        ink &= _ellipse_mask(ink.shape, local, 0.97) if _is_oval(img, bubble, tb, bg_lum) else \
            _rect_mask(ink.shape, local, 3)

    shape = ("ellipse" if _is_oval(img, bubble, tb, bg_lum) else "rect") if bubble else "rect"

    return {"tb": tb, "bubble": bubble, "bg": bg, "flat": flat, "ink": ink, "ink_color": ink_color,
            "char_px": float(char_px), "shape": shape}


def _style(it: dict, a: dict) -> str:
    orig, tr = it["original"], it["translation"]
    if not a["bubble"] and not a["flat"]:
        letters = re.sub(r"[\W\d_]", "", orig)
        cjk = len(CJK_CHARS.findall(orig))
        if (cjk and cjk <= 5) or (not cjk and len(letters) <= 10 and " " not in orig.strip()):
            return "sfx"
        return "note"
    if re.search(r"[!！]{2,}|[!！][?？]|[?？][!！]", orig) or tr.rstrip().endswith("!!"):
        return "shout"
    if re.match(r"^\s*[(（]", orig) and re.search(r"[)）]\s*$", orig):
        return "thought"
    if not a["bubble"]:
        return "narration"   # captions and boxes outside speech bubbles
    return "speech"


# --------------------------------------------------------------------------- erasing

def _erase(img: np.ndarray, it: dict, a: dict):
    x0, y0, x1, y1 = a["tb"]
    if x1 <= x0 or y1 <= y0:
        return
    k = max(2, int(a["char_px"] * 0.12))
    mask = cv2.dilate(a["ink"].astype(np.uint8) * 255, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    if a["flat"]:
        # Flat bubble or box: repaint the text with the background color
        region = img[y0:y1, x0:x1]
        region[mask > 0] = a["bg"].astype(np.uint8)
    else:
        # Text on artwork: rebuild the picture behind the text
        pad = 12
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
        full = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        full[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = mask
        patch = cv2.inpaint(np.ascontiguousarray(img[Y0:Y1, X0:X1]), full, 5, cv2.INPAINT_TELEA)
        img[Y0:Y1, X0:X1] = patch


# --------------------------------------------------------------------------- layout

def _area(it: dict, a: dict, W: int, H: int):
    """(x0, y0, x1, y1, shape) where the translation is lettered."""
    x0, y0, x1, y1 = a["tb"]
    if a["bubble"]:
        bx0, by0, bx1, by1 = a["bubble"]
        if a["shape"] == "ellipse":
            # stay inside the bubble outline (the box also contains the tail and the border)
            sx, sy = (bx1 - bx0) * 0.08, (by1 - by0) * 0.09
        else:
            sx, sy = (bx1 - bx0) * 0.06, (by1 - by0) * 0.08
        return bx0 + sx, by0 + sy, bx1 - sx, by1 - sy, a["shape"]
    w, h = x1 - x0, y1 - y0
    if it.get("vertical") and w < h * 0.7:  # vertical CJK text: English needs more width
        grow = (h * 0.7 - w) / 2
        x0, x1 = max(0, x0 - grow), min(W, x1 + grow)
    return x0, y0, x1, y1, "rect"


def _line_widths(n, line_h, x0, y0, x1, y1, shape):
    """Usable width of each of n centered lines."""
    w, h = x1 - x0, y1 - y0
    top = y0 + (h - n * line_h) / 2
    out = []
    for i in range(n):
        if shape != "ellipse":
            out.append(w)
            continue
        # narrowest point of the line inside the ellipse
        ya, yb = top + i * line_h, top + (i + 1) * line_h
        dy = max(abs(ya - (y0 + h / 2)), abs(yb - (y0 + h / 2))) / (h / 2)
        out.append(w * np.sqrt(max(0.0, 1 - dy * dy)))
    return out


def _wrap(words, widths, font):
    lines, i = [], 0
    space = font.getlength(" ")
    for wmax in widths:
        line, lw = [], 0.0
        while i < len(words):
            wl = font.getlength(words[i])
            add = wl if not line else lw + space + wl
            if add <= wmax:
                line.append(words[i])
                lw = add
                i += 1
            else:
                break
        if not line and i < len(words):
            return None  # a single word does not fit on this line
        lines.append(" ".join(line))
    return lines if i == len(words) else None


def _fit(text: str, style: str, box, size_hint: float):
    """Largest font size (up to the original lettering size) and line breaks that fit the area."""
    x0, y0, x1, y1, shape = box
    words = text.split()
    if not words:
        return None
    spacing = STYLES[style][2]
    hi = int(max(MIN_SIZE, min(size_hint, (y1 - y0) * 0.9)))
    for size in range(hi, MIN_SIZE - 1, -1):
        font = _font(style, size)
        line_h = size * spacing
        for n in range(1, int((y1 - y0) // line_h) + 1):
            lines = _wrap(words, _line_widths(n, line_h, *box), font)
            if lines:
                return font, lines, line_h
    font = _font(style, MIN_SIZE)  # does not fit even at the minimum: rectangle, may overflow
    line_h = MIN_SIZE * spacing
    widths = [x1 - x0] * len(words)
    lines = _wrap(words, widths, font) or words
    return font, [l for l in lines if l], line_h


def _draw(draw: ImageDraw.ImageDraw, text: str, style: str, box, a: dict, size_hint: float, outline: bool):
    fitted = _fit(text, style, box, size_hint)
    if not fitted:
        return
    font, lines, line_h = fitted
    x0, y0, x1, y1, _ = box
    cx = (x0 + x1) / 2
    top = y0 + ((y1 - y0) - len(lines) * line_h) / 2
    fill = tuple(int(v) for v in a["ink_color"])
    stroke = 0
    stroke_fill = None
    if outline:
        stroke = max(2, int(font.size * 0.12))
        stroke_fill = (255, 255, 255) if _lum(np.array(fill, float)) < 128 else (15, 15, 15)
    for i, line in enumerate(lines):
        draw.text((cx, top + (i + 0.5) * line_h), line, font=font, fill=fill, anchor="mm",
                  stroke_width=stroke, stroke_fill=stroke_fill)


# --------------------------------------------------------------------------- API

def _legacy_boxes(it: dict) -> dict:
    """Translations cached by older versions only have the overlay rectangle."""
    if "text_box" not in it:
        it = dict(it, text_box=[it["x"], it["y"], it["x"] + it["w"], it["y"] + it["h"]], bubble=None)
    return it


def render_page(image_bytes: bytes, data: dict, hide_sfx: bool = False, fmt: str = "PNG") -> bytes:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil)
    H, W = img.shape[:2]

    items = [_legacy_boxes(it) for it in data.get("items", [])
             if it["original"].strip() and it["original"].strip() != it["translation"].strip()
             and not WATERMARK.search(it["original"])]
    plans = []
    for it in items:
        a = _analyse(img, it)
        plans.append((it, a, _style(it, a)))

    # Pass 1: erase every original text first (areas can overlap)
    for it, a, style in plans:
        if style != "sfx":
            _erase(img, it, a)

    # Pass 2: letter the translations
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    for it, a, style in plans:
        cjk_source = bool(CJK_CHARS.search(it["original"]))
        # English glyphs look as large as CJK ones at ~95% of the column width; Latin originals are
        # mostly capitals, whose height is ~70% of the font size.
        size_hint = a["char_px"] * (0.95 if cjk_source else 1.35)
        if style == "sfx":
            if hide_sfx:
                continue
            x0, y0, x1, y1 = a["tb"]
            w, h = x1 - x0, y1 - y0
            box = (x0 - w * 0.3, y0 + h * 0.3, x1 + w * 0.3, y1 - h * 0.3, "rect")
            _draw(draw, it["translation"].upper(), "sfx", box, a, max(14, min(w, h) * 0.45), outline=True)
            continue
        box = _area(it, a, W, H)
        text = it["translation"].upper() if style == "shout" else it["translation"]
        _draw(draw, text, style, box, a, size_hint, outline=not a["flat"])

    buf = io.BytesIO()
    if fmt.upper() == "WEBP":
        out.save(buf, format="WEBP", quality=92, method=4)
    else:
        out.save(buf, format="PNG")
    return buf.getvalue()
