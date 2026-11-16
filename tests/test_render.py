"""Typesetting on a synthetic page: a white oval bubble with black 'text' on a grey background."""
import io

import numpy as np
from PIL import Image, ImageDraw

import render


def _page():
    img = Image.new("RGB", (600, 800), (150, 150, 150))
    d = ImageDraw.Draw(img)
    d.ellipse([100, 100, 500, 400], fill="white", outline="black", width=4)
    for y in (200, 240, 280):                      # three lines of fake glyphs
        for x in range(220, 380, 22):
            d.rectangle([x, y, x + 14, y + 22], fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


ITEM = {
    "x": 0.3, "y": 0.2, "w": 0.4, "h": 0.2, "kind": "speech",
    "original": "これはテストです", "translation": "This is a test!",
    "text_box": [210 / 600, 190 / 800, 390 / 600, 310 / 800],
    "bubble": [100 / 600, 100 / 800, 500 / 600, 400 / 800], "vertical": False,
}


def _render(items, **kw):
    out = render.render_page(_page(), {"items": items}, **kw)
    return np.array(Image.open(io.BytesIO(out)).convert("RGB")).astype(int)


def test_bubble_is_erased_and_lettered():
    before = np.array(Image.open(io.BytesIO(_page())).convert("RGB")).astype(int)
    after = _render([ITEM])
    glyphs = (slice(200, 302), slice(220, 380))
    assert (before[glyphs] < 50).mean() > 0.2           # the fake text was dense
    assert (after[glyphs] < 50).mean() < 0.2            # replaced by thinner lettering
    assert (after[glyphs] < 100).any()                  # ... and there is new text
    outline = (slice(98, 106), slice(290, 310))         # top of the bubble outline
    assert (after[outline] < 80).any()                  # the outline is kept
    assert np.abs(after[600:, :] - before[600:, :]).max() == 0   # nothing outside the bubble changed


def test_bubble_shape_and_style():
    img = np.array(Image.open(io.BytesIO(_page())).convert("RGB"))
    a = render._analyse(img, ITEM)
    assert a["shape"] == "ellipse" and a["flat"]
    assert render._style(ITEM, a) == "speech"
    assert render._style(dict(ITEM, original="なに！？"), a) == "shout"
    assert render._style(dict(ITEM, original="（そうか）"), a) == "thought"


def test_watermarks_and_untranslated_text_are_left_alone():
    before = np.array(Image.open(io.BytesIO(_page())).convert("RGB")).astype(int)
    for item in (dict(ITEM, original="t.me/some_team"), dict(ITEM, translation=ITEM["original"])):
        assert np.abs(_render([item]) - before).max() == 0


def test_text_fits_inside_an_ellipse():
    box = (100, 100, 500, 400, "ellipse")
    font, lines, line_h = render._fit("word " * 40, "speech", box, 40)
    widths = render._line_widths(len(lines), line_h, *box)
    assert all(font.getlength(l) <= w + 1 for l, w in zip(lines, widths))
    assert len(lines) * line_h <= 300
