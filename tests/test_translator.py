"""Pure helpers of the translation pipeline (no model is loaded)."""
from PIL import Image, ImageDraw

import translator as t


def test_tall_pages_are_sliced_with_ownership_bands():
    assert t._slices(1000, 800) == [(0, 1000, 0, 1000)]
    slices = t._slices(8000, 800)
    assert len(slices) > 1 and slices[0][0] == 0 and slices[-1][1] == 8000
    # every y belongs to exactly one slice
    owned = [(ot, ob) for _, _, ot, ob in slices]
    assert all(a[1] == b[0] for a, b in zip(owned, owned[1:]))


def test_nms_keeps_best_of_overlapping_boxes():
    dets = [("text_bubble", 0.9, [0, 0, 100, 100]), ("text_bubble", 0.5, [5, 5, 95, 95]),
            ("text_free", 0.8, [200, 200, 300, 300])]
    assert [d[1] for d in t._nms(dets)] == [0.9, 0.8]


def test_overlay_box_uses_enclosing_bubble():
    box, bubble = t._overlay_box([40, 40, 60, 60], [[0, 0, 100, 100], [500, 500, 600, 600]])
    assert bubble == [0, 0, 100, 100]
    assert box[0] >= 0 and box[2] <= 100


def test_latin_language_detection():
    assert t._latin_language(["¿QUÉ ESTÁS HACIENDO?"]) == "es"
    assert t._latin_language(["Você não sabe isso"]) == "pt"
    assert t._latin_language(["What are you doing here?"]) == "en"


def test_ocr_clean_up():
    assert t._clean_ocr("ЗАЙН ПОПРО- БОВАЛ; ЯДА HET", "ru") == "ЗАЙН ПОПРОБОВАЛ; ЯДА НЕТ"
    assert t._clean_ocr("POR CIERTO, ZHAY ALGUN LIMITE?", "es") == "POR CIERTO, ¿HAY ALGUN LIMITE?"


def test_english_tidy_up():
    assert t._tidy_english("there 's no poison in it .") == "There's no poison in it."
    assert t._tidy_english("i want to go, i think") == "I want to go, I think"
    assert t._tidy_english("- Liar.") == "Liar."
    assert t._tidy_english("♪ magic magic ♪") == "Magic magic"


def test_sentence_case_for_all_caps_lettering():
    assert t._sentence_case("¿QUÉ ESTÁS HACIENDO? NO LO SÉ.") == "¿Qué estás haciendo? No lo sé."


def test_vertical_columns_are_reflowed_into_a_line():
    img = Image.new("RGB", (60, 200), "white")
    d = ImageDraw.Draw(img)
    for col_x in (10, 35):                 # two columns of three "characters"
        for y in (10, 70, 130):
            d.rectangle([col_x, y, col_x + 15, y + 40], fill="black")
    line = t._reflow_vertical(img)
    assert line.width > line.height        # now a horizontal line


def test_language_menu():
    codes = [o["code"] for o in t.language_options()]
    assert codes[0] == "auto" and {"ja", "zh", "zh-hant", "ko", "es", "pt", "ru"} <= set(codes)
