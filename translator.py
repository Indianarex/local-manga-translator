"""Fully local translation (no API key): bubble detection -> OCR -> English translation.

Source languages: Japanese, Chinese (simplified / traditional), Korean, Spanish, Portuguese, Russian
(plus automatic detection). Models are downloaded automatically on first use into ./models:
  - ogkalu/comic-text-and-bubble-detector  : speech bubble and text area detection (RT-DETR v2)
  - kha-white/manga-ocr-base               : Japanese OCR tuned for manga (vertical text, furigana...)
  - EasyOCR                                : Chinese, Korean, Latin (es/pt) and Cyrillic (ru) OCR
  - LiquidAI/LFM2-350M-ENJP-MT             : Japanese -> English translation
  - facebook/nllb-200-distilled-1.3B       : translation of the other languages -> English
"""
from __future__ import annotations

import io
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
EASYOCR_DIR = ROOT / "models" / "easyocr"

import numpy as np
from PIL import Image

# Refuse "decompression bombs" (small files that decode into huge images): manga pages are a few
# megapixels and long webtoon strips stay well under this. Pillow raises an error beyond twice the limit.
Image.MAX_IMAGE_PIXELS = 60_000_000

DETECTOR_ID = "ogkalu/comic-text-and-bubble-detector"
JA_OCR_ID = "kha-white/manga-ocr-base"
JA_MT_ID = "LiquidAI/LFM2-350M-ENJP-MT"
NLLB_ID = "facebook/nllb-200-distilled-1.3B"
# Exact model revisions: an update on Hugging Face cannot change or break the app. For the two models
# published only as pytorch_model.bin, these are the commits that add a model.safetensors file, so the
# weights are downloaded once, in the safe format (otherwise both files end up downloaded).
REVISIONS = {
    DETECTOR_ID: "16e8a622f91fabc6b5b65c96d32d1183f8843546",
    JA_OCR_ID: "82c77fb84369e0fce0d6e8f0ad96fa82646d0346",   # refs/pr/4 (safetensors)
    JA_MT_ID: "80367784d525777ad7565b24534ba5810eeac59f",
    NLLB_ID: "f9e6ff7fc090740fcce5af7d7636cb27a33d0f18",     # refs/pr/5 (safetensors)
}


def _hub(model_id: str) -> dict:
    """Arguments for from_pretrained: pinned revision, safetensors weights only."""
    return {"revision": REVISIONS[model_id]}


def _weights(model_id: str) -> dict:
    return {"revision": REVISIONS[model_id], "use_safetensors": True}

FORMAT = 3            # version of the translation result; older cached results are recomputed
DET_THRESHOLD = 0.35
TALL_RATIO = 2.2      # taller than this (webtoon strips), the image is cut into slices
SLICE_RATIO = 1.5     # slice height = 1.5 x width
OVERLAP = 0.15

# code -> (display name, NLLB code, EasyOCR group)
LANGUAGES = {
    "ja": ("Japanese", "jpn_Jpan", "ja"),
    "zh": ("Chinese (simplified)", "zho_Hans", "ch_sim"),
    "zh-hant": ("Chinese (traditional)", "zho_Hant", "ch_tra"),
    "ko": ("Korean", "kor_Hang", "ko"),
    "es": ("Spanish", "spa_Latn", "latin"),
    "pt": ("Portuguese", "por_Latn", "latin"),
    "ru": ("Russian", "rus_Cyrl", "ru"),
    "en": ("English", "eng_Latn", "latin"),
}
EASYOCR_LANGS = {
    "ja": ["ja", "en"], "ch_sim": ["ch_sim", "en"], "ch_tra": ["ch_tra", "en"],
    "ko": ["ko", "en"], "ru": ["ru", "en"], "latin": ["es", "pt", "en"],
}

RE_KANA = re.compile(r"[぀-ヿｦ-ﾟ]")
RE_HAN = re.compile(r"[㐀-鿿豈-﫿]")
RE_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
RE_CYR = re.compile(r"[Ѐ-ӿ]")
RE_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
RE_LETTER = re.compile(r"[^\W\d_]")

STOPWORDS = {
    "es": set("que el la los las un una y de en no es por con para lo se mi tu yo eso esto qué está estás pero muy "
              "usted también aquí ahora así porque dónde cómo hay ya del al le les nos".split()),
    "pt": set("que o a os as um uma e de em não é por com para se meu eu isso isto você está mas muito também aqui "
              "agora assim porque onde como há já do da dos das no na nos ele ela".split()),
    "en": set("the a an and of to in is it you i that this what are was for with on my your be not have do don't "
              "i'm it's we he she they".split()),
}


class TranslationError(Exception):
    pass


def language_options() -> list[dict]:
    return [{"code": "auto", "name": "Auto-detect"}] + [
        {"code": c, "name": n} for c, (n, _, _) in LANGUAGES.items() if c != "en"
    ]


# --------------------------------------------------------------------------- models

class _Models:
    """Lazily loaded models; a lock serializes GPU inference."""

    def __init__(self):
        self.lock = threading.RLock()
        self.stage = "waiting"
        self.error: str | None = None
        self.base_loaded = False
        self._readers: dict[str, object] = {}
        self._ja = None
        self._ja_mt = None
        self._nllb = None

    # ---- base: torch + bubble detector
    def load_base(self):
        with self.lock:
            if self.base_loaded:
                return
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModelForObjectDetection

                self.torch = torch
                if torch.cuda.is_available():
                    self.device = "cuda"                       # NVIDIA GPU
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self.device = "mps"                        # Apple Silicon GPU
                else:
                    self.device = "cpu"
                self.half = torch.float32 if self.device == "cpu" else torch.float16
                self.stage = "loading the bubble detector"
                self.det_proc = AutoImageProcessor.from_pretrained(DETECTOR_ID, **_hub(DETECTOR_ID))
                self.det = AutoModelForObjectDetection.from_pretrained(
                    DETECTOR_ID, **_weights(DETECTOR_ID)).to(self.device).eval()
                self.base_loaded = True
                self.stage = f"ready ({ {'cuda': 'GPU', 'mps': 'Apple GPU'}.get(self.device, 'CPU') })"
                self.error = None
            except Exception as e:  # noqa: BLE001
                self.error = f"Could not load the models: {e}"
                self.stage = "error"
                raise TranslationError(self.error) from e

    def _step(self, label, fn):
        prev = self.stage
        self.stage = label
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            raise TranslationError(f"Failed ({label}): {e}") from e
        finally:
            self.stage = prev

    # ---- EasyOCR (one reader per script, sharing a single CRAFT text detector)
    def reader(self, group: str):
        if group not in self._readers:
            def make():
                import easyocr

                EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
                shared = next(iter(self._readers.values()), None)
                r = easyocr.Reader(
                    EASYOCR_LANGS[group], gpu=self.device if self.device != "cpu" else False,
                    model_storage_directory=str(EASYOCR_DIR), detector=shared is None, verbose=False,
                )
                if shared is not None:  # reuse the CRAFT detector that is already loaded (saves VRAM)
                    r.detector = shared.detector
                    r.get_textbox = shared.get_textbox
                    r.get_detector = shared.get_detector
                return r
            self._readers[group] = self._step(f"loading the {group} OCR", make)
        return self._readers[group]

    # ---- Japanese: manga-ocr (also used to recognize Japanese pages) and LFM2, loaded separately so the
    # translation model is only loaded when a page really is Japanese
    def ja_ocr(self):
        if self._ja is None:
            def make():
                from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

                proc = ViTImageProcessor.from_pretrained(JA_OCR_ID, **_hub(JA_OCR_ID))
                tok = AutoTokenizer.from_pretrained(JA_OCR_ID, **_hub(JA_OCR_ID))
                ocr = VisionEncoderDecoderModel.from_pretrained(
                    JA_OCR_ID, **_weights(JA_OCR_ID)).to(self.device, dtype=self.half).eval()
                return proc, tok, ocr
            self._ja = self._step("loading the Japanese OCR", make)
        return self._ja

    def ja_mt(self):
        if self._ja_mt is None:
            def make():
                from transformers import AutoModelForCausalLM, AutoTokenizer

                t = self.torch
                tok = AutoTokenizer.from_pretrained(JA_MT_ID, **_hub(JA_MT_ID))
                tok.padding_side = "left"
                dtype = {"cuda": t.bfloat16, "mps": t.float16}.get(self.device, t.float32)
                mt = AutoModelForCausalLM.from_pretrained(
                    JA_MT_ID, dtype=dtype, **_weights(JA_MT_ID)).to(self.device).eval()
                return tok, mt
            self._ja_mt = self._step("loading the Japanese translation model", make)
        return self._ja_mt

    def ja(self):  # both Japanese models (kept for scripts and tests)
        return (*self.ja_ocr(), *self.ja_mt())

    # ---- NLLB (every other language)
    def nllb(self):
        if self._nllb is None:
            def make():
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(NLLB_ID, **_hub(NLLB_ID))
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    NLLB_ID, dtype=self.half, **_weights(NLLB_ID)).to(self.device).eval()
                return tok, model
            self._nllb = self._step("loading the multilingual translation model (5 GB download on first use)", make)
        return self._nllb

    # ------------------------------------------------------------------ inference
    def detect(self, img: Image.Image):
        t = self.torch
        inputs = self.det_proc(images=img, return_tensors="pt").to(self.device)
        with t.inference_mode():
            out = self.det(**inputs)
        res = self.det_proc.post_process_object_detection(
            out, target_sizes=t.tensor([img.size[::-1]], device=self.device), threshold=DET_THRESHOLD
        )[0]
        labels = self.det.config.id2label
        return [
            (labels[int(l)], float(s), [float(v) for v in b])
            for s, l, b in zip(res["scores"].cpu(), res["labels"].cpu(), res["boxes"].cpu())
        ]

    def ocr_ja(self, crops):
        proc, tok, ocr = self.ja_ocr()
        out = []
        for i in range(0, len(crops), 16):
            batch = [c.convert("L").convert("RGB") for c in crops[i:i + 16]]
            pv = proc(batch, return_tensors="pt").pixel_values.to(self.device, dtype=self.half)
            with self.torch.inference_mode():
                ids = ocr.generate(pv, max_length=300)
            for seq in ids:
                text = "".join(tok.decode(seq, skip_special_tokens=True).split()).replace("…", "...")
                out.append(re.sub("[・.]{2,}", lambda m: (m.end() - m.start()) * ".", text))
        return out

    def ocr_easy(self, group, crop: Image.Image, cache: dict | None = None):
        """Return (text, mean confidence) for one text area.

        Finding the text lines (CRAFT) is the slow part and does not depend on the language, so with a
        `cache` dict it is done once per area and shared by every script tried on it; results are cached too.
        """
        cache = {} if cache is None else cache
        reflow = group in ("ch_sim", "ch_tra") and crop.height > 1.3 * crop.width
        if (group, id(crop)) in cache:
            return cache[(group, id(crop))]
        key = ("lines", id(crop), reflow)
        if key not in cache:
            img = crop.convert("RGB")
            if reflow:
                img = _reflow_vertical(img)  # EasyOCR only reads horizontal lines
            if img.height < 80 or img.width < 80:  # small text reads better when upscaled
                f = 80 / min(img.height, img.width)
                img = img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)
            from easyocr.utils import reformat_input

            rgb, grey = reformat_input(np.array(img))
            horizontal, free = self.reader(group).detect(rgb, reformat=False)
            cache[key] = (grey, horizontal[0], free[0])
        grey, horizontal, free = cache[key]
        res = []
        if horizontal or free:
            res = self.reader(group).recognize(grey, horizontal, free, detail=1, paragraph=False, reformat=False)
        cache[(group, id(crop))] = out = self._join_lines(group, res)
        return out

    @staticmethod
    def _join_lines(group, res):
        if not res:
            return "", 0.0
        # sort into lines (top -> bottom), then left -> right
        res.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        sep = "" if group in ("ch_sim", "ch_tra", "ja") else " "
        text = sep.join(r[1] for r in res).strip()
        n = sum(len(r[1]) for r in res) or 1
        conf = sum(r[2] * len(r[1]) for r in res) / n
        return text, conf

    def translate_ja(self, texts):
        tok, mt = self.ja_mt()
        out = []
        for i in range(0, len(texts), 16):
            chunk = texts[i:i + 16]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": "Translate to English."}, {"role": "user", "content": s}],
                    add_generation_prompt=True, tokenize=False,
                )
                for s in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with self.torch.inference_mode():
                gen = mt.generate(**enc, max_new_tokens=160, do_sample=False, repetition_penalty=1.05,
                                  pad_token_id=tok.pad_token_id)
            out += [_tidy_english(tok.decode(r, skip_special_tokens=True)) for r in gen[:, enc["input_ids"].shape[1]:]]
        return out

    def translate_nllb(self, texts, lang):
        tok, model = self.nllb()
        tok.src_lang = LANGUAGES[lang][1]
        out = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            # The output budget follows the input length, which stops NLLB from looping on noisy OCR text
            budget = min(200, int(enc["input_ids"].shape[1] * 2.5) + 10)
            with self.torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                                     max_new_tokens=budget, max_length=None, num_beams=4,
                                     no_repeat_ngram_size=4, repetition_penalty=1.1)
            out += [_tidy_english(s) for s in tok.batch_decode(gen, skip_special_tokens=True)]
        return out


MODELS = _Models()


def status() -> dict:
    return {"ready": MODELS.base_loaded, "stage": MODELS.stage, "error": MODELS.error}


def preload():
    try:
        MODELS.load_base()
    except TranslationError:
        pass


# --------------------------------------------------------------------------- language detection

def _script_ratio(text: str, pattern) -> float:
    letters = RE_LETTER.findall(text)
    return len(pattern.findall(text)) / len(letters) if letters else 0.0


def _latin_language(texts: list[str]) -> str:
    words = re.findall(r"[a-zà-ÿ']+", " ".join(texts).lower())
    scores = {l: sum(w in sw for w in words) for l, sw in STOPWORDS.items()}
    blob = " ".join(texts).lower()
    scores["es"] += 3 * len(re.findall("[ñ¿¡]", blob))
    scores["pt"] += 3 * len(re.findall("[ãõç]|ção|ões", blob))
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "es"


def _letter_density(text: str) -> float:
    """Share of letters among non-space characters: OCR noise ('88bf 1 #') scores low."""
    chars = [ch for ch in text if not ch.isspace()]
    return len(RE_LETTER.findall(text)) / len(chars) if chars else 0.0


def _kana_share(texts: list[str]) -> float:
    letters = sum(len(RE_LETTER.findall(t)) for t in texts)
    return sum(len(RE_KANA.findall(t)) for t in texts) / letters if letters else 0.0


def _kana_density(texts: list[str]) -> float:
    """Share of kana among all characters read by manga-ocr. On non-Japanese text manga-ocr outputs
    garbage full of digits and symbols, which keeps this low (Korean ~0.14, Japanese ~0.6)."""
    chars = sum(len([ch for ch in t if not ch.isspace()]) for t in texts)
    return sum(len(RE_KANA.findall(t)) for t in texts) / chars if chars else 0.0


_FAMILY = {"ja": "ja", "ch_sim": "zh", "ch_tra": "zh", "ko": "ko", "ru": "ru", "latin": "latin"}
_SCRIPTS = {"ja": RE_KANA, "ch_sim": RE_HAN, "ch_tra": RE_HAN, "ko": RE_HANGUL, "ru": RE_CYR, "latin": RE_LATIN}


def detect_language(crops: list[Image.Image], cache: dict | None = None) -> tuple[str, float]:
    """Find the language of a page from a few of its text areas.

    Returns (language, confidence): confidence goes from 0 (two scripts score the same) to 1 (only one
    script reads anything sensible)."""
    sample = sorted(crops, key=lambda c: -c.width * c.height)[:3]
    if not sample:
        return "ja", 0.0

    # 1. Japanese, the most common case, in one fast pass: manga-ocr (which also reads vertical text)
    #    on Japanese dialogue gives text full of kana.
    vertical = sum(c.height > 1.3 * c.width for c in sample) >= len(sample) / 2
    kana = _kana_density(MODELS.ocr_ja(sample))
    if kana >= 0.35 or (vertical and kana >= 0.2):
        return "ja", min(1.0, 0.5 + kana)

    # 2. Otherwise every other script is tried with EasyOCR, and the most consistent reading wins.
    #    EasyOCR cannot read vertical columns (except Chinese, which is reflowed into lines first).
    #    Traditional Chinese is only tried when simplified Chinese is ahead: it is the same script.
    groups = ["latin", "ru", "ko", "ch_sim"] + (["ja"] if kana >= 0.1 and not vertical else [])

    def score(group: str) -> tuple[float, list[str]]:
        s, ts = 0.0, []
        for c in sample:
            text, conf = MODELS.ocr_easy(group, c, cache)
            ts.append(text)
            ratio = _script_ratio(text, _SCRIPTS[group])
            if group == "ja":  # Japanese is recognized by its kana, mixed with kanji
                ratio = min(1.0, (len(RE_KANA.findall(text)) + len(RE_HAN.findall(text)))
                            / max(1, len(RE_LETTER.findall(text)))) if RE_KANA.search(text) else 0.0
            if group == "latin":  # real Spanish/Portuguese/English text contains common words
                words = re.findall(r"[a-zà-ÿ']+", text.lower())
                known = sum(any(w in sw for sw in STOPWORDS.values()) for w in words)
                ratio *= min(1.0, 0.3 + known / max(1, len(words)))
            s += conf * ratio * _letter_density(text) * max(1, len(text)) ** 0.5
        return s, ts

    scores, texts = {}, {}
    for group in groups:
        scores[group], texts[group] = score(group)
    if max(scores, key=scores.get) == "ch_sim":
        scores["ch_tra"], texts["ch_tra"] = score("ch_tra")
    best = max(scores, key=scores.get)
    # confidence: lead over the best group of another script family (simplified vs traditional Chinese
    # use the same script, so their scores are always close)
    rival = max((v for g, v in scores.items() if _FAMILY[g] != _FAMILY[best]), default=0.0)
    confidence = (scores[best] - rival) / scores[best] if scores[best] > 0 else 0.0
    if best == "latin":
        return _latin_language(texts["latin"]), confidence
    return {"ja": "ja", "ch_sim": "zh", "ch_tra": "zh-hant", "ko": "ko", "ru": "ru"}[best], confidence


# Below this confidence, the language most often found in the chapter (or announced by the site) wins
HINT_CONFIDENCE = 0.35


# --------------------------------------------------------------------------- vertical text

def _runs(mask_1d: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """[start, end) ranges where mask is true, merging ranges separated by fewer than min_gap."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            runs.append((start, prev + 1))
            start = i
        prev = i
    runs.append((start, prev + 1))
    return runs


def _reflow_vertical(img: Image.Image) -> Image.Image:
    """Turn vertical text (columns read right to left) into one horizontal line of upright characters."""
    gray = np.asarray(img.convert("L"))
    ink = gray < 128
    if ink.sum() < 20:
        return img
    cols = [(a, b) for a, b in _runs(ink.sum(axis=0) > 0, min_gap=max(2, img.width // 40)) if b - a >= 4]
    if not cols:
        return img
    cells = []
    for x0, x1 in sorted(cols, reverse=True):          # right-to-left column order
        w = x1 - x0
        col = ink[:, x0:x1]
        # Characters are about one column-width tall; small gaps inside a character are merged.
        for y0, y1 in _runs(col.sum(axis=1) > 0, min_gap=max(2, int(w * 0.3))):
            h = y1 - y0
            n = max(1, round(h / (w * 1.05))) if h > 1.4 * w else 1
            for k in range(n):
                a, b = y0 + k * h // n, y0 + (k + 1) * h // n
                cells.append(Image.fromarray(gray[a:b, x0:x1]))
    if not cells:
        return img
    size = max(max(c.width, c.height) for c in cells)
    pad = size // 6
    line = Image.new("L", (len(cells) * (size + pad) + pad, size + 2 * pad), 255)
    x = pad
    for c in cells:
        line.paste(c, (x + (size - c.width) // 2, pad + (size - c.height) // 2))
        x += size + pad
    return line.convert("RGB")


# --------------------------------------------------------------------------- text clean-up

def _tidy_english(text: str) -> str:
    """Undo tokenizer spacing in the translation: "there 's" -> "there's", "it ." -> "it."."""
    text = text.replace("♪", " ")                     # subtitle-style music notes
    text = re.sub(r"\s+([.,!?;:)\]])", r"\1", text)
    text = re.sub(r"(\w) n't\b", r"\1n't", text)
    text = re.sub(r"(\w) '(s|re|ve|ll|d|m|t)\b", r"\1'\2", text)
    text = re.sub(r"^\s*[-–—]\s*", "", text)    # leading dialogue dash
    text = re.sub(r"\bi\b(?!\.)", "I", text)               # lone lowercase "i"
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:1].upper() + text[1:]


_CYR_LOOKALIKES = str.maketrans("ABCEHKMOPTXYaceopxy", "АВСЕНКМОРТХУасеорху")


def _clean_ocr(text: str, lang: str) -> str:
    if lang in ("ru", "es", "pt", "en"):
        # words split across lines with a hyphen: "ПОПРО- БОВАЛ" -> "ПОПРОБОВАЛ"
        text = re.sub(r"(\w)[-\u2010\u2011]\s+(\w)", r"\1\2", text)
    if lang == "ru":
        # The Russian OCR sometimes returns Latin letters that look identical to Cyrillic ones
        def fix(m):
            word = m.group(0)
            if RE_CYR.search(word) or all(ch in "ABCEHKMOPTXYaceopxy" for ch in word):
                return word.translate(_CYR_LOOKALIKES)
            return word
        text = re.sub(r"[^\W\d_]+", fix, text)
    if lang == "es" and "?" in text:
        # an inverted question mark is often read as '2' or 'Z' in front of an all-caps word
        text = re.sub(r"(^|[\s,.!\"'])[2Z](?=[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00d1]{2,})", "\\1\u00bf", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# --------------------------------------------------------------------------- geometry

def _slices(h: int, w: int):
    """[(top, bottom, own_top, own_bottom)]; 'own' is the band each slice is responsible for."""
    if h <= w * TALL_RATIO:
        return [(0, h, 0, h)]
    slice_h = int(w * SLICE_RATIO)
    ov = int(slice_h * OVERLAP)
    out, top = [], 0
    while True:
        bottom = min(h, top + slice_h)
        out.append((top, bottom))
        if bottom >= h:
            break
        top += slice_h - ov
    own_tops = [0] + [t + ov // 2 for t, _ in out[1:]]
    own_bottoms = own_tops[1:] + [h]
    return [(t, b, ot, ob) for (t, b), ot, ob in zip(out, own_tops, own_bottoms)]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _nms(dets, iou=0.5):
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    for d in dets:
        if all(_inter(d[2], k[2]) / (min(_area(d[2]), _area(k[2])) or 1) < iou for k in keep):
            keep.append(d)
    return keep


def _overlay_box(text_box, bubbles):
    """Area to cover: the inside of the bubble holding the text (inscribed rectangle), else the padded text box."""
    x0, y0, x1, y1 = text_box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    best = None
    for b in bubbles:
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and _inter(text_box, b) > 0.5 * _area(text_box):
            if best is None or _area(b) < _area(best):
                best = b
    pad_x, pad_y = (x1 - x0) * 0.08 + 2, (y1 - y0) * 0.05 + 2
    box = [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y]
    if best is not None:
        bw, bh = best[2] - best[0], best[3] - best[1]
        inset = [best[0] + bw * 0.14, best[1] + bh * 0.14, best[2] - bw * 0.14, best[3] - bh * 0.14]
        box = [min(box[0], inset[0]), min(box[1], inset[1]), max(box[2], inset[2]), max(box[3], inset[3])]
        box = [max(box[0], best[0]), max(box[1], best[1]), min(box[2], best[2]), min(box[3], best[3])]
    return box, best


def _sentence_case(text: str) -> str:
    """Latin-script comics are often lettered in ALL CAPS; the models translate normal casing better."""
    letters = RE_LATIN.findall(text) + RE_CYR.findall(text)
    if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.8:
        text = text.lower()
        text = re.sub("(^|[.!?¿¡…]\\s*)(\\w)", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


# --------------------------------------------------------------------------- API

def _open_page(image_bytes: bytes) -> Image.Image:
    """Decode a page image as RGB (transparent areas become white)."""
    try:
        full = Image.open(io.BytesIO(image_bytes))
        full.load()
    except Image.DecompressionBombError as e:
        raise TranslationError("Image too large to be a manga page.") from e
    except Exception as e:
        raise TranslationError("Unreadable image.") from e
    if full.mode in ("RGBA", "LA", "P"):
        rgba = full.convert("RGBA")
        bg = Image.new("RGB", full.size, "white")
        bg.paste(rgba, mask=rgba.split()[-1])
        full = bg
    return full.convert("RGB")


class Translator:
    def translate(self, image_bytes: bytes, lang: str = "auto", hint: str | None = None) -> dict:
        """lang: 'auto' or a LANGUAGES code; hint: likely language (e.g. already detected for the chapter)."""
        full = _open_page(image_bytes)
        MODELS.load_base()
        W, H = full.size

        with MODELS.lock:
            texts, bubbles = [], []
            for t, b, own_t, own_b in _slices(H, W):
                crop = full.crop((0, t, W, b))
                for label, score, (x0, y0, x1, y1) in MODELS.detect(crop):
                    box = [max(0, x0), max(0, y0 + t), min(W, x1), min(H, y1 + t)]
                    if label == "bubble":
                        bubbles.append(box)
                        continue
                    cy = (box[1] + box[3]) / 2
                    if own_t <= cy < own_b and _area(box) > 64:
                        texts.append((label, score, box))
            texts = _nms(texts)
            crops = [full.crop(tuple(int(v) for v in box)) for _, _, box in texts]

            auto = lang not in LANGUAGES
            ocr_cache: dict = {}
            if not crops:
                lang = lang if not auto else (hint if hint in LANGUAGES else "ja")
                return {"format": FORMAT, "source_language": lang, "source_language_name": LANGUAGES[lang][0],
                        "width": W, "height": H, "items": []}
            if auto:
                # Every page is checked: a chapter can mix languages (credits, covers, English titles), and a
                # language announced by the site can be wrong. The hint only settles uncertain cases.
                detected, confidence = detect_language(crops, ocr_cache)
                lang = hint if hint in LANGUAGES and hint != detected and confidence < HINT_CONFIDENCE else detected

            # ---- OCR (areas already read while detecting the language are not read again)
            if lang == "ja":
                originals = MODELS.ocr_ja(crops)
            else:
                originals = [MODELS.ocr_easy(LANGUAGES[lang][2], c, ocr_cache)[0] for c in crops]
            if auto and lang in ("es", "pt", "en"):
                lang = _latin_language(originals)
            originals = [_clean_ocr(s, lang) for s in originals]

            # ---- translation (only text that contains letters)
            idx = [i for i, s in enumerate(originals) if RE_LETTER.search(s)]
            src = [_sentence_case(originals[i]) for i in idx]
            if lang == "en":
                out = [originals[i] for i in idx]
            elif lang == "ja":
                out = MODELS.translate_ja(src)
            else:
                out = MODELS.translate_nllb(src, lang)
            translations = list(originals)
            for i, tr in zip(idx, out):
                translations[i] = tr

        def norm(b):
            return [b[0] / W, b[1] / H, b[2] / W, b[3] / H]

        items = []
        for (label, _, box), orig, tr in zip(texts, originals, translations):
            if not orig.strip():
                continue
            ob, bubble = _overlay_box(box, bubbles)
            items.append({
                "x": ob[0] / W, "y": ob[1] / H, "w": (ob[2] - ob[0]) / W, "h": (ob[3] - ob[1]) / H,
                "kind": "speech" if bubble is not None or label == "text_bubble" else "narration",
                "original": orig, "translation": tr,
                # for the typesetter: where the original text is, and the bubble that holds it
                "text_box": norm(box), "bubble": norm(bubble) if bubble is not None else None,
                "vertical": lang in ("ja", "zh", "zh-hant") and (box[3] - box[1]) > 1.3 * (box[2] - box[0]),
            })
        rtl = lang in ("ja", "zh-hant")
        items.sort(key=lambda it: (round(it["y"] * 8), -it["x"] if rtl else it["x"]))
        return {"format": FORMAT, "source_language": lang, "source_language_name": LANGUAGES[lang][0],
                "width": W, "height": H, "items": items}
