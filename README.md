# Manga Translator (fully local)

[![tests](https://github.com/Indianarex/local-manga-translator/actions/workflows/tests.yml/badge.svg)](https://github.com/Indianarex/local-manga-translator/actions/workflows/tests.yml)

A reader that automatically translates manga, manhwa and manhua into English, **entirely on your computer**: no API key and no online translation service. Paste the URL of a chapter. The app downloads the pages, detects the speech bubbles, reads the text, translates it into English and letters the translation into the page in place of the original text. You read page by page, and the next pages are translated ahead of time.

**Source languages:** Japanese, Chinese (simplified and traditional), Korean, Spanish, Portuguese, Russian. The language is detected automatically, or can be chosen in the toolbar.

> Status: work in progress. All six languages have been tested on real pages. Translation quality depends mostly on how cleanly the OCR reads the lettering: Japanese and Spanish read very well, while stylized or handwritten fonts produce rougher results.

## Requirements

| | |
|---|---|
| Python | 3.12 or newer (tested with 3.12, 3.13 and 3.14) |
| GPU | Recommended: an NVIDIA GPU with 4 GB of memory or more (a few seconds per page). Apple Silicon Macs use their GPU too (not tested yet). Without a GPU everything runs on the CPU, only slower. |
| Disk | PyTorch: about 3 GB with CUDA. Models, downloaded on first use: 1.3 GB for Japanese, plus 5.5 GB for the other languages. |
| Memory used on the GPU | About 2 GB for Japanese alone, 4.5 GB with every language loaded (measured on an RTX 5060) |

## Getting started

**Windows:** double-click **`run.bat`**. **Linux / macOS:** run `./run.sh`.

1. The first run creates a Python environment in `.venv` and installs PyTorch in the build that matches the machine (NVIDIA CUDA, Apple Silicon or CPU), then the other dependencies. This happens once.
2. Your browser opens at <http://127.0.0.1:5000>.
3. Paste the URL of a chapter and click **Translate**.

The first time a language is used, the models it needs are downloaded once into the `models/` folder. The badge in the top-right corner shows their status.

<details>
<summary>Manual installation</summary>

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
# Pick the PyTorch build for your machine: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python app.py
```

Set `PORT` to use another port and `NO_BROWSER=1` to not open the browser.
</details>

## Reading

| Action | Button | Key |
|---|---|---|
| Next page | Next ▶ | `→` or `D` |
| Previous page | ◀ Previous | `←` or `A` |
| Go to a page | page list | |
| Choose the source language | "Auto-detect" menu | |
| Show the original | "Original" checkbox | `O` |
| Hide the sound-effect translations | "Hide SFX" checkbox | `S` |
| Translate the page again | ↻ Retranslate | |
| Save the translated page | ⬇ PNG | |

- On the last page, the button becomes **Next chapter** and loads the following chapter.
- Hovering over a bubble shows the original text.
- Pages, translations and lettered pages are cached in `cache/`, so rereading a translated page is instant. The cache is kept under 2 GB by deleting the chapters read longest ago (change the limit with the `MANGA_CACHE_MB` environment variable).
- The address bar keeps the chapter and page (`#url=…&p=3`), and the home screen lists recently read chapters.

## Supported sites

- **MangaDex** (`mangadex.org/chapter/…`): native support through its API, including next/previous chapter and the chapter's language.
- **Most readers that show a whole chapter on one page**: images are extracted from the HTML, including lazy-loading attributes (`data-src`, `srcset`…) and URLs found in scripts.
- **Readers with one page per URL**, on any site. The page image is told apart from the interface icons, and the list of pages is built from:
  1. the reader's page menu, when it has one;
  2. numbered addresses (`…/page/3`, `?p=3`, `…/003.html`): every page from the first to the last linked one;
  3. otherwise the "previous" links are followed back to page 1, and the following pages are discovered while reading through the "next" links. The page counter then shows `12+`.

  A link to the middle of a chapter opens at that page.
- **A direct image URL** also works.

Some sites only load their images with JavaScript. For those, install Playwright (optional):

```bash
pip install playwright
python -m playwright install chromium
```

## How it works

1. **Detection** of speech bubbles and text areas; long webtoon strips are cut into overlapping slices.
2. **Language detection** (in auto mode): every script is tried on a few bubbles, and the most consistent one wins.
3. **OCR** of the text in each bubble.
4. **Translation** into English.
5. **Typesetting**: the original text is erased and the translation is lettered into the page, adapted to each case:
   - **Erasing**: text in a plain bubble or box is repainted with the bubble's own color; text on artwork is removed by inpainting (the picture behind it is reconstructed). Bubble outlines are kept.
   - **Color**: the English text takes the color of the original (dark on light, light on dark), with a contrasting outline when it sits on artwork.
   - **Size**: estimated from the size of the original lettering, then reduced only as much as needed to fit.
   - **Shape**: lines follow the oval of a speech bubble (shorter at the top and bottom) or fill a rectangular box.
   - **Font**: dialogue, shouting (`!!`, `!?`), thoughts (text in parentheses), narration outside bubbles, handwritten notes on artwork, and sound effects each get their own font.
   - **Sound effects** drawn into the art are left intact and get a small translated label next to them.
   - Scanlation credits and watermarks (links, `@handles`) are left untouched.

| Step | Model | License |
|---|---|---|
| Bubble detection | [`ogkalu/comic-text-and-bubble-detector`](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) | Apache 2.0 |
| Japanese OCR | [`kha-white/manga-ocr-base`](https://huggingface.co/kha-white/manga-ocr-base) | Apache 2.0 |
| Chinese, Korean, Latin and Cyrillic OCR | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Apache 2.0 |
| Japanese → English translation | [`LiquidAI/LFM2-350M-ENJP-MT`](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT) | LFM Open License v1.0 |
| Other languages → English translation | [`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | CC BY-NC 4.0 (non-commercial use) |

The models are not part of this repository; they are downloaded from Hugging Face on first use and keep their own licenses. Each model is pinned to an exact revision (in `translator.py`), so an update on Hugging Face cannot change the app's behavior, and only `safetensors` weights are downloaded.

| Lettering | Font | License |
|---|---|---|
| Dialogue, thoughts | [Comic Neue](https://fonts.google.com/specimen/Comic+Neue) Bold / Bold Italic | SIL Open Font License 1.1 |
| Shouting, sound effects | [Bangers](https://fonts.google.com/specimen/Bangers) | SIL Open Font License 1.1 |
| Handwritten notes | [Patrick Hand](https://fonts.google.com/specimen/Patrick+Hand) | SIL Open Font License 1.1 |
| Narration | [Roboto Condensed](https://fonts.google.com/specimen/Roboto+Condensed) | SIL Open Font License 1.1 |

The fonts are included in `fonts/`, each with its license file.

## Files

| File | Role |
|---|---|
| `app.py` | Local Flask server: image proxy, cache, background translation and lettering of the next 2 pages |
| `scraper.py` | Extracts the chapter's page images (and next/previous chapter) from the URL |
| `translator.py` | Detection → language → OCR → translation, all local |
| `render.py` | Typesetting: erases the original text and letters the translation into the page |
| `fonts/` | Lettering fonts (SIL Open Font License) |
| `static/` | Reader interface |
| `run.bat`, `run.sh` | Launchers for Windows and Linux/macOS (environment, dependencies, server) |
| `tests/` | Test suite, with a fake local manga site |

To free disk space, delete the `models/` (models) and `cache/` (downloaded pages and translations) folders.

## Troubleshooting

- **"No manga images found"**: the site probably loads its images with JavaScript; install Playwright (see above). Sites behind a login or a captcha are not supported.
- **Slow translation**: check the badge in the top-right corner. "ready (CPU)" means no GPU is used; on an NVIDIA machine, delete `.venv` and run the launcher again so it installs the CUDA build of PyTorch.
- **SSL / certificate errors**: antivirus software or corporate proxies that inspect HTTPS are handled through the system certificate store (`truststore`); make sure it is installed.
- **A page looks wrong**: click **↻ Retranslate**, or pick the source language by hand instead of "Auto-detect".

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests need neither PyTorch nor the models: they run against a fake manga site served locally (`tests/conftest.py`). GitHub Actions runs them on every push, on Linux and Windows.

## License

The code in this repository is released under the [PolyForm Noncommercial License 1.0.0](LICENSE).

In short (the [license text](LICENSE) is what applies):

- You **may** use, copy, modify and share this software, and build on it, for any **noncommercial** purpose: personal use, hobby projects, study, research, and use by schools, charities and public institutions.
- You **may not** use it commercially: no selling it or a modified version, no paid service built on it, no use as part of a business, without a separate license from the author. For commercial licensing, contact [Indianarex](https://github.com/Indianarex).
- When you share it, keep the `LICENSE` file, including its `Required Notice` line.

The machine-learning models are not covered by this license: they are downloaded from their authors on first use and keep their own licenses (see the table above). In particular, the NLLB translation model is licensed for non-commercial use only. The fonts in `fonts/` are not covered by it either: they stay under the SIL Open Font License.

Respect the rights of the manga authors and the terms of the sites you read from: translated pages are for personal use.
