"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const state = {
  chapter: null,       // {id, title, url, page_count, next_url, prev_url}
  page: 0,
  data: null,          // translation of the current page
  token: 0,            // discards stale responses when pages change quickly
};

const els = {
  welcome: $("#welcome"), reader: $("#reader"), title: $("#chapter-title"),
  img: $("#page-img"), overlay: $("#overlay"), spinner: $("#page-spinner"),
  select: $("#page-select"), label: $("#page-label"), status: $("#status"),
  urlInput: $("#url-input"),
};

/* ------------------------------------------------------------------ helpers */

function store(key, val) { try { localStorage.setItem(key, JSON.stringify(val)); } catch {} }
function load(key, dflt) { try { return JSON.parse(localStorage.getItem(key)) ?? dflt; } catch { return dflt; } }

function showStatus(msg, kind = "error") {
  els.status.textContent = msg;
  els.status.className = `status ${kind === "info" ? "info" : ""}`;
}
function hideStatus() { els.status.className = "status hidden"; }

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch {}
  if (!r.ok) throw new Error(body?.error || `Error ${r.status}`);
  return body;
}

const pageBase = (n) => `/api/chapter/${state.chapter.id}/page/${n}`;
const langQuery = () => `lang=${encodeURIComponent(state.lang)}`;
const imageUrl = (n) => `${pageBase(n)}/image?rev=${state.chapter.rev || ""}`;
const hideSfx = () => $("#hide-sfx").checked;
state.stamps = {};  // page -> time of the last retranslation, so the browser fetches the new rendering
const renderedUrl = (n) =>
  `${pageBase(n)}/rendered?${langQuery()}&sfx=${hideSfx() ? 0 : 1}&rev=${state.chapter.rev || ""}&t=${state.stamps[n] || 0}`;

/* Show the original page, or the page with the translation lettered in */
function showPage(token) {
  const n = state.page;
  if (document.body.classList.contains("show-original") || !state.data) {
    els.img.src = imageUrl(n);
    return;
  }
  const url = renderedUrl(n);
  const pre = new Image();
  pre.onload = () => { if (token === state.token) { els.img.src = url; els.spinner.classList.add("hidden"); } };
  pre.onerror = () => {
    if (token !== state.token) return;
    els.spinner.classList.add("hidden");
    showStatus(`Page ${n + 1}: could not letter the translation.`);
  };
  pre.src = url;
}
state.lang = load("lang", "auto");

/* While a page is being translated, show the current step (e.g. first download of a model) */
function watchSpinner(token) {
  const span = $("#page-spinner span");
  span.textContent = "Translating…";
  const tick = async () => {
    if (token !== state.token || els.spinner.classList.contains("hidden")) return;
    try {
      const s = await api("/api/status");
      span.textContent = s.stage && s.stage.startsWith("loading") ? `Preparing: ${s.stage}…` : "Translating…";
    } catch {}
    setTimeout(tick, 1500);
  };
  setTimeout(tick, 1500);
}

function showDetected(data) {
  const el = $("#lang-detected");
  el.textContent = data && data.source_language_name && state.lang === "auto" ? `→ ${data.source_language_name}` : "";
}

/* ------------------------------------------------------------------ chapters */

async function openChapter(url, page = null) {
  els.urlInput.value = url;
  showStatus("Fetching the chapter pages…", "info");
  els.welcome.classList.add("hidden");
  els.reader.classList.remove("hidden");
  try {
    state.chapter = await api("/api/chapter", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }),
    });
  } catch (e) {
    showStatus(e.message);
    if (!state.chapter) { els.reader.classList.add("hidden"); els.welcome.classList.remove("hidden"); }
    return;
  }
  hideStatus();
  const ch = state.chapter;
  buildPageList();
  const recent = load("recent", []).filter((r) => r.url !== ch.url);
  recent.unshift({ url: ch.url, title: ch.title, page: 0 });
  store("recent", recent.slice(0, 8));
  if (page === null) page = ch.start_page || 0;   // e.g. a link to page 12 of a gallery opens at page 12
  if (page < 0) page = ch.page_count - 1;
  goTo(Math.min(page, ch.page_count - 1));
}

/* Page count, "+" when more pages are discovered while reading (one page per URL readers) */
const countLabel = () => `${state.chapter.page_count}${state.chapter.more ? "+" : ""}`;

function buildPageList() {
  const ch = state.chapter;
  const pages = `${countLabel()} page${ch.page_count > 1 || ch.more ? "s" : ""}`;
  els.title.textContent = `${ch.title} — ${pages}`;
  els.select.innerHTML = "";
  for (let i = 0; i < ch.page_count; i++) els.select.add(new Option(`Page ${i + 1} / ${countLabel()}`, i));
  els.select.value = state.page;
  els.label.textContent = `${state.page + 1} / ${countLabel()}`;
}

async function refreshChapter() {
  if (!state.chapter?.more) return;
  try {
    const ch = await api(`/api/chapter/${state.chapter.id}`);
    if (ch.page_count !== state.chapter.page_count || ch.more !== state.chapter.more) {
      Object.assign(state.chapter, { page_count: ch.page_count, more: ch.more });
      buildPageList();
      updateNavButtons();
    }
  } catch {}
}

/* ------------------------------------------------------------------ pages */

function goTo(n) {
  const ch = state.chapter;
  if (!ch || n < 0 || n >= ch.page_count) return;
  state.page = n;
  state.data = null;
  const token = ++state.token;

  els.select.value = n;
  els.label.textContent = `${n + 1} / ${countLabel()}`;
  history.replaceState(null, "", `#url=${encodeURIComponent(ch.url)}&p=${n + 1}`);
  const recent = load("recent", []);
  const entry = recent.find((r) => r.url === ch.url);
  if (entry) { entry.page = n; store("recent", recent); }
  updateNavButtons();
  hideStatus();

  els.overlay.innerHTML = "";
  els.spinner.classList.remove("hidden");
  els.img.src = imageUrl(n);
  window.scrollTo({ top: 0 });

  // Preload the next image
  if (n + 1 < ch.page_count) new Image().src = imageUrl(n + 1);

  watchSpinner(token);
  api(`${pageBase(n)}/translation?${langQuery()}`)
    .then((data) => {
      if (token !== state.token) return;
      state.data = data;
      showDetected(data);
      renderOverlay();
      if (document.body.classList.contains("show-original")) els.spinner.classList.add("hidden");
      else showPage(token);  // the spinner stays until the lettered page is loaded
      setTimeout(refreshChapter, 1500);  // background pre-translation may have discovered new pages
    })
    .catch((e) => {
      if (token !== state.token) return;
      showStatus(`Page ${n + 1}: ${e.message}`);
      els.spinner.classList.add("hidden");
    });
}

function updateNavButtons() {
  const ch = state.chapter, n = state.page;
  const atEnd = n >= ch.page_count - 1 && !ch.more, atStart = n <= 0;
  $$(".nav-next").forEach((b) => {
    b.textContent = atEnd ? (ch.next_url ? "Next chapter ▶▶" : "End") : "Next ▶";
    b.disabled = atEnd && !ch.next_url;
  });
  $$(".nav-prev").forEach((b) => {
    b.textContent = atStart ? (ch.prev_url ? "◀◀ Previous chapter" : "Start") : "◀ Previous";
    b.disabled = atStart && !ch.prev_url;
  });
}

async function next() {
  const ch = state.chapter;
  if (!ch) return;
  if (state.page < ch.page_count - 1) return goTo(state.page + 1);
  if (ch.more) {
    // Discover the following page on the site
    showStatus("Looking for the next page…", "info");
    try {
      const upd = await api(`/api/chapter/${ch.id}/extend?n=${state.page + 1}`, { method: "POST" });
      Object.assign(ch, { page_count: upd.page_count, more: upd.more });
      hideStatus();
      buildPageList();
      if (state.page < ch.page_count - 1) return goTo(state.page + 1);
      updateNavButtons();  // that was the last page
    } catch (e) {
      return showStatus(e.message);
    }
  }
  if (ch.next_url) openChapter(ch.next_url, 0);
}

function prev() {
  const ch = state.chapter;
  if (!ch) return;
  if (state.page > 0) goTo(state.page - 1);
  else if (ch.prev_url) openChapter(ch.prev_url, -1);
}

/* ------------------------------------------------------------------ overlay */

/* The translation is lettered into the image by the server; these invisible zones only show the
   original text when hovering over a bubble. */
function renderOverlay() {
  els.overlay.innerHTML = "";
  const data = state.data;
  if (!data) return;
  for (const it of data.items) {
    if (it.original.trim() === it.translation.trim()) continue; // already in English
    const [x0, y0, x1, y1] = it.bubble || it.text_box || [it.x, it.y, it.x + it.w, it.y + it.h];
    const d = document.createElement("div");
    d.className = "zone";
    d.style.left = `${x0 * 100}%`;
    d.style.top = `${y0 * 100}%`;
    d.style.width = `${(x1 - x0) * 100}%`;
    d.style.height = `${(y1 - y0) * 100}%`;
    d.dataset.original = it.original;
    els.overlay.appendChild(d);
  }
  if (!data.items.length) showStatus("No text detected on this page.", "info");
}

/* ------------------------------------------------------------------ local model status */

async function pollStatus() {
  const chip = $("#model-status");
  let s;
  try { s = await api("/api/status"); } catch { setTimeout(pollStatus, 3000); return; }
  if (s.ready) {
    chip.textContent = `● Models ${s.stage}`;
    chip.className = "model-status ready";
    return;
  }
  if (s.error) {
    chip.textContent = "● Model error";
    chip.title = s.error;
    chip.className = "model-status error";
    return;
  }
  chip.textContent = `◌ Models: ${s.stage}…`;
  chip.className = "model-status";
  setTimeout(pollStatus, 1500);
}

/* ------------------------------------------------------------------ events */

$("#url-form").addEventListener("submit", (e) => {
  e.preventDefault();
  openChapter(els.urlInput.value.trim());
});

$$(".nav-next").forEach((b) => b.addEventListener("click", next));
$$(".nav-prev").forEach((b) => b.addEventListener("click", prev));
els.select.addEventListener("change", () => goTo(Number(els.select.value)));

$("#show-original").addEventListener("change", (e) => {
  document.body.classList.toggle("show-original", e.target.checked);
  if (state.chapter) showPage(state.token);
});
$("#hide-sfx").addEventListener("change", (e) => {
  store("hideSfx", e.target.checked);
  if (state.chapter && state.data) showPage(state.token);
});

$("#retranslate-btn").addEventListener("click", () => {
  if (!state.chapter) return;
  const n = state.page, token = ++state.token;
  els.overlay.innerHTML = "";
  els.spinner.classList.remove("hidden");
  hideStatus();
  watchSpinner(token);
  api(`${pageBase(n)}/retranslate?${langQuery()}`, { method: "POST" })
    .then((data) => {
      if (token !== state.token) return;
      state.data = data;
      state.stamps[n] = Date.now();  // new rendering: new URL
      showDetected(data);
      renderOverlay();
      showPage(token);
    })
    .catch((e) => {
      if (token !== state.token) return;
      showStatus(e.message);
      els.spinner.classList.add("hidden");
    });
});

$("#download-btn").addEventListener("click", () => {
  if (!state.chapter || !state.data) return showStatus("Wait until the page is translated.", "info");
  const sfx = $("#hide-sfx").checked ? "0" : "1";
  window.location.href = `${pageBase(state.page)}/download?sfx=${sfx}&${langQuery()}`;
});

const langSelect = $("#lang-select");
langSelect.addEventListener("change", () => {
  state.lang = langSelect.value;
  store("lang", state.lang);
  if (state.chapter) goTo(state.page);
});

async function loadLanguages() {
  try {
    const langs = await api("/api/languages");
    langSelect.innerHTML = "";
    for (const l of langs) langSelect.add(new Option(l.name, l.code));
    langSelect.value = langs.some((l) => l.code === state.lang) ? state.lang : "auto";
    state.lang = langSelect.value;
  } catch {}
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, select, textarea, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "arrowright" || k === "d") { e.preventDefault(); next(); }
  else if (k === "arrowleft" || k === "a") { e.preventDefault(); prev(); }
  else if (k === "o") $("#show-original").click();
  else if (k === "s") $("#hide-sfx").click();
});

els.img.addEventListener("error", () => showStatus("Could not load the image for this page."));

/* ------------------------------------------------------------------ startup */

function renderRecent() {
  const recent = load("recent", []);
  const box = $("#recent");
  if (!recent.length) { box.innerHTML = ""; return; }
  box.innerHTML = "<h3>Recently read</h3>";
  for (const r of recent) {
    const a = document.createElement("a");
    a.href = `#url=${encodeURIComponent(r.url)}&p=${(r.page || 0) + 1}`;
    a.textContent = r.title || r.url;
    const small = document.createElement("small");
    small.textContent = `  · page ${(r.page || 0) + 1}`;
    a.appendChild(small);
    a.addEventListener("click", (e) => { e.preventDefault(); openChapter(r.url, r.page || 0); });
    box.appendChild(a);
  }
}

function openFromHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const url = params.get("url");
  if (!url) return;
  const page = Math.max(0, Number(params.get("p") || 1) - 1);
  if (state.chapter && state.chapter.url === url) goTo(page);
  else openChapter(url, page);
}
window.addEventListener("hashchange", openFromHash);

(async function init() {
  if (load("hideSfx", false)) $("#hide-sfx").click();
  renderRecent();
  await loadLanguages();
  openFromHash();
  pollStatus();
})();
