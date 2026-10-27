# %% [markdown]
# # 3. Interactive plots (HTML)
# Builds ONE self-contained HTML page (no internet needed) with
# - one plot per column of the `load` output (original data, sentinel values optionally hidden)
# - one extra plot overlaying the ORIGINAL and the CLEANED target channel (`value_column`)
#
# The page handles very long records (e.g. a year of 1-minute data):
# - horizontally scrollable (scroll bar, Shift+wheel, trackpad), the y axes stay in place
# - zoom: Ctrl+wheel (or pinch), drag a box, the +/- and window buttons, double-click = fit all
# - all plots share the same time axis; hovering shows the values of all plots at that time
# - data are drawn as a min/max envelope per screen pixel, so no spike is ever lost when zoomed out
#
# Settings: `plots_config` in env.yaml. Times are UTC.

# %% tags=["parameters"]
upstream = None
product = None
channel = None
sentinel_values = None
config = None

# %%
import base64
import gzip
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

title = config.get("title", "UV measurements")
panel_height = int(config.get("panel_height", 200))
mask_sentinels = bool(config.get("mask_sentinels", True))
initial_days = config.get("initial_days")
selected = config.get("channels")

raw = pd.read_csv(upstream["load"]["data"], parse_dates=["timestamp"])
raw = raw.sort_values("timestamp", kind="stable").reset_index(drop=True)
available = [c for c in raw.columns if c not in ("timestamp", "time_raw", "source_file")]
channels = list(selected) if selected else available
unknown = [c for c in channels + [channel] if c not in available]
if unknown:
    raise KeyError(f"Channel(s) {unknown} not in the load output; available: {available}")

clean = pd.read_csv(upstream["clean"]["data"], parse_dates=["timestamp"]).sort_values("timestamp")
if channel not in clean.columns:
    raise KeyError(f"Cleaned channel {channel!r} not found in the clean output; columns: {list(clean.columns)}")
print(f"original: {len(raw):,} rows | cleaned: {len(clean):,} rows | plotted columns: {channels} | overlay: {channel}")

# %% [markdown]
# ## Prepare the data
# Times become epoch seconds (float64), values float32 (NaN = missing). Every array is gzip-compressed and
# base64-encoded into the page, then decoded by the browser (`DecompressionStream`).

# %%
def epoch_seconds(ts):
    return ((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def pack(a, dtype):
    a = np.ascontiguousarray(a, dtype=dtype)
    return {"type": "f8" if dtype == "<f8" else "f4", "n": int(a.size),
            "data": base64.b64encode(gzip.compress(a.tobytes(), compresslevel=6)).decode("ascii")}


arrays = {"t_raw": pack(epoch_seconds(raw["timestamp"]), "<f8"),
          "t_clean": pack(epoch_seconds(clean["timestamp"]), "<f8")}

masked = {}
for c in dict.fromkeys(channels + [channel]):
    v = raw[c].astype("float64")
    if mask_sentinels and sentinel_values:
        is_sentinel = v.isin(sentinel_values)
        masked[c] = int(is_sentinel.sum())
        v = v.mask(is_sentinel)
    arrays[f"raw:{c}"] = pack(v.to_numpy(), "<f4")
arrays[f"clean:{channel}"] = pack(clean[channel].to_numpy(dtype="float64"), "<f4")

panels = [{"id": c, "title": f"{c}  (original data)",
           "series": [{"name": c, "role": "main", "time": "t_raw", "values": f"raw:{c}"}]} for c in channels]
panels.append({"id": "overlay", "title": f"{channel}: original vs cleaned",
               "series": [{"name": "original", "role": "raw", "time": "t_raw", "values": f"raw:{channel}"},
                          {"name": "cleaned", "role": "clean", "time": "t_clean", "values": f"clean:{channel}"}]})

fmt = "%Y-%m-%d %H:%M"
meta = [f"{len(raw):,} rows  ·  {raw['timestamp'].min():{fmt}} → {raw['timestamp'].max():{fmt}} UTC",
        f"cleaned series: {channel}, {int(clean[channel].notna().sum()):,} valid grid points"]
if any(masked.values()):
    meta.append("sentinel values hidden in the plots: " + ", ".join(f"{c} {n:,}" for c, n in masked.items() if n))

payload = {"title": title, "panelHeight": panel_height, "initialDays": initial_days, "meta": meta,
           "channels": channels, "channel": channel, "arrays": arrays, "panels": panels}

# %% [markdown]
# ## HTML template (CSS + JavaScript, no external libraries)

# %%
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light dark;--bg:#ffffff;--panel:#fbfcfe;--fg:#1d2733;--muted:#66727f;--grid:rgba(100,116,139,.22);
  --border:#d6dde6;--raw:#a3adb8;--sel:rgba(37,99,235,.14);--selb:#2563eb;
  --c0:#2563eb;--c1:#d97706;--c2:#059669;--c3:#c026d3;--c4:#dc2626;--c5:#0891b2;--c6:#7c3aed;--c7:#65a30d}
@media (prefers-color-scheme:dark){:root{--bg:#10151c;--panel:#151b24;--fg:#e5e9ee;--muted:#93a0ae;--grid:rgba(148,163,184,.2);
  --border:#2b3542;--raw:#6b7683;--sel:rgba(96,165,250,.18);--selb:#60a5fa;
  --c0:#60a5fa;--c1:#fbbf24;--c2:#34d399;--c3:#e879f9;--c4:#f87171;--c5:#22d3ee;--c6:#a78bfa;--c7:#a3e635}}
html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
body{display:flex;flex-direction:column}
header{padding:10px 16px 2px}
h1{font-size:18px;margin:0 0 2px}
.meta{color:var(--muted);font-size:12px}
#toolbar{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:6px 16px}
#toolbar .grp{display:inline-flex;gap:4px;align-items:center}
button,select{font:inherit;padding:3px 10px;border:1px solid var(--border);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover,select:hover{border-color:var(--selb)}
label{color:var(--muted);font-size:13px}
#scale{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
#info{padding:2px 16px 6px;min-height:20px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
#scroller{flex:1;min-height:0;overflow:auto;border-top:1px solid var(--border);position:relative;cursor:crosshair;background:var(--panel)}
#scroller.pan{cursor:grab}#scroller.pan.grabbing{cursor:grabbing}
.panel{position:sticky;left:0;border-bottom:1px solid var(--border);overflow:hidden}
.panel canvas{position:absolute;left:0;top:0}
.panel canvas.over{pointer-events:none}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="meta" id="meta"></div></header>
<div id="toolbar">
  <span class="grp"><button id="zout" title="Zoom out">&minus;</button><button id="zin" title="Zoom in">+</button></span>
  <span class="grp">Window:
    <button data-win="3600">1 h</button><button data-win="86400">1 day</button><button data-win="604800">1 week</button>
    <button data-win="2629800">1 month</button><button id="fit">Fit all</button></span>
  <label>Mouse drag <select id="dragmode"><option value="zoom">zooms to a box</option><option value="pan">pans</option></select></label>
  <label>Y axis <select id="ymode"><option value="visible">fits the visible range</option><option value="full">uses the full data range</option></select></label>
  <span id="scale"></span>
</div>
<div id="info">Decoding data&hellip;</div>
<div id="scroller"><div id="track"></div></div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(async function () {
'use strict';
const $ = (id) => document.getElementById(id);
const P = JSON.parse($('payload').textContent);
const info = $('info'), scroller = $('scroller'), track = $('track');
const HELP = 'Hover: values at that time  ·  Ctrl+wheel or pinch: zoom  ·  drag: zoom box (or pan)  ·  scroll bar / Shift+wheel: scroll  ·  double-click: fit all';
$('meta').textContent = P.meta.join('   ·   ');

if (typeof DecompressionStream === 'undefined') {
  info.textContent = 'This browser cannot decompress the data (DecompressionStream missing). Use a current Chrome, Edge, Firefox or Safari.';
  return;
}
async function decode(e) {
  const bin = atob(e.data), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buf = await new Response(stream).arrayBuffer();
  return e.type === 'f8' ? new Float64Array(buf) : new Float32Array(buf);
}
const A = {};
await Promise.all(Object.keys(P.arrays).map(async (k) => { A[k] = await decode(P.arrays[k]); }));

// ---------- helpers ----------
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const pad = (n) => String(n).padStart(2, '0');
function lowerBound(a, x) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >>> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; }
function nearest(t, x) { const n = t.length, i = lowerBound(t, x); if (i <= 0) return 0; if (i >= n) return n - 1; return (x - t[i - 1] < t[i] - x) ? i - 1 : i; }
function medianStep(t) {
  const k = Math.max(1, Math.floor(t.length / 5000)), d = [];
  for (let i = k; i < t.length; i += k) { const x = t[i] - t[i - k]; if (x > 0) d.push(x / k); }
  d.sort((a, b) => a - b); return d.length ? d[d.length >> 1] : 60;
}
function fmtDateTime(t) { const d = new Date(t * 1000); return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`; }
function human(sec) {
  if (sec < 120) return sec.toFixed(sec < 10 ? 1 : 0) + ' s';
  if (sec < 7200) return (sec / 60).toFixed(1) + ' min';
  if (sec < 172800) return (sec / 3600).toFixed(1) + ' h';
  if (sec < 86400 * 90) return (sec / 86400).toFixed(1) + ' days';
  return (sec / 86400 / 30.44).toFixed(1) + ' months';
}
function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const dec = clamp(-Math.floor(Math.log10(step)), 0, 8), ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(dec + 2));
  return { ticks, dec };
}
const SEC_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800];
const MON_STEPS = [1, 2, 3, 6, 12, 24, 60, 120];
function fmtTick(t, step) {
  const d = new Date(t * 1000), hh = pad(d.getUTCHours()), mm = pad(d.getUTCMinutes()), MM = pad(d.getUTCMonth() + 1), DD = pad(d.getUTCDate());
  if (step < 60) return `${hh}:${mm}:${pad(d.getUTCSeconds())}`;
  if (step < 86400) return (hh === '00' && mm === '00') ? `${MM}-${DD}` : `${hh}:${mm}`;
  return `${d.getUTCFullYear()}-${MM}-${DD}`;
}
function timeTicks(tv0, tv1, pps) {
  const minPx = 100, out = [], step = SEC_STEPS.find((s) => s * pps >= minPx);
  if (step !== undefined) {
    for (let t = Math.ceil(tv0 / step) * step; t <= tv1; t += step) out.push({ t, label: fmtTick(t, step) });
    return out;
  }
  const m = MON_STEPS.find((k) => k * 2629800 * pps >= minPx) || 120, d0 = new Date(tv0 * 1000);
  let y = d0.getUTCFullYear(), mo = d0.getUTCMonth();
  for (let g = 0; g < 5000; g++) {
    const t = Date.UTC(y, mo, 1) / 1000;
    if (t > tv1) break;
    const ok = m <= 12 ? (mo % m === 0) : (mo === 0 && y % (m / 12) === 0);
    if (ok && t >= tv0) out.push({ t, label: m < 12 ? `${y}-${pad(mo + 1)}` : String(y) });
    if (++mo > 11) { mo = 0; y++; }
  }
  return out;
}

// ---------- colours (follow the light / dark theme) ----------
let colors = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement), g = (n) => cs.getPropertyValue(n).trim();
  colors = { fg: g('--fg'), muted: g('--muted'), grid: g('--grid'), raw: g('--raw'), sel: g('--sel'), selb: g('--selb'), bg: g('--panel'),
             pal: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => g('--c' + i)) };
}
readColors();

// ---------- data ----------
const stepOf = {};
for (const k of Object.keys(A)) if (k.startsWith('t_')) stepOf[k] = medianStep(A[k]);
const T0 = Math.min(A.t_raw[0], A.t_clean[0]);
const T1 = Math.max(A.t_raw[A.t_raw.length - 1], A.t_clean[A.t_clean.length - 1]);
const span = Math.max(T1 - T0, 1), medStep = stepOf.t_raw;

const LM = 68, RM = 14, TOPM = 28, BOTM = 24, H = P.panelHeight;
let VW = 0, PW = 0, pps = 1, ppsMin = 1, ppsMax = 1, dpr = 1;
let cursorT = null, dragMode = 'zoom', yMode = 'visible', sel = null, drag = false, lastX = 0;

const panels = P.panels.map((pd) => {
  const el = document.createElement('div'); el.className = 'panel'; el.style.height = H + 'px';
  const cd = document.createElement('canvas'), co = document.createElement('canvas'); co.className = 'over';
  el.append(cd, co); track.append(el);
  const series = pd.series.map((s) => {
    const t = A[s.time], v = A[s.values]; let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < v.length; i++) { const x = v[i]; if (x === x) { if (x < mn) mn = x; if (x > mx) mx = x; } }
    const ci = s.role === 'clean' ? P.channels.indexOf(P.channel) : P.channels.indexOf(pd.id);
    return { name: s.name, role: s.role, t, v, step: stepOf[s.time], gap: 5 * stepOf[s.time], mn, mx, ci: Math.max(0, ci), i0: 0, i1: 0 };
  });
  return { id: pd.id, title: pd.title, el, cd, co, series, ctx: null, octx: null };
});

// ---------- drawing ----------
function drawSeries(ctx, s, tv0, Y) {
  const color = s.role === 'raw' ? colors.raw : colors.pal[s.ci % colors.pal.length], n = s.i1 - s.i0;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (n > PW * 2) {                       // many points per pixel: min/max envelope per pixel column
    const mins = new Float32Array(PW).fill(Infinity), maxs = new Float32Array(PW).fill(-Infinity);
    for (let i = s.i0; i < s.i1; i++) {
      const v = s.v[i]; if (v !== v) continue;
      const x = Math.floor((s.t[i] - tv0) * pps); if (x < 0 || x >= PW) continue;
      if (v < mins[x]) mins[x] = v; if (v > maxs[x]) maxs[x] = v;
    }
    ctx.lineWidth = 1; ctx.beginPath();
    for (let x = 0; x < PW; x++) {
      if (mins[x] === Infinity) continue;
      const xx = LM + x + 0.5, ya = Y(mins[x]); let yb = Y(maxs[x]); if (ya - yb < 1) yb = ya - 1;
      ctx.moveTo(xx, ya); ctx.lineTo(xx, yb);
    }
    ctx.stroke();
    return;
  }
  ctx.lineWidth = s.role === 'raw' ? 1 : 1.3; ctx.beginPath();
  let pen = false, prevT = 0;
  for (let i = s.i0; i < s.i1; i++) {
    const v = s.v[i]; if (v !== v) { pen = false; continue; }
    const t = s.t[i], x = LM + (t - tv0) * pps, y = Y(v);
    if (pen && t - prevT <= s.gap) ctx.lineTo(x, y); else { ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y); }
    pen = true; prevT = t;
  }
  ctx.stroke();
  if (pps * s.step > 7) {                 // zoomed in far enough to see the individual samples
    for (let i = s.i0; i < s.i1; i++) {
      const v = s.v[i]; if (v !== v) continue;
      ctx.beginPath(); ctx.arc(LM + (s.t[i] - tv0) * pps, Y(v), 1.9, 0, 6.2832); ctx.fill();
    }
  }
}

function drawPanel(p, tv0, tv1) {
  const ctx = p.ctx, px1 = LM + PW, top = TOPM, bot = H - BOTM, ph = bot - top;
  ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, VW, H);
  let lo = Infinity, hi = -Infinity;
  for (const s of p.series) {
    s.i0 = Math.max(0, lowerBound(s.t, tv0) - 1); s.i1 = Math.min(s.t.length, lowerBound(s.t, tv1) + 1);
    if (yMode === 'visible') { for (let i = s.i0; i < s.i1; i++) { const x = s.v[i]; if (x === x) { if (x < lo) lo = x; if (x > hi) hi = x; } } }
    else { if (s.mn < lo) lo = s.mn; if (s.mx > hi) hi = s.mx; }
  }
  if (!(lo <= hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
  const padv = (hi - lo) * 0.04; lo -= padv; hi += padv;
  const Y = (v) => bot - (v - lo) / (hi - lo) * ph;

  ctx.font = '11px system-ui, sans-serif'; ctx.lineWidth = 1; ctx.strokeStyle = colors.grid; ctx.fillStyle = colors.muted;
  const yt = niceTicks(lo, hi, Math.max(2, Math.floor(ph / 40)));
  ctx.beginPath();
  for (const v of yt.ticks) { const y = Math.round(Y(v)) + 0.5; ctx.moveTo(LM, y); ctx.lineTo(px1, y); }
  ctx.stroke();
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt.ticks) ctx.fillText(v.toFixed(yt.dec), LM - 7, Y(v));
  const xt = timeTicks(tv0, tv1, pps);
  ctx.beginPath();
  for (const k of xt) { const x = Math.round(LM + (k.t - tv0) * pps) + 0.5; ctx.moveTo(x, top); ctx.lineTo(x, bot); }
  ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const k of xt) ctx.fillText(k.label, LM + (k.t - tv0) * pps, bot + 6);
  ctx.strokeRect(LM + 0.5, top + 0.5, PW, ph);

  ctx.save(); ctx.beginPath(); ctx.rect(LM, top, PW, ph); ctx.clip();
  for (const s of p.series) drawSeries(ctx, s, tv0, Y);
  ctx.restore();

  ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '600 12px system-ui, sans-serif'; ctx.fillStyle = colors.fg;
  ctx.fillText(p.title, LM + 4, 7);
  if (p.series.length > 1) {
    let x = LM + 4 + ctx.measureText(p.title).width + 22; ctx.font = '12px system-ui, sans-serif';
    for (const s of p.series) {
      ctx.strokeStyle = s.role === 'raw' ? colors.raw : colors.pal[s.ci % colors.pal.length]; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x, 14); ctx.lineTo(x + 18, 14); ctx.stroke();
      ctx.fillStyle = colors.fg; ctx.fillText(s.name, x + 24, 7); x += 24 + ctx.measureText(s.name).width + 18;
    }
  }
}

function drawOverlay() {
  const tv0 = T0 + scroller.scrollLeft / pps;
  for (const p of panels) {
    const o = p.octx; o.clearRect(0, 0, VW, H);
    if (sel) {
      const a = Math.min(sel.x0, sel.x1), b = Math.max(sel.x0, sel.x1);
      o.fillStyle = colors.sel; o.fillRect(a, TOPM, b - a, H - TOPM - BOTM);
      o.strokeStyle = colors.selb; o.lineWidth = 1; o.strokeRect(a + 0.5, TOPM + 0.5, b - a, H - TOPM - BOTM - 1);
    }
    if (cursorT !== null) {
      const x = LM + (cursorT - tv0) * pps;
      if (x >= LM && x <= LM + PW) {
        o.strokeStyle = colors.muted; o.lineWidth = 1; o.setLineDash([4, 3]); o.beginPath();
        o.moveTo(Math.round(x) + 0.5, TOPM); o.lineTo(Math.round(x) + 0.5, H - BOTM); o.stroke(); o.setLineDash([]);
      }
    }
  }
}

function draw() {
  const tv0 = T0 + scroller.scrollLeft / pps, tv1 = tv0 + PW / pps;
  for (const p of panels) drawPanel(p, tv0, tv1);
  drawOverlay();
  $('scale').textContent = `visible: ${human(PW / pps)}  ·  1 px = ${human(1 / pps)}`;
}
let raf = 0;
function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }

// ---------- scale / layout ----------
function applyTrack() { track.style.width = Math.max(VW, span * pps + LM + RM) + 'px'; }
function setScale(newPps, ax, ta) {          // keep time `ta` at x offset `ax` inside the plot area
  pps = clamp(newPps, ppsMin, ppsMax);
  applyTrack();
  scroller.scrollLeft = (ta - ax / pps - T0) * pps;
  schedule();
}
function centerTime() { return T0 + scroller.scrollLeft / pps + PW / (2 * pps); }
function zoomBy(f) { setScale(pps * f, PW / 2, centerTime()); }
function showSeconds(sec) { setScale(PW / sec, PW / 2, centerTime()); }

function resize(first) {
  const oldT0 = first ? T0 : T0 + scroller.scrollLeft / pps;
  VW = scroller.clientWidth; PW = Math.max(60, VW - LM - RM); dpr = window.devicePixelRatio || 1;
  ppsMin = PW / span; ppsMax = Math.max(ppsMin, Math.min(30 / medStep, 8e6 / span));
  if (first) pps = P.initialDays ? PW / (P.initialDays * 86400) : ppsMin;
  pps = clamp(pps, ppsMin, ppsMax);
  for (const p of panels) {
    p.el.style.width = VW + 'px';
    for (const c of [p.cd, p.co]) { c.width = Math.round(VW * dpr); c.height = Math.round(H * dpr); c.style.width = VW + 'px'; c.style.height = H + 'px'; }
    p.ctx = p.cd.getContext('2d'); p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    p.octx = p.co.getContext('2d'); p.octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  applyTrack(); scroller.scrollLeft = (oldT0 - T0) * pps; draw();
}

// ---------- interaction ----------
function pointer(e) {
  const r = scroller.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  return { x, y, ok: x >= LM && x <= LM + PW && x < scroller.clientWidth && y < scroller.clientHeight };
}
function updateInfo() {
  if (cursorT === null) { info.textContent = HELP; return; }
  const parts = [fmtDateTime(cursorT) + ' UTC'];
  for (const p of panels) for (const s of p.series) {
    const label = p.id === 'overlay' ? `${P.channel} ${s.name}` : p.id, i = nearest(s.t, cursorT);
    let txt = '–';
    if (Math.abs(s.t[i] - cursorT) <= 1.5 * s.step) { const v = s.v[i]; if (v === v) txt = String(Number(v.toPrecision(6))); }
    parts.push(`${label} ${txt}`);
  }
  info.textContent = parts.join('   ·   ');
}
scroller.addEventListener('scroll', schedule);
scroller.addEventListener('mousemove', (e) => {
  const m = pointer(e);
  if (drag) {
    if (dragMode === 'pan') { scroller.scrollLeft -= e.clientX - lastX; lastX = e.clientX; }
    else if (sel) sel.x1 = clamp(m.x, LM, LM + PW);
  }
  cursorT = m.ok ? T0 + scroller.scrollLeft / pps + (m.x - LM) / pps : null;
  updateInfo(); drawOverlay();
});
scroller.addEventListener('mouseleave', () => { if (!drag) { cursorT = null; updateInfo(); drawOverlay(); } });
scroller.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  const m = pointer(e); if (!m.ok) return;     // clicks on a scroll bar are left to the browser
  e.preventDefault(); drag = true; lastX = e.clientX;
  if (dragMode === 'zoom') sel = { x0: m.x, x1: m.x }; else scroller.classList.add('grabbing');
});
window.addEventListener('mouseup', () => {
  if (!drag) return; drag = false; scroller.classList.remove('grabbing');
  if (dragMode === 'zoom' && sel) {
    const a = Math.min(sel.x0, sel.x1), b = Math.max(sel.x0, sel.x1); sel = null;
    if (b - a >= 8) { const tv0 = T0 + scroller.scrollLeft / pps; setScale(PW / ((b - a) / pps), 0, tv0 + (a - LM) / pps); }
    else drawOverlay();
  }
});
scroller.addEventListener('wheel', (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;        // plain wheel keeps scrolling vertically
  e.preventDefault();
  const ax = clamp(pointer(e).x - LM, 0, PW), ta = T0 + scroller.scrollLeft / pps + ax / pps;
  setScale(pps * Math.exp(-e.deltaY * 0.002), ax, ta);
}, { passive: false });
scroller.addEventListener('dblclick', () => setScale(ppsMin, 0, T0));
$('zin').onclick = () => zoomBy(2); $('zout').onclick = () => zoomBy(0.5); $('fit').onclick = () => setScale(ppsMin, 0, T0);
document.querySelectorAll('[data-win]').forEach((b) => { b.onclick = () => showSeconds(+b.dataset.win); });
$('dragmode').onchange = (e) => { dragMode = e.target.value; scroller.classList.toggle('pan', dragMode === 'pan'); };
$('ymode').onchange = (e) => { yMode = e.target.value; schedule(); };
new ResizeObserver(() => resize(false)).observe(scroller);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); schedule(); });

resize(true); updateInfo();
document.body.dataset.ready = '1';
})();
</script>
</body>
</html>
"""

# %% [markdown]
# ## Write the page

# %%
payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
page = TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload_json)

out_path = Path(product["html"])
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(page, encoding="utf-8")
pd.DataFrame({"file": [str(out_path)], "size_MB": [round(out_path.stat().st_size / 1e6, 2)],
              "panels": [len(panels)], "points_original": [len(raw)], "points_cleaned": [len(clean)]})
