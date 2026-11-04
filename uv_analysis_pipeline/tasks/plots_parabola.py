# %% [markdown]
# # 7b. Parabola plots (HTML)
# Three self-contained pages made from the output of `parabola` (and the original data from `load`):
# 1. **grid page**: the fitted parabola of every day as a small plot, in groups by fit quality;
# 2. **overlay page**: the same parabolas drawn over the ORIGINAL signal of that day (grey), the fitted interval
#    is shaded; hover a plot to read time and value;
# 3. **timeline page**: the ENTIRE signal of a channel in ONE horizontally scrollable plot (like `plots`), with the
#    parabolas of all days drawn on top of it; zoom, scroll, hover a parabola to see its class and statistics.
#
# On all pages the quality groups (perfect, lightly_cloudy, very_cloudy, ...) can be hidden/shown with the check boxes
# at the top; every plot carries its class and statistics (maximum position and value, R^2, share of cloud-free points,
# fitted points and interval, number and duration of cloud dips). Time axis: hours of the day, UTC.
# Settings: `plots_parabola_config` in env.yaml.

# %% tags=["parameters"]
upstream = None
product = None
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

title = config.get("title", "Solar UV: daily parabola fits")
selected = config.get("channels")
columns0 = int(config.get("columns", 4))
grid_same_y = bool(config.get("grid_same_y_scale", True))
overlay_same_y = bool(config.get("overlay_same_y_scale", False))
timeline_height = int(config.get("timeline_panel_height", 260))
timeline_initial_days = config.get("timeline_initial_days")
if not 1 <= columns0 <= 10:
    raise ValueError(f"columns must be between 1 and 10 (got {columns0})")

par = pd.read_csv(upstream["parabola"]["data"], index_col="date", parse_dates=True)
available = [c[: -len("_quality")] for c in par.columns if c.endswith("_quality")]
channels = list(selected) if selected else available
unknown = [c for c in channels if c not in available]
if unknown:
    raise KeyError(f"Channel(s) {unknown} not in the parabola output; available: {available}")
raw = pd.read_csv(upstream["load"]["data"], parse_dates=["timestamp"])
print(f"{len(par)} days, channels: {channels}")

# %% [markdown]
# ## Quality classes
# Same order and wording as the summary of `parabola`. Days without any data (`no_data`) have nothing to plot.

# %%
CLASSES = [
    ("perfect", "perfect", "#2a9d5b"),
    ("lightly_cloudy", "lightly cloudy", "#e0a100"),
    ("very_cloudy", "very cloudy", "#d9480f"),
    ("not_concave", "not concave", "#7b5ea7"),
    ("peak_outside_window", "peak outside window", "#2f6fbd"),
    ("gap_in_window", "gap in window", "#1b998b"),
    ("too_few_points", "too few points", "#6b7a8f"),
    ("low_activity", "low activity", "#8d6e63"),
]
known = {k for k, _, _ in CLASSES}
FIELDS = {"a": "a", "b": "b", "c": "c", "window_start": "ws", "window_end": "we", "t_max": "tm", "y_max": "ym", "r2": "r2",
          "clear_fraction": "cf", "n_points": "n", "n_dips": "nd", "dip_hours": "dh", "longest_dip_hours": "ld",
          "max_dip_depth_pct": "dd", "peak_obs": "pk", "t_peak_obs": "tpk"}
DIGITS = {"a": 10, "b": 10, "c": 10}


def num(v, nd=4):
    return None if pd.isna(v) else round(float(v), nd)


# %% [markdown]
# ## Build the data of every channel

# %%
def channel_payload(c, with_raw):
    d = par[[col for col in par.columns if col.startswith(f"{c}_")]].rename(columns=lambda x: x[len(c) + 1:])
    d = d[d["quality"].notna() & (d["quality"] != "no_data")]
    extra = sorted(set(d["quality"]) - known)               # unforeseen labels still get a group
    days = []
    for i, (date, r) in enumerate(d.iterrows()):
        row = {"i": i, "date": f"{date:%Y-%m-%d}", "q": r["quality"]}
        row.update({short: num(r[f], DIGITS.get(f, 4)) for f, short in FIELDS.items()})
        days.append(row)

    v = raw[["timestamp", c]].copy()
    v[c] = v[c].astype("float64")
    if sentinel_values:
        v[c] = v[c].mask(v[c].isin(sentinel_values))
    v = v.dropna().groupby("timestamp")[c].mean().sort_index()
    p1 = float(np.percentile(v, 1)) if len(v) else 0.0

    tops = [x["ym"] if x["ym"] is not None else x["pk"] for x in days]
    tops = [t for t in tops if t is not None]
    ymax = (max(tops) if tops else 1.0) * 1.15
    ymin = p1 if p1 > 0.3 * ymax else min(0.0, p1)           # 0 for UV; the baseline for channels that never go to 0
    ws = [x["ws"] for x in days if x["ws"] is not None]
    we = [x["we"] for x in days if x["we"] is not None]
    xmin = max(0.0, np.floor(min(ws) - 1.5)) if ws else 4.0
    xmax = min(24.0, np.ceil(max(we) + 1.5)) if we else 18.0
    out = {"name": c, "days": days, "xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)}
    if with_raw:
        hours = (v.index.hour + v.index.minute / 60 + v.index.second / 3600).to_numpy(dtype="float32")
        idx = pd.Series(np.arange(len(v)), index=v.index.normalize()).groupby(level=0).agg(["min", "max"])
        T, Y, off = [], [], [0]
        for date in d.index:
            if date in idx.index:
                a, b = int(idx.loc[date, "min"]), int(idx.loc[date, "max"]) + 1
                T.append(hours[a:b])
                Y.append(v.to_numpy(dtype="float32")[a:b])
            off.append(off[-1] + (len(T[-1]) if date in idx.index else 0))
        T = np.concatenate(T) if T else np.zeros(0, "float32")
        Y = np.concatenate(Y) if Y else np.zeros(0, "float32")
        out.update(N=int(len(T)), off=off, blob=base64.b64encode(gzip.compress(T.tobytes() + Y.tobytes(), 6)).decode("ascii"))
    return out, extra


def classes_for(chs):
    present = {x["q"] for ch in chs for x in ch["days"]}
    cls = [{"key": k, "label": lab, "color": col} for k, lab, col in CLASSES if k in present]
    cls += [{"key": k, "label": k.replace("_", " "), "color": "#9aa5b1"} for k in sorted(present - known)]
    return cls


payload_grid = [channel_payload(c, False)[0] for c in channels]
payload_overlay = [channel_payload(c, True)[0] for c in channels]
summary = pd.DataFrame({ch["name"]: pd.Series([x["q"] for x in ch["days"]]).value_counts() for ch in payload_grid}).fillna(0).astype(int)
summary

# %% [markdown]
# ## Write the two pages

# %%
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--cols:__COLS__}
html,body{margin:0;background:#fff;color:#1d2733;font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:12px 20px 4px}
h1{font-size:19px;margin:0 0 2px}
.meta{color:#66727f;font-size:12px}
#bar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;padding:8px 20px;background:#f6f8fb;border-block:1px solid #d6dde6}
#bar a{color:#2563eb;text-decoration:none}
button{font:inherit;padding:2px 10px;border:1px solid #cdd5df;background:#fff;border-radius:6px;cursor:pointer}
button:hover{border-color:#2563eb}
.chip{display:inline-flex;align-items:center;gap:5px;padding:1px 9px 1px 6px;border:1.5px solid var(--c);border-radius:999px;background:#fff;cursor:pointer;user-select:none}
.chip input{accent-color:var(--c);margin:0}
.chip b{font-weight:600;color:#44505c}
section.ch{padding:4px 20px 14px;border-bottom:1px solid #e3e8ef}
h2{font-size:17px;margin:12px 0 0}
h3{font-size:14px;margin:14px 0 6px;display:flex;align-items:center;gap:8px}
h3 i{display:inline-block;width:12px;height:12px;border-radius:3px;background:var(--c)}
h3 span{font-weight:400;color:#66727f;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:10px}
.cell{border:1px solid #dde3ea;border-top:3px solid var(--c);border-radius:6px;padding:4px 6px 6px;background:#fff;min-width:0}
.cell .top{display:flex;justify-content:space-between;align-items:center;font-weight:600;font-size:13px}
.cell .tag{font-weight:500;font-size:11px;color:var(--c);border:1px solid var(--c);border-radius:999px;padding:0 7px;white-space:nowrap}
canvas{display:block;width:100%;aspect-ratio:5/3}
.stats{font-size:11.5px;color:#44505c;line-height:1.45;font-variant-numeric:tabular-nums}
.stats b{color:#1d2733;font-weight:600}
body.nostats .stats{display:none}
.hide{display:none}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="meta">__META__</div></header>
<div id="bar"></div>
<div id="root"></div>
<script>
(async function () {
  const D = __DATA__, MODE = D.mode;
  const CLS = {}; D.classes.forEach((c) => { CLS[c.key] = c; });
  const state = { common: D.common };
  const L = 38, R = 6, T = 6, B = 18;
  const f = (v, d) => (v == null ? '–' : v.toFixed(d));
  const hhmm = (h) => { const m = Math.round(h * 60); return String(Math.floor(m / 60) % 24).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };

  async function inflate(b64) {
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  for (const ch of D.channels) {
    if (ch.blob) {
      const buf = await inflate(ch.blob);
      ch.T = new Float32Array(buf, 0, ch.N);
      ch.Y = new Float32Array(buf, 4 * ch.N, ch.N);
    }
  }

  function niceStep(r) { const e = Math.pow(10, Math.floor(Math.log10(r))), m = r / e; return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * e; }
  function yRange(day, ch) {
    if (state.common) return [ch.ymin, ch.ymax];
    if (day._y) return day._y;
    let top = day.ym != null ? day.ym : (day.pk != null ? day.pk : ch.ymax);
    if (ch.T) {
      const a = ch.off[day.i], b = ch.off[day.i + 1];
      if (b > a) {
        const v = Array.from(ch.Y.subarray(a, b)).filter(Number.isFinite).sort((p, q) => p - q);
        if (v.length) top = Math.max(top, v[Math.floor(0.98 * (v.length - 1))]);
      }
    }
    return (day._y = [ch.ymin, ch.ymin + Math.max(top - ch.ymin, 1e-9) * 1.12]);
  }

  function draw(cv, day, ch, hoverT) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const pw = W - L - R, ph = H - T - B, color = CLS[day.q].color;
    const [y0, y1] = yRange(day, ch), x0 = ch.xmin, x1 = ch.xmax;
    const X = (t) => L + (t - x0) / (x1 - x0) * pw, Y = (v) => T + ph - (v - y0) / (y1 - y0) * ph;
    g.font = '10px system-ui, sans-serif'; g.lineWidth = 1; g.strokeStyle = '#e8edf3'; g.fillStyle = '#66727f';
    const xs = x1 - x0 > 10 ? 2 : 1;
    g.textAlign = 'center';
    for (let t = Math.ceil(x0 / xs) * xs; t <= x1; t += xs) { g.beginPath(); g.moveTo(X(t), T); g.lineTo(X(t), T + ph); g.stroke(); g.fillText(t, X(t), H - 5); }
    const ys = niceStep((y1 - y0) / 4); g.textAlign = 'right';
    for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-12; v += ys) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(L + pw, Y(v)); g.stroke(); g.fillText(+v.toPrecision(3), L - 4, Y(v) + 3); }
    g.save(); g.beginPath(); g.rect(L, T, pw, ph); g.clip();
    if (day.ws != null) { g.fillStyle = color + '22'; g.fillRect(X(day.ws), T, X(day.we) - X(day.ws), ph); }
    let a = 0, b = 0;
    if (ch.T) {
      a = ch.off[day.i]; b = ch.off[day.i + 1];
      g.strokeStyle = '#8a99a8'; g.lineWidth = 0.8; g.beginPath();
      let prev = -1e9;
      for (let i = a; i < b; i++) {
        const v = ch.Y[i], t = ch.T[i];
        if (!Number.isFinite(v)) { prev = -1e9; continue; }
        if (t - prev > 0.25) g.moveTo(X(t), Y(v)); else g.lineTo(X(t), Y(v));   // break the line at data gaps
        prev = t;
      }
      g.stroke();
    }
    if (day.a != null && day.ws != null) {
      g.strokeStyle = '#111'; g.lineWidth = 2; g.beginPath();
      for (let k = 0; k <= 80; k++) { const t = day.ws + (day.we - day.ws) * k / 80, v = day.a * t * t + day.b * t + day.c; if (k) g.lineTo(X(t), Y(v)); else g.moveTo(X(t), Y(v)); }
      g.stroke();
    }
    if (day.tm != null) {
      g.setLineDash([3, 3]); g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); g.moveTo(X(day.tm), T); g.lineTo(X(day.tm), T + ph); g.stroke(); g.setLineDash([]);
      g.fillStyle = color; g.strokeStyle = '#111'; g.lineWidth = 1.2; g.beginPath();
      g.moveTo(X(day.tm), Y(day.ym) - 7); g.lineTo(X(day.tm) - 5, Y(day.ym) + 2); g.lineTo(X(day.tm) + 5, Y(day.ym) + 2); g.closePath(); g.fill(); g.stroke();
    }
    if (day.a == null) { g.fillStyle = '#66727f'; g.textAlign = 'center'; g.font = '12px system-ui, sans-serif'; g.fillText('no fit', L + pw / 2, T + ph / 2); }
    g.restore();
    g.strokeStyle = '#c9d2dc'; g.lineWidth = 1; g.strokeRect(L, T, pw, ph);
    if (hoverT != null && ch.T && b > a) {                                      // hover read-out (overlay page)
      let lo = a, hi = b - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (ch.T[m] < hoverT) lo = m + 1; else hi = m; }
      if (lo > a && Math.abs(ch.T[lo - 1] - hoverT) < Math.abs(ch.T[lo] - hoverT)) lo--;
      const t = ch.T[lo], v = ch.Y[lo];
      if (Number.isFinite(v)) {
        g.strokeStyle = '#1d2733'; g.lineWidth = 1; g.beginPath(); g.moveTo(X(t), T); g.lineTo(X(t), T + ph); g.stroke();
        g.fillStyle = '#1d2733'; g.beginPath(); g.arc(X(t), Y(v), 3, 0, 6.3); g.fill();
        const txt = hhmm(t) + ' UTC   ' + (+v.toPrecision(4)); g.font = '11px system-ui, sans-serif';
        const tw = g.measureText(txt).width + 8, bx = Math.min(Math.max(X(t) - tw / 2, L), L + pw - tw);
        g.fillStyle = 'rgba(29,39,51,.92)'; g.fillRect(bx, T + 2, tw, 16); g.fillStyle = '#fff'; g.textAlign = 'left'; g.fillText(txt, bx + 4, T + 14);
      }
    }
  }

  function statsHtml(d) {
    const l1 = d.tm != null ? `max at <b>${f(d.tm, 2)} h</b> (${hhmm(d.tm)} UTC) &middot; value <b>${f(d.ym, 3)}</b>`
                            : `no reliable maximum &middot; observed peak ${f(d.tpk, 2)} h, ${f(d.pk, 3)}`;
    const l2 = `R&sup2; <b>${f(d.r2, 3)}</b> &middot; cloud-free points <b>${d.cf == null ? '–' : Math.round(d.cf * 100) + ' %'}</b> &middot; n ${d.n == null ? '–' : d.n}`;
    const l3 = d.ws != null ? `fit interval ${f(d.ws, 2)}&ndash;${f(d.we, 2)} h (${f(d.we - d.ws, 1)} h)` : 'no fit interval';
    const l4 = `dips <b>${d.nd == null ? '–' : d.nd}</b> &middot; ${f(d.dh, 1)} h (longest ${f(d.ld, 1)} h) &middot; deepest ${d.dd == null ? '–' : Math.round(d.dd) + ' %'}`;
    return [l1, l2, l3, l4].join('<br>');
  }

  // ---- build the page
  const total = {};
  D.channels.forEach((ch) => ch.days.forEach((d) => { total[d.q] = (total[d.q] || 0) + 1; }));
  const bar = document.getElementById('bar');
  bar.innerHTML = '<span><b>Show:</b></span>' + D.classes.map((c) =>
    `<label class="chip" style="--c:${c.color}"><input type="checkbox" checked data-q="${c.key}">${c.label} <b>${total[c.key] || 0}</b></label>`).join('')
    + '<button id="all">All</button><button id="none">None</button>'
    + '<label>Columns <input type="range" id="cols" min="1" max="10" value="__COLS__"> <span id="colsv">__COLS__</span></label>'
    + '<label><input type="checkbox" id="stat" checked> statistics</label>'
    + '<label><input type="checkbox" id="common"' + (state.common ? ' checked' : '') + '> same vertical scale</label>'
    + '<span>Jump to: ' + D.channels.map((ch, i) => `<a href="#ch${i}">${ch.name}</a>`).join(' &middot; ') + '</span>';

  const root = document.getElementById('root'), cells = [];
  const ro = new ResizeObserver((es) => es.forEach((e) => { const c = e.target._cell; if (c) draw(c.cv, c.day, c.ch); }));
  D.channels.forEach((ch, ci) => {
    const sec = document.createElement('section'); sec.className = 'ch'; sec.id = 'ch' + ci;
    sec.innerHTML = `<h2>${ch.name}</h2>`;
    D.classes.forEach((c) => {
      const ds = ch.days.filter((d) => d.q === c.key);
      if (!ds.length) return;
      const grp = document.createElement('div'); grp.className = 'group'; grp.dataset.q = c.key;
      grp.innerHTML = `<h3 style="--c:${c.color}"><i></i>${c.label} <span>${ds.length} day${ds.length > 1 ? 's' : ''}</span></h3><div class="grid"></div>`;
      const grid = grp.querySelector('.grid');
      ds.forEach((d) => {
        const cell = document.createElement('div'); cell.className = 'cell'; cell.style.setProperty('--c', c.color);
        cell.innerHTML = `<div class="top"><span>${d.date}</span><span class="tag">${c.label}</span></div><canvas></canvas><div class="stats">${statsHtml(d)}</div>`;
        grid.appendChild(cell);
        const cv = cell.querySelector('canvas'), obj = { cv, day: d, ch };
        cv._cell = obj; cells.push(obj); ro.observe(cv);
        if (MODE === 'overlay') {
          cv.addEventListener('mousemove', (e) => {
            const r = cv.getBoundingClientRect(), pw = r.width - L - R;
            draw(cv, d, ch, ch.xmin + (e.clientX - r.left - L) / pw * (ch.xmax - ch.xmin));
          });
          cv.addEventListener('mouseleave', () => draw(cv, d, ch));
        }
      });
      sec.appendChild(grp);
    });
    root.appendChild(sec);
  });

  const redrawAll = () => cells.forEach((c) => draw(c.cv, c.day, c.ch));
  const setGroup = (q, on) => document.querySelectorAll(`.group[data-q="${q}"]`).forEach((g) => g.classList.toggle('hide', !on));
  bar.querySelectorAll('input[data-q]').forEach((cb) => { cb.onchange = () => { setGroup(cb.dataset.q, cb.checked); }; });
  const setAll = (on) => bar.querySelectorAll('input[data-q]').forEach((cb) => { cb.checked = on; setGroup(cb.dataset.q, on); });
  document.getElementById('all').onclick = () => setAll(true);
  document.getElementById('none').onclick = () => setAll(false);
  document.getElementById('cols').oninput = (e) => { document.documentElement.style.setProperty('--cols', e.target.value); document.getElementById('colsv').textContent = e.target.value; };
  document.getElementById('stat').onchange = (e) => document.body.classList.toggle('nostats', !e.target.checked);
  document.getElementById('common').onchange = (e) => { state.common = e.target.checked; redrawAll(); };
  window.__cells = cells;                                                      // for testing
})();
</script>
</body>
</html>
"""


def write_page(path, payload, mode, same_y):
    cls = classes_for(payload)
    data = json.dumps({"mode": mode, "common": same_y, "classes": cls, "channels": payload}, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    n_days = sum(len(ch["days"]) for ch in payload)
    what = ("fitted parabolas (hours of the day, UTC)" if mode == "grid"
            else "fitted parabolas over the original signal (grey; shaded = fitted interval; hours of the day, UTC)")
    meta = f"{par.index.min():%Y-%m-%d} &rarr; {par.index.max():%Y-%m-%d} &middot; {n_days} plots &middot; {what}"
    ttl = html.escape(title + (" (grid)" if mode == "grid" else " (over the signal)"))
    page = (TEMPLATE.replace("__TITLE__", ttl).replace("__META__", meta).replace("__COLS__", str(columns0))
            .replace("__DATA__", data))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


write_page(product["grid"], payload_grid, "grid", grid_same_y)
write_page(product["overlay"], payload_overlay, "overlay", overlay_same_y)

# %% [markdown]
# ## Timeline page
# The whole record of every channel in one scrollable plot, the parabolas of all days drawn over it (time in UTC).

# %%
def timeline_payload():
    arrays, chans = {}, []
    for ci, c in enumerate(channels):
        info = next(ch for ch in payload_grid if ch["name"] == c)
        v = raw[["timestamp", c]].copy()
        v[c] = v[c].astype("float64")
        if sentinel_values:
            v[c] = v[c].mask(v[c].isin(sentinel_values))
        v = v.dropna().groupby("timestamp")[c].mean().sort_index()
        t = ((v.index - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")).to_numpy(dtype="float64")
        arrays[f"t{ci}"] = {"type": "f8", "data": base64.b64encode(gzip.compress(t.tobytes(), 6)).decode("ascii")}
        arrays[f"v{ci}"] = {"type": "f4", "data": base64.b64encode(gzip.compress(v.to_numpy(dtype="float32").tobytes(), 6)).decode("ascii")}
        days = []
        for d in info["days"]:
            d = dict(d)
            d["t0"] = int((pd.Timestamp(d["date"]) - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s"))
            days.append(d)
        chans.append({"name": c, "days": days, "ymin": info["ymin"], "ymax": info["ymax"]})
    return {"channels": chans, "arrays": arrays, "classes": classes_for(payload_grid), "panelHeight": timeline_height,
            "initialDays": timeline_initial_days,
            "meta": [f"{par.index.min():%Y-%m-%d} to {par.index.max():%Y-%m-%d} UTC",
                     f"{sum(len(ch['days']) for ch in payload_grid)} parabolas in {len(channels)} channel(s)"]}


TIMELINE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--bg:#ffffff;--panel:#fbfcfe;--fg:#1d2733;--muted:#66727f;--grid:rgba(100,116,139,.22);--border:#d6dde6;--raw:#8a99a8;--sel:rgba(37,99,235,.14);--selb:#2563eb}
html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
body{display:flex;flex-direction:column}
header{padding:10px 16px 2px}
h1{font-size:18px;margin:0 0 2px}
.meta{color:var(--muted);font-size:12px}
#toolbar{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:6px 16px}
#toolbar .grp{display:inline-flex;gap:4px;align-items:center}
button,select,input[type=date]{font:inherit;padding:2px 9px;border:1px solid var(--border);background:var(--panel);color:var(--fg);border-radius:6px;cursor:pointer}
button:hover,select:hover{border-color:var(--selb)}
label{color:var(--muted);font-size:13px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:1px 9px 1px 6px;border:1.5px solid var(--c);border-radius:999px;background:#fff;cursor:pointer;user-select:none;color:var(--fg)}
.chip input{accent-color:var(--c);margin:0}
.chip b{font-weight:600;color:#44505c}
.tag{font-weight:500;font-size:11px;color:var(--c);border:1px solid var(--c);border-radius:999px;padding:0 7px;white-space:nowrap}
#scale{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
#info{padding:2px 16px 6px;min-height:38px;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
#info .l2{color:var(--fg);margin-top:2px}
#info b{font-weight:600}
#scroller{flex:1;min-height:0;overflow:auto;border-top:1px solid var(--border);position:relative;cursor:crosshair;background:var(--panel)}
#scroller.pan{cursor:grab}#scroller.pan.grabbing{cursor:grabbing}
.panel{position:sticky;left:0;border-bottom:1px solid var(--border);overflow:hidden}
.panel canvas{position:absolute;left:0;top:0}
.panel canvas.over{pointer-events:none}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="meta" id="meta"></div></header>
<div id="toolbar"></div>
<div id="info">Decoding data&hellip;</div>
<div id="scroller"><div id="track"></div></div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(async function () {
'use strict';
const $ = (id) => document.getElementById(id);
const P = JSON.parse($('payload').textContent);
const info = $('info'), scroller = $('scroller'), track = $('track');
const HELP = 'Hover: values at that time and the fit of that day  ·  Ctrl+wheel or pinch: zoom  ·  drag: zoom box (or pan)  ·  scroll bar / Shift+wheel: scroll  ·  double-click: fit all';
$('meta').textContent = P.meta.join('   ·   ');
if (typeof DecompressionStream === 'undefined') { info.textContent = 'This browser cannot decompress the data (DecompressionStream missing). Use a current Chrome, Edge, Firefox or Safari.'; return; }
async function decode(e) {
  const bin = atob(e.data), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const buf = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  return e.type === 'f8' ? new Float64Array(buf) : new Float32Array(buf);
}
const A = {};
await Promise.all(Object.keys(P.arrays).map(async (k) => { A[k] = await decode(P.arrays[k]); }));

// ---------- helpers ----------
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const pad = (n) => String(n).padStart(2, '0');
const f = (v, d) => (v == null ? '–' : v.toFixed(d));
const hhmm = (h) => { const m = Math.round(h * 60); return pad(Math.floor(m / 60) % 24) + ':' + pad(m % 60); };
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
  if (step !== undefined) { for (let t = Math.ceil(tv0 / step) * step; t <= tv1; t += step) out.push({ t, label: fmtTick(t, step) }); return out; }
  const m = MON_STEPS.find((k) => k * 2629800 * pps >= minPx) || 120, d0 = new Date(tv0 * 1000);
  let y = d0.getUTCFullYear(), mo = d0.getUTCMonth();
  for (let g = 0; g < 5000; g++) {
    const t = Date.UTC(y, mo, 1) / 1000; if (t > tv1) break;
    const ok = m <= 12 ? (mo % m === 0) : (mo === 0 && y % (m / 12) === 0);
    if (ok && t >= tv0) out.push({ t, label: m < 12 ? `${y}-${pad(mo + 1)}` : String(y) });
    if (++mo > 11) { mo = 0; y++; }
  }
  return out;
}

// ---------- toolbar ----------
const CLS = {}; P.classes.forEach((c) => { CLS[c.key] = c; });
const total = {}; P.channels.forEach((ch) => ch.days.forEach((d) => { total[d.q] = (total[d.q] || 0) + 1; }));
const on = {}; P.classes.forEach((c) => { on[c.key] = true; });
const show = { signal: true, shade: false, vertex: true };
$('toolbar').innerHTML =
  '<span><b>Show:</b></span>' + P.classes.map((c) => `<label class="chip" style="--c:${c.color}"><input type="checkbox" checked data-q="${c.key}">${c.label} <b>${total[c.key] || 0}</b></label>`).join('')
  + '<span class="grp"><button id="all">All</button><button id="none">None</button></span>'
  + '<label><input type="checkbox" id="sig" checked> signal</label><label><input type="checkbox" id="shade"> fitted intervals shaded</label><label><input type="checkbox" id="vtx" checked> maxima marked</label>'
  + '<span class="grp"><button id="zout" title="Zoom out">&minus;</button><button id="zin" title="Zoom in">+</button></span>'
  + '<span class="grp">Window: <button data-win="86400">1 day</button><button data-win="604800">1 week</button><button data-win="2629800">1 month</button><button id="fit">Fit all</button></span>'
  + '<label>Go to day <input type="date" id="goto"></label>'
  + '<label>Mouse drag <select id="dragmode"><option value="zoom">zooms to a box</option><option value="pan">pans</option></select></label>'
  + '<label>Y axis <select id="ymode"><option value="visible">fits the visible range</option><option value="full">uses the full data range</option></select></label>'
  + '<span id="scale"></span>';

// ---------- data ----------
const CH = P.channels.map((c, ci) => {
  const t = A['t' + ci], v = A['v' + ci]; let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < v.length; i++) { const x = v[i]; if (x === x) { if (x < mn) mn = x; if (x > mx) mx = x; } }
  return { name: c.name, t, v, days: c.days, t0s: c.days.map((d) => d.t0), byDay: new Map(c.days.map((d) => [d.t0, d])), step: medianStep(t), mn, mx, i0: 0, i1: 0, lo: 0, hi: 1 };
});
const T0 = Math.min(...CH.map((c) => c.t[0])), T1 = Math.max(...CH.map((c) => c.t[c.t.length - 1]));
const span = Math.max(T1 - T0, 1), medStep = CH[0].step;
const LM = 68, RM = 14, TOPM = 28, BOTM = 24, H = P.panelHeight;
let VW = 0, PW = 0, pps = 1, ppsMin = 1, ppsMax = 1, dpr = 1;
let cursorT = null, cursorPanel = -1, dragMode = 'zoom', yMode = 'visible', sel = null, drag = false, lastX = 0;
const panels = CH.map((c) => {
  const el = document.createElement('div'); el.className = 'panel'; el.style.height = H + 'px';
  const cd = document.createElement('canvas'), co = document.createElement('canvas'); co.className = 'over';
  el.append(cd, co); track.append(el);
  return { ch: c, el, cd, co, ctx: null, octx: null };
});

// ---------- drawing ----------
const val = (d, h) => (d.a * h + d.b) * h + d.c;
function extremes(d) {                       // min and max of the parabola over its fitted interval
  const e = [val(d, d.ws), val(d, d.we)];
  const hv = -d.b / (2 * d.a);
  if (d.a !== 0 && hv > d.ws && hv < d.we) e.push(val(d, hv));
  return [Math.min(...e), Math.max(...e)];
}
function visibleDays(ch, tv0, tv1) {
  const out = [];
  for (let i = Math.max(0, lowerBound(ch.t0s, tv0 - 86400)); i < ch.days.length && ch.t0s[i] <= tv1; i++) {
    const d = ch.days[i]; if (on[d.q] && d.a != null && d.ws != null) out.push(d);
  }
  return out;
}
function drawSignal(ctx, ch, tv0, Y) {
  const n = ch.i1 - ch.i0;
  ctx.strokeStyle = ch.color = '#8a99a8'; ctx.fillStyle = '#8a99a8'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  if (n > PW * 2) {                           // min/max envelope per pixel column: no spike or dip is lost
    const mins = new Float32Array(PW).fill(Infinity), maxs = new Float32Array(PW).fill(-Infinity);
    for (let i = ch.i0; i < ch.i1; i++) {
      const v = ch.v[i]; if (v !== v) continue;
      const x = Math.floor((ch.t[i] - tv0) * pps); if (x < 0 || x >= PW) continue;
      if (v < mins[x]) mins[x] = v; if (v > maxs[x]) maxs[x] = v;
    }
    ctx.lineWidth = 1; ctx.beginPath();
    for (let x = 0; x < PW; x++) {
      if (mins[x] === Infinity) continue;
      const xx = LM + x + 0.5, ya = Y(mins[x]); let yb = Y(maxs[x]); if (ya - yb < 1) yb = ya - 1;
      ctx.moveTo(xx, ya); ctx.lineTo(xx, yb);
    }
    ctx.stroke(); return;
  }
  ctx.lineWidth = 1; ctx.beginPath();
  let pen = false, prevT = 0; const gap = 5 * ch.step;
  for (let i = ch.i0; i < ch.i1; i++) {
    const v = ch.v[i]; if (v !== v) { pen = false; continue; }
    const t = ch.t[i], x = LM + (t - tv0) * pps, y = Y(v);
    if (pen && t - prevT <= gap) ctx.lineTo(x, y); else { ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y); }
    pen = true; prevT = t;
  }
  ctx.stroke();
  if (pps * ch.step > 7) for (let i = ch.i0; i < ch.i1; i++) { const v = ch.v[i]; if (v !== v) continue; ctx.beginPath(); ctx.arc(LM + (ch.t[i] - tv0) * pps, Y(v), 1.9, 0, 6.2832); ctx.fill(); }
}
function pathParab(ctx, d, tv0, Y) {
  const xa = LM + (d.t0 + d.ws * 3600 - tv0) * pps, xb = LM + (d.t0 + d.we * 3600 - tv0) * pps;
  const n = clamp(Math.ceil((xb - xa) / 3), 2, 90);
  ctx.beginPath();
  for (let k = 0; k <= n; k++) {
    const h = d.ws + (d.we - d.ws) * k / n, x = LM + (d.t0 + h * 3600 - tv0) * pps, y = Y(val(d, h));
    if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  }
}
function drawPanel(p, tv0, tv1) {
  const ctx = p.ctx, ch = p.ch, px1 = LM + PW, top = TOPM, bot = H - BOTM, ph = bot - top;
  ctx.fillStyle = '#fbfcfe'; ctx.fillRect(0, 0, VW, H);
  ch.i0 = Math.max(0, lowerBound(ch.t, tv0) - 1); ch.i1 = Math.min(ch.t.length, lowerBound(ch.t, tv1) + 1);
  const dv = visibleDays(ch, tv0, tv1);
  let lo = Infinity, hi = -Infinity;
  if (yMode === 'visible') {
    if (show.signal) for (let i = ch.i0; i < ch.i1; i++) { const x = ch.v[i]; if (x === x) { if (x < lo) lo = x; if (x > hi) hi = x; } }
    for (const d of dv) { const [a, b] = extremes(d); if (a < lo) lo = a; if (b > hi) hi = b; }
  } else { if (show.signal) { lo = ch.mn; hi = ch.mx; } for (const d of ch.days) if (on[d.q] && d.a != null) { const [a, b] = extremes(d); if (a < lo) lo = a; if (b > hi) hi = b; } }
  if (!(lo <= hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
  const padv = (hi - lo) * 0.04; lo -= padv; hi += padv; ch.lo = lo; ch.hi = hi;
  const Y = (v) => bot - (v - lo) / (hi - lo) * ph;
  ctx.font = '11px system-ui, sans-serif'; ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(100,116,139,.22)'; ctx.fillStyle = '#66727f';
  const yt = niceTicks(lo, hi, Math.max(2, Math.floor(ph / 40)));
  ctx.beginPath(); for (const v of yt.ticks) { const y = Math.round(Y(v)) + 0.5; ctx.moveTo(LM, y); ctx.lineTo(px1, y); } ctx.stroke();
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; for (const v of yt.ticks) ctx.fillText(v.toFixed(yt.dec), LM - 7, Y(v));
  const xt = timeTicks(tv0, tv1, pps);
  ctx.beginPath(); for (const k of xt) { const x = Math.round(LM + (k.t - tv0) * pps) + 0.5; ctx.moveTo(x, top); ctx.lineTo(x, bot); } ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'top'; for (const k of xt) ctx.fillText(k.label, LM + (k.t - tv0) * pps, bot + 6);
  ctx.strokeRect(LM + 0.5, top + 0.5, PW, ph);
  ctx.save(); ctx.beginPath(); ctx.rect(LM, top, PW, ph); ctx.clip();
  if (show.shade) for (const d of dv) {
    const xa = LM + (d.t0 + d.ws * 3600 - tv0) * pps, xb = LM + (d.t0 + d.we * 3600 - tv0) * pps;
    ctx.fillStyle = CLS[d.q].color + '22'; ctx.fillRect(xa, top, Math.max(1, xb - xa), ph);
  }
  if (show.signal) drawSignal(ctx, ch, tv0, Y);
  ctx.lineWidth = 2.2; ctx.lineCap = 'round';
  for (const d of dv) { ctx.strokeStyle = CLS[d.q].color; pathParab(ctx, d, tv0, Y); ctx.stroke(); }
  if (show.vertex && pps * 86400 > 14) for (const d of dv) {
    if (d.tm == null) continue;
    const x = LM + (d.t0 + d.tm * 3600 - tv0) * pps, y = Y(d.ym);
    ctx.fillStyle = CLS[d.q].color; ctx.strokeStyle = '#111'; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(x, y - 6); ctx.lineTo(x - 4.5, y + 2); ctx.lineTo(x + 4.5, y + 2); ctx.closePath(); ctx.fill(); ctx.stroke();
  }
  ctx.restore();
  ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '600 12px system-ui, sans-serif'; ctx.fillStyle = '#1d2733';
  ctx.fillText(ch.name, LM + 4, 7);
}
function hoveredDay(p) {
  if (cursorT === null) return null;
  const d = p.ch.byDay.get(Math.floor(cursorT / 86400) * 86400);
  return d && on[d.q] && d.a != null && d.ws != null ? d : null;
}
function drawOverlay() {
  const tv0 = T0 + scroller.scrollLeft / pps;
  panels.forEach((p, pi) => {
    const o = p.octx; o.clearRect(0, 0, VW, H);
    if (sel) {
      const a = Math.min(sel.x0, sel.x1), b = Math.max(sel.x0, sel.x1);
      o.fillStyle = 'rgba(37,99,235,.14)'; o.fillRect(a, TOPM, b - a, H - TOPM - BOTM);
      o.strokeStyle = '#2563eb'; o.lineWidth = 1; o.strokeRect(a + 0.5, TOPM + 0.5, b - a, H - TOPM - BOTM - 1);
    }
    if (cursorT !== null) {
      const x = LM + (cursorT - tv0) * pps;
      if (x >= LM && x <= LM + PW) {
        o.strokeStyle = '#66727f'; o.lineWidth = 1; o.setLineDash([4, 3]); o.beginPath();
        o.moveTo(Math.round(x) + 0.5, TOPM); o.lineTo(Math.round(x) + 0.5, H - BOTM); o.stroke(); o.setLineDash([]);
      }
    }
    const d = pi === cursorPanel ? hoveredDay(p) : null;       // highlight the parabola of the hovered day
    if (d) {
      const ch = p.ch, bot = H - BOTM, ph = bot - TOPM, Y = (v) => bot - (v - ch.lo) / (ch.hi - ch.lo) * ph;
      o.save(); o.beginPath(); o.rect(LM, TOPM, PW, ph); o.clip();
      const xa = LM + (d.t0 + d.ws * 3600 - tv0) * pps, xb = LM + (d.t0 + d.we * 3600 - tv0) * pps;
      o.fillStyle = CLS[d.q].color + '2e'; o.fillRect(xa, TOPM, Math.max(1, xb - xa), ph);
      o.lineCap = 'round'; o.strokeStyle = '#111'; o.lineWidth = 5; pathParab(o, d, tv0, Y); o.stroke();
      o.strokeStyle = CLS[d.q].color; o.lineWidth = 3; pathParab(o, d, tv0, Y); o.stroke();
      o.restore();
    }
  });
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
function setScale(newPps, ax, ta) { pps = clamp(newPps, ppsMin, ppsMax); applyTrack(); scroller.scrollLeft = (ta - ax / pps - T0) * pps; schedule(); }
function centerTime() { return T0 + scroller.scrollLeft / pps + PW / (2 * pps); }
function zoomBy(k) { setScale(pps * k, PW / 2, centerTime()); }
function showSeconds(sec, center) { setScale(PW / sec, PW / 2, center == null ? centerTime() : center); }
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
  for (const p of panels) {
    const c = p.ch, i = nearest(c.t, cursorT); let txt = '–';
    if (Math.abs(c.t[i] - cursorT) <= 1.5 * c.step) { const v = c.v[i]; if (v === v) txt = String(Number(v.toPrecision(6))); }
    parts.push(`${c.name} ${txt}`);
  }
  let l2 = '';
  const p = panels[cursorPanel];
  if (p) {
    const d = p.ch.byDay.get(Math.floor(cursorT / 86400) * 86400);
    if (!d) l2 = `${p.ch.name}: no fit for this day`;
    else if (!on[d.q]) l2 = `${p.ch.name} ${d.date}: <span class="tag" style="--c:${CLS[d.q].color}">${CLS[d.q].label}</span> (hidden, class switched off)`;
    else {
      const tag = `<span class="tag" style="--c:${CLS[d.q].color}">${CLS[d.q].label}</span>`;
      const mx = d.tm != null ? `max at <b>${f(d.tm, 2)} h</b> (${hhmm(d.tm)} UTC), value <b>${f(d.ym, 3)}</b>` : `no reliable maximum (observed peak ${f(d.tpk, 2)} h, ${f(d.pk, 3)})`;
      l2 = `<b>${p.ch.name} ${d.date}</b> ${tag} ${mx} &middot; R&sup2; <b>${f(d.r2, 3)}</b> &middot; cloud-free <b>${d.cf == null ? '–' : Math.round(d.cf * 100) + ' %'}</b> &middot; n ${d.n == null ? '–' : d.n}`
         + ` &middot; interval ${d.ws != null ? f(d.ws, 2) + '–' + f(d.we, 2) + ' h (' + f(d.we - d.ws, 1) + ' h)' : '–'}`
         + ` &middot; dips <b>${d.nd == null ? '–' : d.nd}</b>, ${f(d.dh, 1)} h (longest ${f(d.ld, 1)} h), deepest ${d.dd == null ? '–' : Math.round(d.dd) + ' %'}`;
    }
  }
  info.innerHTML = `<div>${parts.join('   ·   ')}</div><div class="l2">${l2}</div>`;
}
scroller.addEventListener('scroll', schedule);
scroller.addEventListener('mousemove', (e) => {
  const m = pointer(e);
  if (drag) {
    if (dragMode === 'pan') { scroller.scrollLeft -= e.clientX - lastX; lastX = e.clientX; }
    else if (sel) sel.x1 = clamp(m.x, LM, LM + PW);
  }
  cursorT = m.ok ? T0 + scroller.scrollLeft / pps + (m.x - LM) / pps : null;
  cursorPanel = m.ok ? Math.floor((m.y + scroller.scrollTop) / H) : -1;
  updateInfo(); drawOverlay();
});
scroller.addEventListener('mouseleave', () => { if (!drag) { cursorT = null; cursorPanel = -1; updateInfo(); drawOverlay(); } });
scroller.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  const m = pointer(e); if (!m.ok) return;
  e.preventDefault(); drag = true; lastX = e.clientX;
  if (dragMode === 'zoom') sel = { x0: m.x, x1: m.x }; else scroller.classList.add('grabbing');
});
window.addEventListener('mouseup', () => {
  if (!drag) return; drag = false; scroller.classList.remove('grabbing');
  if (dragMode === 'zoom' && sel) {
    const a = Math.min(sel.x0, sel.x1), b = Math.max(sel.x0, sel.x1); sel = null;
    if (b - a >= 8) { const tv0 = T0 + scroller.scrollLeft / pps; setScale(PW / ((b - a) / pps), 0, tv0 + (a - LM) / pps); } else drawOverlay();
  }
});
scroller.addEventListener('wheel', (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  const ax = clamp(pointer(e).x - LM, 0, PW), ta = T0 + scroller.scrollLeft / pps + ax / pps;
  setScale(pps * Math.exp(-e.deltaY * 0.002), ax, ta);
}, { passive: false });
scroller.addEventListener('dblclick', () => setScale(ppsMin, 0, T0));
$('zin').onclick = () => zoomBy(2); $('zout').onclick = () => zoomBy(0.5); $('fit').onclick = () => setScale(ppsMin, 0, T0);
document.querySelectorAll('[data-win]').forEach((b) => { b.onclick = () => showSeconds(+b.dataset.win); });
$('goto').onchange = (e) => { const t = Date.parse(e.target.value + 'T12:00:00Z') / 1000; if (t === t) showSeconds(86400, t); };
$('dragmode').onchange = (e) => { dragMode = e.target.value; scroller.classList.toggle('pan', dragMode === 'pan'); };
$('ymode').onchange = (e) => { yMode = e.target.value; schedule(); };
document.querySelectorAll('input[data-q]').forEach((cb) => { cb.onchange = () => { on[cb.dataset.q] = cb.checked; schedule(); updateInfo(); }; });
const setAll = (v) => document.querySelectorAll('input[data-q]').forEach((cb) => { cb.checked = v; on[cb.dataset.q] = v; });
$('all').onclick = () => { setAll(true); schedule(); updateInfo(); };
$('none').onclick = () => { setAll(false); schedule(); updateInfo(); };
$('sig').onchange = (e) => { show.signal = e.target.checked; schedule(); };
$('shade').onchange = (e) => { show.shade = e.target.checked; schedule(); };
$('vtx').onchange = (e) => { show.vertex = e.target.checked; schedule(); };
new ResizeObserver(() => resize(false)).observe(scroller);
resize(true); updateInfo();
document.body.dataset.ready = '1';
window.__tl = { panels, CH, get pps() { return pps; }, scroller, setScale, showSeconds, T0, T1, on };   // for testing
})();
</script>
</body>
</html>
"""


def write_timeline(path):
    data = json.dumps(timeline_payload(), separators=(",", ":")).replace("</", "<\\/")
    page = TIMELINE_TEMPLATE.replace("__TITLE__", html.escape(title + " (timeline)")).replace("__PAYLOAD__", data)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


write_timeline(product["timeline"])
