# %% [markdown]
# # 4. 3D surface plots (HTML)
# One interactive 3D surface per column of the `load` output (original data), on ONE self-contained
# HTML page (Plotly is embedded, no internet needed).
#
# Every plot is `z = f(x, y)`:
# - x = time of day (UTC), the record is sliced into single days
# - y = the date
# - z = the value of the column (aggregated into bins of `x_bin`, e.g. 10 min, to keep the page light)
#
# Days without data (outages) appear as holes in the surface, they are never interpolated.
# Interaction: drag = rotate, wheel = zoom, right-drag / Shift-drag = pan, hover = date, time, value.
# The cameras of all plots can be kept in sync (checkbox on the page). Settings: `plots3d_config` in env.yaml.

# %% tags=["parameters"]
upstream = None
product = None
sentinel_values = None
config = None

# %%
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

title = config.get("title", "UV measurements: daily 3D view")
x_bin = config.get("x_bin", "10min")
aggregate = config.get("aggregate", "mean")
selected = config.get("channels")
mask_sentinels = bool(config.get("mask_sentinels", True))
colorscale = config.get("colorscale", "Viridis")
plot_height = int(config.get("plot_height", 680))
aspect = config.get("aspect", [1.4, 2.0, 0.8])
tick_hours = float(config.get("x_tick_hours", 3))
opacity = float(config.get("opacity", 1.0))
if not 0 < opacity <= 1:
    raise ValueError(f"opacity must be in (0, 1] (got {opacity})")

# Value-dependent opacity: None, a Plotly preset name, or a FLAT list of (position, opacity) pairs,
# e.g. [0, 0.0, 0.05, 0.25, 1, 1]  = pairs (0 -> 0.0), (0.05 -> 0.25), (1 -> 1);  position 0 = lowest value, 1 = highest
opacityscale = config.get("opacityscale")
if isinstance(opacityscale, (list, tuple)):
    flat = [x for item in opacityscale for x in (item if isinstance(item, (list, tuple)) else [item])]
    if len(flat) < 4 or len(flat) % 2:
        raise ValueError("opacityscale needs at least two (position, opacity) pairs, e.g. [0, 0.1, 1, 1]")
    opacityscale = [[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat), 2)]
    if opacityscale[0][0] != 0 or opacityscale[-1][0] != 1:
        raise ValueError("opacityscale must start at position 0 and end at position 1")
elif opacityscale is not None and opacityscale not in ("min", "max", "extremes", "uniform"):
    raise ValueError(f"opacityscale must be null, min, max, extremes, uniform or a list of pairs (got {opacityscale!r})")

if aggregate not in ("mean", "median", "max", "min"):
    raise ValueError(f"aggregate must be mean, median, max or min (got {aggregate!r})")
bin_s = int(pd.Timedelta(x_bin).total_seconds())
if bin_s < 1 or 86400 % bin_s:
    raise ValueError(f"x_bin={x_bin!r} must divide one day evenly (e.g. 1min, 5min, 10min, 15min, 30min, 1h)")
n_bins = 86400 // bin_s

raw = pd.read_csv(upstream["load"]["data"], parse_dates=["timestamp"])
available = [c for c in raw.columns if c not in ("timestamp", "time_raw", "source_file")]
channels = list(selected) if selected else available
unknown = [c for c in channels if c not in available]
if unknown:
    raise KeyError(f"Channel(s) {unknown} not in the load output; available: {available}")

# %% [markdown]
# ## Slice into days and time-of-day bins
# Rows get a day index (y) and a bin index (x). The grid covers EVERY calendar day between the first and the
# last measurement, so a missing day is an empty row (a hole), not a stretched surface.

# %%
ts = raw["timestamp"]
day = ts.dt.floor("D")
bin_idx = ((ts - day).dt.total_seconds() // bin_s).astype(int).to_numpy()
days = pd.date_range(day.min(), day.max(), freq="D")
n_days = len(days)
if n_days < 2:
    raise ValueError("The data cover a single day; at least two days are needed for a 3D surface.")
day_idx = ((day - days[0]) / pd.Timedelta("1D")).round().astype(int).to_numpy()

fmt_t = "%H:%M" if bin_s % 60 == 0 else "%H:%M:%S"
x_labels = [(pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=i * bin_s)).strftime(fmt_t) for i in range(n_bins)]
step = max(1, int(round(tick_hours * 3600 / bin_s)))
tick_idx = list(range(0, n_bins, step))
y_dates = days.strftime("%Y-%m-%d").tolist()
print(f"{len(raw):,} rows -> grid of {n_days} days x {n_bins} bins ({x_bin}) = {n_days * n_bins:,} cells per column")

# %%
surfaces, masked = {}, {}
for c in channels:
    v = raw[c].astype("float64")
    if mask_sentinels and sentinel_values:
        is_sentinel = v.isin(sentinel_values)
        masked[c] = int(is_sentinel.sum())
        v = v.mask(is_sentinel)
    g = pd.DataFrame({"d": day_idx, "b": bin_idx, "v": v.to_numpy()}).groupby(["d", "b"])["v"].agg(aggregate)
    z = np.full((n_days, n_bins), np.nan)
    z[g.index.get_level_values(0).to_numpy(), g.index.get_level_values(1).to_numpy()] = g.to_numpy()
    surfaces[c] = z.astype("float32")

summary = pd.DataFrame({
    "days": n_days, "bins_per_day": n_bins,
    "cells_with_data": {c: int(np.isfinite(z).sum()) for c, z in surfaces.items()},
    "empty_days": {c: int((~np.isfinite(z).any(axis=1)).sum()) for c, z in surfaces.items()},
    "sentinels_hidden": {c: masked.get(c, 0) for c in surfaces},
    "z_min": {c: float(np.nanmin(z)) for c, z in surfaces.items()},
    "z_max": {c: float(np.nanmax(z)) for c, z in surfaces.items()},
})
summary

# %% [markdown]
# ## Build one figure per column

# %%
CAMERA = dict(eye=dict(x=1.7, y=-1.9, z=0.95), center=dict(x=0, y=0, z=-0.1), up=dict(x=0, y=0, z=1))
PLOT_CONFIG = {"displaylogo": False, "responsive": True}
ids, blocks = [], []
for c, z in surfaces.items():
    fig = go.Figure(go.Surface(
        x=x_labels, y=y_dates, z=z, colorscale=colorscale, connectgaps=False,
        opacity=opacity, **({"opacityscale": opacityscale} if opacityscale else {}),
        lighting=dict(ambient=0.85, diffuse=0.45, specular=0.03, roughness=0.9, fresnel=0.05),  # soft: no speckle on noise
        colorbar=dict(title=dict(text=c), thickness=14, len=0.7),
        hovertemplate="date %{y|%Y-%m-%d}<br>time %{x} UTC<br>" + f"{c} ({aggregate}) " + "%{z:.4g}<extra></extra>"))
    fig.update_layout(
        height=plot_height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#ffffff",
        scene=dict(
            xaxis=dict(title=dict(text="time of day (UTC)"), type="category", tickmode="array",
                       tickvals=[x_labels[i] for i in tick_idx], ticktext=[x_labels[i] for i in tick_idx]),
            yaxis=dict(title=dict(text="date"), type="date", tickformat="%Y-%m-%d"),
            zaxis=dict(title=dict(text=c)),
            aspectmode="manual", aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]), camera=CAMERA))
    div_id = f"surf_{len(ids)}"
    ids.append(div_id)
    note = f"{int(np.isfinite(z).sum()):,} of {z.size:,} cells with data"
    if masked.get(c):
        note += f"  &middot;  {masked[c]:,} sentinel values hidden"
    blocks.append(f'<section><h2>{html.escape(c)}</h2><div class="note">{note}</div>'
                  + fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config=PLOT_CONFIG)
                  + "</section>")

# %% [markdown]
# ## Write the page

# %%
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light}
html,body{margin:0;background:#fff;color:#1d2733;font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:12px 20px 4px}
h1{font-size:19px;margin:0 0 2px}
.meta,.note{color:#66727f;font-size:12px}
#bar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 20px;background:#f6f8fb;border-block:1px solid #d6dde6}
#bar a{color:#2563eb;text-decoration:none}
button{font:inherit;padding:3px 10px;border:1px solid #cdd5df;background:#fff;border-radius:6px;cursor:pointer}
button:hover{border-color:#2563eb}
section{padding:6px 20px 14px;border-bottom:1px solid #e3e8ef}
h2{font-size:16px;margin:10px 0 0}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="meta">__META__</div></header>
<div id="bar">
  <label><input type="checkbox" id="sync" checked> Keep the cameras of all plots in sync</label>
  <button id="reset">Reset all views</button>
  <label>Days from <input type="date" id="d0" min="__DMIN__" max="__DMAX__" value="__DMIN__"> to <input type="date" id="d1" min="__DMIN__" max="__DMAX__" value="__DMAX__"></label>
  <button id="full">Full record</button>
  <label>Opacity <input type="range" id="opa" min="0.1" max="1" step="0.05" value="__OPACITY__"> <span id="opav"></span></label>
  <span>Jump to: __NAV__</span>
  <span class="meta">drag = rotate &middot; wheel = zoom &middot; right-drag or Shift+drag = pan &middot; hover = values &middot; double-click = reset one plot</span>
</div>
__FIGS__
<script>
(function () {
  const ids = __IDS__, CAM0 = __CAMERA__;
  const gds = ids.map((id) => document.getElementById(id));
  const camOf = (gd) => (gd.layout.scene && gd.layout.scene.camera) || gd._fullLayout.scene.camera;
  const same = (a, b) => ['eye', 'up', 'center'].every((k) => ['x', 'y', 'z'].every((d) => Math.abs(a[k][d] - b[k][d]) < 1e-6));
  gds.forEach((gd) => gd.on('plotly_relayout', (ev) => {
    const cam = ev['scene.camera'];
    if (!cam || !document.getElementById('sync').checked) return;
    gds.filter((o) => o !== gd && !same(camOf(o), cam)).forEach((o) => Plotly.relayout(o, { 'scene.camera': cam }));
  }));
  document.getElementById('reset').onclick = () => gds.forEach((gd) => Plotly.relayout(gd, { 'scene.camera': CAM0 }));
  const d0 = document.getElementById('d0'), d1 = document.getElementById('d1'), DMIN = '__DMIN__', DMAX = '__DMAX__';
  const applySpan = () => {
    let a = d0.value || DMIN, b = d1.value || DMAX;
    if (a > b) { [a, b] = [b, a]; d0.value = a; d1.value = b; }
    const full = a === DMIN && b === DMAX;
    gds.forEach((gd) => Plotly.relayout(gd, full ? { 'scene.yaxis.autorange': true } : { 'scene.yaxis.range': [a, b] }));
  };
  d0.onchange = d1.onchange = applySpan;
  document.getElementById('full').onclick = () => { d0.value = DMIN; d1.value = DMAX; applySpan(); };
  const opa = document.getElementById('opa'), opav = document.getElementById('opav');
  let timer = 0;
  opav.textContent = (+opa.value).toFixed(2);
  opa.oninput = () => {
    opav.textContent = (+opa.value).toFixed(2);
    clearTimeout(timer);                                  // small debounce: redrawing four WebGL surfaces takes a moment
    timer = setTimeout(() => gds.forEach((gd) => Plotly.restyle(gd, { opacity: +opa.value })), 80);
  };
})();
</script>
</body>
</html>
"""

fmt = "%Y-%m-%d"
meta = (f"{len(raw):,} rows  &middot;  {days[0]:{fmt}} &rarr; {days[-1]:{fmt}} UTC  &middot;  {n_days} days  &middot;  "
        f"bins of {x_bin} ({aggregate} per bin)")
nav = " &middot; ".join(f'<a href="#surf_{i}">{html.escape(c)}</a>' for i, c in enumerate(surfaces))
page = (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__META__", meta).replace("__NAV__", nav)
        .replace("__OPACITY__", str(opacity)).replace("__DMIN__", y_dates[0]).replace("__DMAX__", y_dates[-1])
        .replace("__FIGS__", "\n".join(blocks)).replace("__IDS__", json.dumps(ids)).replace("__CAMERA__", json.dumps(CAMERA)))
# Plotly itself goes in a <script> in the head, before any figure
page = page.replace("</head>", "<script>" + get_plotlyjs() + "</script>\n</head>", 1)

out_path = Path(product["html"])
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(page, encoding="utf-8")
pd.DataFrame({"file": [str(out_path)], "size_MB": [round(out_path.stat().st_size / 1e6, 2)], "plots": [len(ids)]})
