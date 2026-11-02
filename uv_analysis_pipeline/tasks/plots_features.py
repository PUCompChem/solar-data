# %% [markdown]
# # 5b. Feature plots (HTML)
# One interactive page with the daily features of every channel (output of `features`).
# Per channel: daily maximum with 7/27-day means and anomalies, daily dose, and the anomaly z-score.
# Plotly is embedded, no internet needed. Zoom = drag, pan = Shift+drag, double-click = reset, range
# slider below each channel. The three panels of a channel share the time axis.
# Settings: `plots_features_config` in env.yaml.

# %% tags=["parameters"]
upstream = None
product = None
config = None

# %%
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

title = config.get("title", "UV measurements: daily features")
selected = config.get("channels")
panel_height = int(config.get("panel_height", 230))
z_limit = float(config.get("z_limit", 3.0))

feat = pd.read_csv(upstream["features"]["data"], index_col="date", parse_dates=True)
available = [c[: -len("_is_anomaly")] for c in feat.columns if c.endswith("_is_anomaly")]
channels = list(selected) if selected else available
unknown = [c for c in channels if c not in available]
if unknown:
    raise KeyError(f"Channel(s) {unknown} not in the features output; available: {available}")
print(f"{len(feat)} days, channels: {channels}")

# %% [markdown]
# ## One figure per channel

# %%
PLOT_CONFIG = {"displaylogo": False, "responsive": True}
COL = {"day": "#9aa7b5", "r7": "#2563eb", "r27": "#e0702b", "bad": "#d62728"}
ids, blocks, notes = [], [], {}

for c in channels:
    d = feat[[col for col in feat.columns if col.startswith(f"{c}_")]].rename(columns=lambda x: x[len(c) + 1:])
    n_valid = int(d["max"].notna().sum())
    anomaly = d["is_anomaly"].fillna(False).astype(bool)
    notes[c] = f"{n_valid} days with data &middot; {int(anomaly.sum())} anomaly days (|z| &gt; {z_limit:g})"
    div_id = f"feat_{len(ids)}"
    ids.append(div_id)
    if n_valid == 0:
        blocks.append(f'<section id="s_{div_id}"><h2>{html.escape(c)}</h2><div class="note">no day passes the coverage filter</div></section>')
        continue

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("daily maximum", "daily dose (value units x h)", f"z-score of the daily maximum vs. its 27-day mean"))
    t = d.index
    fig.add_trace(go.Scatter(x=t, y=d["max"], name="daily max", mode="lines", line=dict(width=1, color=COL["day"]),
                             legendgroup="max", hovertemplate="%{x|%Y-%m-%d}<br>max %{y:.4g}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=d["max_roll7"], name="7-day mean", mode="lines", line=dict(width=1.6, color=COL["r7"]),
                             hovertemplate="%{x|%Y-%m-%d}<br>7-day %{y:.4g}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=d["max_roll27"], name="27-day mean", mode="lines", line=dict(width=2.4, color=COL["r27"]),
                             hovertemplate="%{x|%Y-%m-%d}<br>27-day %{y:.4g}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t[anomaly], y=d["max"][anomaly], name=f"anomaly (|z|>{z_limit:g})", mode="markers",
                             marker=dict(size=8, color=COL["bad"]),
                             hovertemplate="%{x|%Y-%m-%d}<br>anomaly, max %{y:.4g}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=d["dose_unit_hours"], name="daily dose", mode="lines", line=dict(width=1, color=COL["day"]),
                             showlegend=False, hovertemplate="%{x|%Y-%m-%d}<br>dose %{y:.4g}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=d["dose_unit_hours_roll27"], name="dose, 27-day mean", mode="lines",
                             line=dict(width=2.4, color=COL["r27"]), showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d}<br>27-day %{y:.4g}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=d["max_zscore"], name="z-score", mode="lines", line=dict(width=1, color="#5b6775"),
                             showlegend=False, hovertemplate="%{x|%Y-%m-%d}<br>z %{y:.2f}<extra></extra>"), 3, 1)
    for lim in (z_limit, -z_limit):
        fig.add_hline(y=lim, line=dict(color=COL["bad"], width=1, dash="dash"), row=3, col=1)
    fig.update_layout(height=3 * panel_height + 90, margin=dict(l=60, r=20, t=40, b=10), paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", hovermode="x unified",
                      legend=dict(orientation="h", y=1.07, x=0, font=dict(size=11)))
    fig.update_xaxes(showgrid=True, gridcolor="#e8edf3")
    fig.update_yaxes(showgrid=True, gridcolor="#e8edf3", zeroline=False)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=3, col=1)
    blocks.append(f'<section id="s_{div_id}"><h2>{html.escape(c)}</h2><div class="note">{notes[c]}</div>'
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
section{padding:6px 20px 14px;border-bottom:1px solid #e3e8ef}
h2{font-size:16px;margin:10px 0 0}
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="meta">__META__</div></header>
<div id="bar">
  <span>Jump to: __NAV__</span>
  <span class="meta">drag = zoom &middot; Shift+drag = pan &middot; double-click = reset &middot; slider = time span &middot; hover = values</span>
</div>
__FIGS__
</body>
</html>
"""

meta = (f"{feat.index.min():%Y-%m-%d} &rarr; {feat.index.max():%Y-%m-%d} UTC &middot; {len(feat)} days &middot; "
        f"{len(channels)} channel(s)")
nav = " &middot; ".join(f'<a href="#s_{i}">{html.escape(c)}</a>' for i, c in zip(ids, channels))
page = (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__META__", meta).replace("__NAV__", nav)
        .replace("__FIGS__", "\n".join(blocks)))
page = page.replace("</head>", "<script>" + get_plotlyjs() + "</script>\n</head>", 1)

out_path = Path(product["html"])
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(page, encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
