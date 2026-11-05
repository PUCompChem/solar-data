# %% [markdown]
# # 7c. Daily maxima by parabola quality (HTML)
# One interactive page: the daily maximum of every channel as points, coloured by the quality class of the
# day's parabola fit (output of `parabola`). Filters: class check boxes (default: perfect + lightly cloudy),
# optional connecting lines (default off), point size, and an optional approximation curve (harmonic annual
# series, clear-sky cos^k model or local smoother) fitted in the browser to the points currently shown. IR is skipped. Plotly is embedded, no internet needed.
# Settings: `plots_max_config` in env.yaml.

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

title = config.get("title", "UV measurements: daily maxima by parabola quality")
selected = config.get("channels")
max_exclude = list(config.get("exclude_channels") or [])
max_default = list(config.get("default_classes") or ["perfect", "lightly_cloudy"])
max_point_size = float(config.get("point_size", 7))
max_lines = bool(config.get("lines", False))
max_panel_height = int(config.get("panel_height", 260))
fit_cfg = dict(config.get("fit") or {})
FIT_ALL = ["harmonic", "clearsky", "smooth"]
fit_methods = list(fit_cfg.get("methods", FIT_ALL))
fit_default = fit_cfg.get("default")
fit_default = "none" if fit_default is None or str(fit_default).lower() == "none" else str(fit_default)   # an unquoted YAML none may arrive as None
if any(m not in FIT_ALL for m in fit_methods):
    raise ValueError(f"fit.methods must be a subset of {FIT_ALL}, got {fit_methods}")
if fit_default != "none" and fit_default not in fit_methods:
    raise ValueError(f"fit.default must be 'none' or one of fit.methods {fit_methods}, got {fit_default!r}")
if fit_cfg.get("use", "visible") not in ("visible", "perfect"):
    raise ValueError("fit.use must be 'visible' or 'perfect'")
FIT = {"harmonics": int(fit_cfg.get("harmonics", 2)), "latitude": float(fit_cfg.get("latitude", 41.7)),
       "smooth_days": float(fit_cfg.get("smooth_days", 30)), "use": fit_cfg.get("use", "visible"),
       "reject_sigma": None if fit_cfg.get("reject_sigma", 2.0) is None else float(fit_cfg.get("reject_sigma", 2.0)),
       "width": float(fit_cfg.get("width", 2.5)), "perfect_color": "#0b6e4f"}
fit_show_perfect = bool(fit_cfg.get("show_perfect", False))

feat = pd.read_csv(upstream["features"]["data"], index_col="date", parse_dates=True)
channels = [c[: -len("_max")] for c in feat.columns if c.endswith("_max")]
channels = [c for c in channels if f"{c}_is_anomaly" in feat.columns]
if selected:
    unknown = [c for c in selected if c not in channels]
    if unknown:
        raise KeyError(f"Channel(s) {unknown} not in the features output; available: {channels}")
    channels = list(selected)
PLOT_CONFIG = {"displaylogo": False, "responsive": True}

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
/* range-slider preview: lighter, thinner, semi-transparent */
.rangeslider-rangeplot .scatterlayer .trace{opacity:.35}
.rangeslider-rangeplot .scatterlayer .js-line{stroke-width:.8px!important;stroke:#7b8794!important}
.rangeslider-rangeplot .scatterlayer .point{d:path("M1.6,0A1.6,1.6 0 1,1 -1.6,0A1.6,1.6 0 1,1 1.6,0Z")}
.eq{font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;color:#1d2733}
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

par = pd.read_csv(upstream["parabola"]["data"], index_col="date", parse_dates=True)
par_channels = [c[: -len("_quality")] for c in par.columns if c.endswith("_quality")]
max_channels = [c for c in channels if c in par_channels and c not in max_exclude]
print(f"Maxima page channels: {max_channels} (skipped: {[c for c in channels if c not in max_channels]})")

series, present = {}, set()
for c in max_channels:
    mx = feat[f"{c}_max"].dropna()
    q = par[f"{c}_quality"].reindex(mx.index).fillna("no_data")
    r2 = par[f"{c}_r2"].reindex(mx.index) if f"{c}_r2" in par.columns else pd.Series(index=mx.index, dtype=float)
    tm = par[f"{c}_t_max"].reindex(mx.index) if f"{c}_t_max" in par.columns else pd.Series(index=mx.index, dtype=float)
    series[c] = (mx, q, r2, tm)
    present |= set(q)
cls_list = [(k, lab, col) for k, lab, col in CLASSES if k in present]
cls_list += [(k, k.replace("_", " "), "#9aa5b1") for k in sorted(present - known)]
counts = {k: 0 for k, _, _ in cls_list}

mblocks, mids = [], []
for c in max_channels:
    mx, q, r2, tm = series[c]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="connecting line", line=dict(width=1, color="#9aa7b5"),
                             hoverinfo="skip", showlegend=False, visible=max_lines, meta="__line__"))
    for k, lab, col in cls_list:
        m = (q == k).to_numpy()
        counts[k] += int(m.sum())
        txt = [f"{d:%Y-%m-%d}<br>{lab}<br>max {v:.4g}"
               + ("" if pd.isna(a) else f"<br>R&sup2; {a:.4f}") + ("" if pd.isna(b) else f"<br>t_max {b:.2f} h UTC")
               for d, v, a, b in zip(mx.index[m], mx[m], r2[m], tm[m])]
        fig.add_trace(go.Scatter(x=[f"{d:%Y-%m-%d}" for d in mx.index[m]], y=[float(v) for v in mx[m]], mode="markers", name=lab, meta=k, text=txt, hoverinfo="text",
                                 marker=dict(size=max_point_size, color=col, line=dict(width=0.5, color="#ffffff")),
                                 visible=k in max_default, showlegend=False))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="fit", meta="__fit__", showlegend=False, visible=False,
                             line=dict(width=FIT["width"], color="#111827"),
                             hovertemplate="%{x|%Y-%m-%d}<br>fit %{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="fit, perfect only", meta="__fitp__", showlegend=False, visible=False,
                             line=dict(width=FIT["width"], color=FIT["perfect_color"], dash="dash"),
                             hovertemplate="%{x|%Y-%m-%d}<br>fit (perfect only) %{y:.4g}<extra></extra>"))
    fig.update_layout(height=max_panel_height + 60, margin=dict(l=60, r=20, t=10, b=10), paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="#e8edf3", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(showgrid=True, gridcolor="#e8edf3", zeroline=False, title="daily max")
    div_id = f"max_{len(mids)}"
    mids.append(div_id)
    mblocks.append(f'<section id="s_{div_id}"><h2>{html.escape(c)}</h2><div class="note">{len(mx)} days with a daily maximum &middot; <span id="fi_{div_id}"></span><br><span id="fe_{div_id}" class="eq"></span></div>'
                   + fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config=PLOT_CONFIG) + "</section>")

MAX_TEMPLATE = TEMPLATE.replace(
    '<span class="meta">drag = zoom',
    '<span id="ctl">__CTL__</span>\n  <span class="meta">drag = zoom', 1).replace(
    "</body>", """<script>
const IDS = __IDS__, FIT = __FIT__;
const DAY = 86400000, OM = 2*Math.PI/365.25;
function visibleKeys(){return [...document.querySelectorAll('.cls:checked')].map(e=>e.value);}
function dnum(s){return Date.parse(s+'T00:00:00Z')/DAY;}
function dstr(d){return new Date(d*DAY).toISOString().slice(0,10);}
function solve(A,b){const n=b.length;for(let i=0;i<n;i++){let p=i;for(let r=i+1;r<n;r++)if(Math.abs(A[r][i])>Math.abs(A[p][i]))p=r;
  [A[i],A[p]]=[A[p],A[i]];[b[i],b[p]]=[b[p],b[i]];if(Math.abs(A[i][i])<1e-12)return null;
  for(let r=i+1;r<n;r++){const f=A[r][i]/A[i][i];for(let c=i;c<n;c++)A[r][c]-=f*A[i][c];b[r]-=f*b[i];}}
  const x=new Array(n);for(let i=n-1;i>=0;i--){let v=b[i];for(let c=i+1;c<n;c++)v-=A[i][c]*x[c];x[i]=v/A[i][i];}return x;}
function wls(rows,y,w){const m=rows[0].length,A=Array.from({length:m},()=>new Array(m).fill(0)),b=new Array(m).fill(0);
  rows.forEach((r,i)=>{const wi=w?w[i]:1;for(let j=0;j<m;j++){b[j]+=wi*r[j]*y[i];for(let k=0;k<m;k++)A[j][k]+=wi*r[j]*r[k];}});return solve(A,b);}
function harmRow(t,H){const r=[1];for(let k=1;k<=H;k++){r.push(Math.cos(k*OM*t),Math.sin(k*OM*t));}return r;}
function cosZen(t){const d=new Date(t*DAY),y0=Date.UTC(d.getUTCFullYear(),0,1)/DAY,doy=t-y0+1;
  const dec=-23.44*Math.PI/180*Math.cos(2*Math.PI*(doy+10)/365),la=FIT.latitude*Math.PI/180;
  return Math.max(Math.sin(la)*Math.sin(dec)+Math.cos(la)*Math.cos(dec),0);}
// every method returns a predictor t -> value (or null when the fit is impossible)
function fitMethod(m,t,y){
  if(m==='harmonic'){const H=FIT.harmonics;if(t.length<2*H+1)return null;
    const c=wls(t.map(v=>harmRow(v,H)),y);if(!c)return null;
    const f=v=>harmRow(v,H).reduce((s,r,i)=>s+r*c[i],0),g=x=>(+x).toPrecision(4),sg=x=>(x<0?' &minus; ':' + ')+g(Math.abs(x));
    let eq='y = '+g(c[0]);for(let k=1;k<=H;k++)eq+=sg(c[2*k-1])+'&middot;cos('+(k>1?k:'')+'&omega;t)'+sg(c[2*k])+'&middot;sin('+(k>1?k:'')+'&omega;t)';
    eq+='<br>&omega; = 2&pi;/365.25 d, t = days since 1970-01-01';
    for(let k=1;k<=H;k++){const am=Math.hypot(c[2*k-1],c[2*k]);let tp=Math.atan2(c[2*k],c[2*k-1])/(k*OM);const per=365.25/k;
      const tref=Math.min(...t);tp=tref+(((tp-tref)%per)+per)%per;
      eq+='<br>harmonic '+k+': amplitude '+g(am)+(k===1?', annual maximum on '+dstr(Math.round(tp)).slice(5)+' (month-day)':'');}
    f.eq=eq;return f;}
  if(m==='clearsky'){if(t.length<3)return null;const f=t.map(cosZen);let best=null;
    for(let k=0.2;k<=8;k+=0.05){let sf=0,sff=0,sy=0;const g=f.map(v=>Math.pow(v,k));
      g.forEach((v,i)=>{sf+=v*y[i];sff+=v*v;});if(sff<=0)continue;const A=sf/sff;let e=0;g.forEach((v,i)=>{e+=(y[i]-A*v)**2;});
      if(!best||e<best.e)best={e,A,k};}
    if(!best)return null;const fn=v=>best.A*Math.pow(cosZen(v),best.k);
    fn.eq='y = A&middot;cos(&theta;)<sup>k</sup> &nbsp; A = '+best.A.toPrecision(4)+', k = '+best.k.toFixed(2)
      +'<br>cos(&theta;) = sin&phi;&middot;sin&delta; + cos&phi;&middot;cos&delta; (solar noon), &phi; = '+FIT.latitude+'&deg;, &delta; = solar declination of the day';return fn;}
  if(m==='smooth'){const h=+document.getElementById('sw').value;const f=v=>{const w=[],rows=[],yy=[];
    t.forEach((tt,i)=>{const u=Math.abs(tt-v)/h;if(u<1){w.push((1-u**3)**3);rows.push([1,tt-v]);yy.push(y[i]);}});
    if(rows.length<4)return null;const c=wls(rows,yy,w);return c?c[0]:null;};
    f.eq='local linear regression, tricube weights, window half-width '+h+' d (no closed-form equation)';return f;}
  return null;}
function robustFit(m,t,y){
  let keep=t.map(()=>true),pred=null,rejected=0;
  for(let it=0;it<6;it++){
    const tt=t.filter((_,i)=>keep[i]),yy=y.filter((_,i)=>keep[i]);
    pred=fitMethod(m,tt,yy);if(!pred||FIT.reject_sigma===null)break;
    const r=t.map((v,i)=>{const p=pred(v);return p===null?NaN:y[i]-p;});
    const rk=r.filter((v,i)=>keep[i]&&!isNaN(v)).sort((a,b)=>a-b);if(rk.length<5)break;
    const med=rk[Math.floor(rk.length/2)],mad=rk.map(v=>Math.abs(v-med)).sort((a,b)=>a-b)[Math.floor(rk.length/2)];
    const sg=Math.max(1.4826*mad,1e-9);
    const nk=t.map((_,i)=>keep[i]&&!(r[i]<-FIT.reject_sigma*sg));   // only low points (clouds) are dropped
    if(nk.every((v,i)=>v===keep[i]))break;keep=nk;}
  rejected=keep.filter(v=>!v).length;
  const used=t.map((v,i)=>keep[i]?[v,y[i]]:null).filter(Boolean);
  return {pred,used,rejected};}
function fitCurve(gd,fitKeys){
  // fits the selected method to the points of the classes in fitKeys; returns {x,y,info,eq} or {msg}
  const m=document.getElementById('fitm').value;
  const t=[],y=[];gd.data.forEach(tr=>{if(fitKeys.includes(tr.meta))for(let j=0;j<tr.x.length;j++){t.push(dnum(tr.x[j]));y.push(tr.y[j]);}});
  const all=[];gd.data.forEach(tr=>{if(tr.meta&&!tr.meta.startsWith('__'))tr.x.forEach(x=>all.push(dnum(x)));});
  if(m==='none'||!all.length)return null;
  if(t.length<3)return {msg:'fit '+m+': not enough points ('+t.length+')'};
  const r=robustFit(m,t,y);
  if(!r.pred)return {msg:'fit '+m+': not enough points ('+t.length+')'};
  const t0=Math.min(...all),t1=Math.max(...all),gx=[],gy=[];
  for(let d=t0;d<=t1;d+=1){const v=r.pred(d);if(v!==null&&isFinite(v)){gx.push(dstr(d));gy.push(v);}}
  const u=r.used,mean=u.reduce((s,p)=>s+p[1],0)/u.length;let se=0,st=0;
  u.forEach(p=>{const v=r.pred(p[0]);se+=(p[1]-v)**2;st+=(p[1]-mean)**2;});
  const span=Math.max(...t)-Math.min(...t);
  const info='fit '+m+': '+u.length+' points used, '+r.rejected+' rejected, R&sup2; '+(st>0?(1-se/st).toFixed(3):'n/a')+', rmse '+Math.sqrt(se/u.length).toPrecision(3)
    +((m==='harmonic'||m==='clearsky')&&span<300?' &middot; <b style="color:#b45309">record shorter than one year, fit unreliable</b>':'');
  return {x:gx,y:gy,info:info,eq:r.pred.eq||''};}
function refresh(){
  const keys = visibleKeys(), lines = document.getElementById('lines').checked;
  const size = +document.getElementById('psize').value, fw = +document.getElementById('fw').value;
  document.getElementById('psv').textContent = size;
  document.getElementById('fwv').textContent = fw;
  document.getElementById('swv').textContent = document.getElementById('sw').value;
  document.getElementById('swl').style.display = document.getElementById('fitm').value==='smooth' ? '' : 'none';
  for (const id of IDS){
    const gd = document.getElementById(id); if(!gd || !gd.data) continue;
    const pts = [];
    gd.data.forEach(t=>{ if(keys.includes(t.meta)) for(let j=0;j<t.x.length;j++) pts.push([+new Date(t.x[j]), t.x[j], t.y[j]]); });
    pts.sort((a,b)=>a[0]-b[0]);
    const fc = fitCurve(gd, FIT.use==='perfect' ? ['perfect'] : keys);
    const pc = document.getElementById('fitp').checked ? fitCurve(gd, ['perfect']) : null;   // comparison curve: perfect class only
    const ok = c => !!(c && c.x), okp = ok(pc);
    const PC = '<span style="color:'+FIT.perfect_color+'">&#9473;</span> perfect only: ';
    document.getElementById('fi_'+id).innerHTML = (fc ? (fc.info||fc.msg) : '') + (pc ? '<br>'+PC+(pc.info||pc.msg) : '');
    document.getElementById('fe_'+id).innerHTML = (fc&&fc.eq ? fc.eq : '') + (pc&&pc.eq ? '<br>'+PC+'<br>'+pc.eq : '');
    const vis = gd.data.map(t=> t.meta==='__line__' ? lines : t.meta==='__fit__' ? ok(fc) : t.meta==='__fitp__' ? okp : keys.includes(t.meta));
    const li = gd.data.findIndex(t=>t.meta==='__line__'), fi = gd.data.findIndex(t=>t.meta==='__fit__'), pi = gd.data.findIndex(t=>t.meta==='__fitp__');
    Plotly.restyle(gd, {visible: vis});
    Plotly.restyle(gd, {x:[pts.map(p=>p[1])], y:[pts.map(p=>p[2])]}, [li]);
    if(ok(fc)) Plotly.restyle(gd, {x:[fc.x], y:[fc.y], 'line.width': fw}, [fi]);
    if(okp) Plotly.restyle(gd, {x:[pc.x], y:[pc.y], 'line.width': fw}, [pi]);
    gd.data.forEach((t,i)=>{ if(!t.meta.startsWith('__')) Plotly.restyle(gd, {'marker.size': size}, [i]); });
  }
}
function setAll(on){document.querySelectorAll('.cls').forEach(e=>e.checked=on); refresh();}
window.addEventListener('load', ()=>{
  document.querySelectorAll('.cls,#lines,#fitm,#fitp').forEach(e=>e.addEventListener('change',refresh));
  document.getElementById('psize').addEventListener('input',refresh);
  document.getElementById('fw').addEventListener('input',refresh);
  document.getElementById('sw').addEventListener('input',refresh);
  document.getElementById('all').onclick=()=>setAll(true);
  document.getElementById('none').onclick=()=>setAll(false);
  refresh();
});
</script>
</body>""", 1)

ctl = " ".join(
    f'<label style="white-space:nowrap"><input type="checkbox" class="cls" value="{k}"{" checked" if k in max_default else ""}> '
    f'<span style="color:{col}">&#9679;</span> {lab} ({counts[k]})</label>' for k, lab, col in cls_list)
ctl += (' <button id="all" type="button">all</button> <button id="none" type="button">none</button>'
        f' &nbsp; <label><input type="checkbox" id="lines"{" checked" if max_lines else ""}> connect with lines</label>'
        f' &nbsp; <label>point size <input type="range" id="psize" min="2" max="20" step="1" value="{max_point_size:g}"> '
        '<span id="psv"></span></label>')
FIT_LABELS = {"harmonic": "harmonic annual series", "clearsky": "clear-sky cos^k", "smooth": "local smoother"}
if fit_methods:
    opts = '<option value="none">none</option>' + "".join(
        f'<option value="{m}"{" selected" if m == fit_default else ""}>{FIT_LABELS[m]}</option>' for m in fit_methods)
    if fit_default == "none":
        opts = opts.replace('<option value="none">', '<option value="none" selected>', 1)
    ctl += (f' &nbsp; <label title="same method, fitted only to the points of class perfect (dashed)">'
            f'<input type="checkbox" id="fitp"{" checked" if fit_show_perfect else ""}> perfect-only curve</label>')
    ctl += (f' &nbsp; <label>fit curve <select id="fitm">{opts}</select></label>'
            f' <label>width <input type="range" id="fw" min="1" max="6" step="0.5" value="{FIT["width"]:g}"> <span id="fwv"></span></label>'
            f' <label id="swl">smoothing half-width <input type="range" id="sw" min="5" max="120" step="1" value="{FIT["smooth_days"]:g}"> '
            '<span id="swv"></span> d</label>')
else:
    ctl += (' <input type="checkbox" id="fitp" hidden><select id="fitm" hidden><option value="none">none</option></select>'
            f'<input type="range" id="fw" hidden value="{FIT["width"]:g}"><span id="fwv" hidden></span>'
            f'<span id="swl" hidden><input type="range" id="sw" min="5" max="120" value="{FIT["smooth_days"]:g}"><span id="swv"></span></span>')
mnav = " &middot; ".join(f'<a href="#s_{i}">{html.escape(c)}</a>' for i, c in zip(mids, max_channels))
mmeta = (f"{feat.index.min():%Y-%m-%d} &rarr; {feat.index.max():%Y-%m-%d} UTC &middot; daily maxima coloured by parabola quality "
         f"&middot; channel(s): {', '.join(max_channels)}")
mpage = (MAX_TEMPLATE.replace("__TITLE__", html.escape(title + " (daily maxima)")).replace("__META__", mmeta)
         .replace("__NAV__", mnav).replace("__CTL__", ctl).replace("__IDS__", str(mids).replace("'", '"')).replace("__FIT__", json.dumps(FIT))
         .replace("__FIGS__", "\n".join(mblocks)))
mpage = mpage.replace("</head>", "<script>" + get_plotlyjs() + "</script>\n</head>", 1)
mout = Path(product["html"])
mout.parent.mkdir(parents=True, exist_ok=True)
mout.write_text(mpage, encoding="utf-8")
print(f"Wrote {mout} ({mout.stat().st_size / 1e6:.1f} MB)")
