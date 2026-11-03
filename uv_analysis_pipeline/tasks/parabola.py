# %% [markdown]
# # 7. Daily parabola fits
# The daytime profile of every day is approximated by a quadratic function of the time of day
# `y(t) = a*t^2 + b*t + c`, t in hours (UTC). From each parabola:
# - `t_max = -b / (2a)`   position of the maximum (hours, UTC)
# - `y_max = c - b^2 / (4a)`   value of the maximum
#
# Only the top part of the daily curve is fitted (where the smoothed signal is above
# `fit_fraction` of its way from the day's baseline to its peak). A cosine-like profile is well described by
# a parabola there, but not over the whole day.
#
# Clouds only LOWER the UV. The fit therefore follows the UPPER ENVELOPE of the data (the cloud-free curve): points far
# below the parabola (`reject_sigma` x the noise above it) are dropped and the parabola is refitted. On a day with broken
# clouds the maximum is thus the cloud-free maximum; `clear_fraction` is the share of points kept (a cloudiness measure).
#
# Every day gets a `status`, because cloudy, truncated or gappy days damage the fit. `t_max` and `y_max` are
# filled ONLY for days with status `ok`; for the others the observed (smoothed) peak is given as a fallback
# in `peak_obs` / `t_peak_obs`. Statuses, checked in this order:
# - `no_data`            too few samples in the day
# - `low_activity`       the day's amplitude is below `low_peak_frac` of the typical clear-day amplitude
#                        (rolling 31-day 90th percentile) or below `min_peak`: heavy clouds / very low UV
# - `too_few_points`     fewer than 10 points in the top part, or a fitted window shorter than `min_fit_hours`
# - `gap_in_window`      missing data (a hole longer than `max_gap_hours` between samples) inside the fitted window
# - `cloudy`            fewer than `min_clear_fraction` of the points are cloud-free: too little to trust the envelope
# - `not_concave`        a >= 0, the parabola has no maximum
# - `peak_outside_window`  the maximum lies outside the fitted data (extrapolation)
# - `poor_fit`           R^2 < `min_r2` (partly cloudy day, the parabola does not describe the data)
# - `ok`
# `perfect` (True/False) marks the `ok` days that are almost cloud-free and fit almost exactly:
# `clear_fraction` >= `perfect_min_clear_fraction`, `r2` >= `perfect_min_r2` and rmse / y_max <= `perfect_max_rel_rmse`.
# Output: `data`: one wide CSV, one row per day, columns `<channel>_<field>`.
#         `summary`: one row per channel and quality class (perfect, ok, poor_fit, cloudy, ...): number and share of days
#         and mean / median / std / min / max of the main parameters. The classes do not overlap: `ok` here means
#         "ok but not perfect". Settings: `parabola_config` in env.yaml.

# %% tags=["parameters"]
upstream = None          # set in pipeline.yaml from `data_source`: "clean" or "load" (raw)
product = None
config = None            # parabola_config from env.yaml
sentinel_values = []

# %%
import numpy as np
import pandas as pd

selected = config.get("channels")
exclude = config.get("exclude_channels") or []
mask_sentinels = bool(config.get("mask_sentinels", True))
smooth_window = int(config.get("smooth_window", 11))
fit_fraction = float(config.get("fit_fraction", 0.5))
reject_sigma = config.get("reject_sigma", 3.0)
min_samples = int(config.get("min_samples", 30))
min_fit_hours = float(config.get("min_fit_hours", 2.0))
max_gap_hours = float(config.get("max_gap_hours", 0.5))
min_r2 = float(config.get("min_r2", 0.9))
min_clear_fraction = float(config.get("min_clear_fraction", 0.5))
low_peak_frac = float(config.get("low_peak_frac", 0.3))
min_peak = config.get("min_peak")
# "perfect" parabolas: status ok AND all of these (stricter than the status checks)
perfect_min_clear_fraction = float(config.get("perfect_min_clear_fraction", 0.95))
perfect_min_r2 = float(config.get("perfect_min_r2", 0.99))
perfect_max_rel_rmse = config.get("perfect_max_rel_rmse", 0.02)    # rmse / y_max; null = not used
if not 0 <= fit_fraction < 1:
    raise ValueError(f"fit_fraction must be in [0, 1) (got {fit_fraction})")
if smooth_window < 1:
    raise ValueError(f"smooth_window must be >= 1 (got {smooth_window})")

source = "clean" if "clean" in upstream else "load"
print(f"Input: {'cleaned' if source == 'clean' else 'RAW (uncleaned)'} data from task '{source}'")
df = pd.read_csv(upstream[source]["data"], index_col="timestamp", parse_dates=True)
available = [c for c in df.columns if c not in ("time_raw", "source_file", "was_filled")]
channels = [c for c in (list(selected) if selected else available) if c not in exclude]
unknown = [c for c in channels if c not in available]
if unknown:
    raise KeyError(f"Channel(s) {unknown} not in the '{source}' output; available: {available}")
if not channels:
    raise ValueError("No channel left to fit (check channels / exclude_channels in parabola_config)")
print(f"Channels: {channels}" + (f"  (excluded: {list(exclude)})" if exclude else ""))

# %% [markdown]
# ## Fit one day

# %%
def fit_day(t, y):
    """t: hours of day, y: values (finite). Returns (dict of results, status or None)."""
    res = dict(n_points=0, clear_fraction=np.nan, window_start=np.nan, window_end=np.nan, a=np.nan, b=np.nan, c=np.nan,
               r2=np.nan, rmse=np.nan, t_max=np.nan, y_max=np.nan, peak_obs=np.nan, t_peak_obs=np.nan, amp=np.nan)
    if len(t) < min_samples:
        return res, "no_data"
    s = pd.Series(y).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
    k = int(np.argmax(s))
    base = float(np.percentile(s, 5))                 # night baseline (0 for UV, ~330 for the IR channel)
    res.update(peak_obs=float(s[k]), t_peak_obs=float(t[k]), amp=float(s[k] - base))
    if res["amp"] <= 0:
        return res, None                              # judged later as low_activity
    sel = s >= base + fit_fraction * res["amp"]
    tt, yy = t[sel], y[sel]
    if len(tt) < 10:
        return res, "too_few_points"
    in_win = t[(t >= tt.min()) & (t <= tt.max())]     # all samples inside the window, not only the selected ones,
    max_gap = float(np.max(np.diff(np.sort(in_win)))) if len(in_win) > 1 else 0.0   # so cloud dips are not counted as gaps
    t0 = tt.mean()
    x = tt - t0
    keep = np.ones(len(x), dtype=bool)
    coef = np.polyfit(x, yy, 2)
    if reject_sigma:
        # Clouds only LOWER the UV, so the parabola should follow the upper envelope (the cloud-free curve):
        # fit, drop points far below it (or far above it: spikes), refit. Noise scale from the points above the curve.
        for _ in range(8):
            r = yy - np.polyval(coef, x)
            pos = r[keep & (r > 0)]
            if len(pos) < 5:
                break
            sig = float(np.sqrt(np.mean(pos ** 2)))
            new_keep = np.abs(r) <= float(reject_sigma) * sig
            if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
                break
            keep = new_keep
            coef = np.polyfit(x[keep], yy[keep], 2)
    n_selected = len(x)
    clear_fraction = float(keep.sum() / n_selected)
    x, yy, tt = x[keep], yy[keep], tt[keep]
    A, B, C = coef                                    # in x = t - t0
    fit = np.polyval(coef, x)
    ss_res, ss_tot = float(np.sum((yy - fit) ** 2)), float(np.sum((yy - yy.mean()) ** 2))
    res.update(n_points=int(len(tt)), clear_fraction=clear_fraction, window_start=float(tt.min()), window_end=float(tt.max()),
               a=float(A), b=float(B - 2 * A * t0), c=float(C - B * t0 + A * t0 ** 2),   # coefficients in absolute t
               r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan, rmse=float(np.sqrt(ss_res / len(tt))))
    if max_gap > max_gap_hours:
        return res, "gap_in_window"
    if tt.max() - tt.min() < min_fit_hours:
        return res, "too_few_points"
    if clear_fraction < min_clear_fraction:
        return res, "cloudy"
    if A >= 0:
        return res, "not_concave"
    t_max = t0 - B / (2 * A)
    y_max = C - B ** 2 / (4 * A)
    res.update(t_max=float(t_max), y_max=float(y_max))
    if not (tt.min() <= t_max <= tt.max()):
        return res, "peak_outside_window"
    if not (res["r2"] >= min_r2):
        return res, "poor_fit"
    return res, "ok"

# %% [markdown]
# ## Fit all days of every channel

# %%
def fit_channel(series):
    v = series.astype("float64")
    if mask_sentinels and source == "load" and sentinel_values:
        v = v.mask(v.isin(sentinel_values))
    v = v.dropna()
    v = v.groupby(level=0).mean().sort_index()        # one value per timestamp
    hours = (v.index.hour + v.index.minute / 60 + v.index.second / 3600).to_numpy()
    days = v.index.normalize()
    rows = {}
    for day, pos in v.groupby(days).indices.items():
        res, status = fit_day(hours[pos], v.to_numpy()[pos])
        res["status"] = status
        rows[day] = res
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index().asfreq("1D")
    # judge amplitude against the typical clear-day amplitude of the surrounding month
    ref = out["amp"].rolling(31, center=True, min_periods=7).quantile(0.9)
    low = out["amp"] < low_peak_frac * ref
    if min_peak is not None:
        low |= out["peak_obs"] < float(min_peak)
    low |= out["amp"] <= 0
    st = out["status"].mask(low & (out["status"] != "no_data"), "low_activity")
    out["status"] = st.fillna("no_data")              # days without any sample (and amp <= 0 days are already low_activity)
    # t_max / y_max only for good parabolas
    bad = out["status"] != "ok"
    out.loc[bad, ["t_max", "y_max"]] = np.nan
    # "perfect": a good parabola on an almost cloud-free day
    perfect = (out["status"] == "ok") & (out["clear_fraction"] >= perfect_min_clear_fraction) & (out["r2"] >= perfect_min_r2)
    if perfect_max_rel_rmse is not None:
        perfect &= (out["rmse"] / out["y_max"]) <= float(perfect_max_rel_rmse)
    out["perfect"] = perfect
    return out.drop(columns="amp")

per_channel = {c: fit_channel(df[c]) for c in channels}
counts = pd.DataFrame({c: d["status"].value_counts() for c, d in per_channel.items()}).fillna(0).astype(int)
counts.loc["  of which perfect"] = [int(d["perfect"].sum()) for d in per_channel.values()]
counts.index.name = "status"
counts

# %% [markdown]
# ## Write the result

# %%
wide = pd.concat(per_channel, axis=1)
wide.columns = [f"{c}_{f}" for c, f in wide.columns]
wide.index.name = "date"
wide.to_csv(product["data"])
print(f"Wrote {wide.shape[0]} days x {wide.shape[1]} columns")

# %% [markdown]
# ## Summary by quality
# One row per channel and quality class. `perfect` = status ok and the `perfect_*` criteria; `ok` = the remaining ok days;
# the other classes are the statuses. t_max and y_max exist only for `perfect` and `ok` days; for the other classes use
# `peak_obs` (observed smoothed peak). Statistics skip empty values.

# %%
QUALITY = ["perfect", "ok", "poor_fit", "cloudy", "not_concave", "peak_outside_window",
           "gap_in_window", "too_few_points", "low_activity", "no_data"]
STAT_FIELDS = ["t_max", "y_max", "a", "r2", "clear_fraction", "peak_obs"]
rows = []
for c, d in per_channel.items():
    quality = d["status"].where(~d["perfect"], "perfect")
    for name in QUALITY + sorted(set(quality) - set(QUALITY)):      # any unforeseen label still gets a row
        sub = d[quality == name]
        row = {"channel": c, "quality": name, "n_days": len(sub), "pct_days": round(100 * len(sub) / len(d), 1)}
        for f in STAT_FIELDS:
            x = sub[f]
            row.update({f"{f}_mean": x.mean(), f"{f}_median": x.median(), f"{f}_std": x.std(),
                        f"{f}_min": x.min(), f"{f}_max": x.max()})
        rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(product["summary"], index=False)
summary[["channel", "quality", "n_days", "pct_days", "t_max_mean", "t_max_std", "y_max_mean", "r2_mean", "clear_fraction_mean"]]