# %% [markdown]
# # 5. Daily features (all channels)
# For every channel: daily mean/max/p95/dose, coverage filter, rolling means (7 d and 27 d solar rotation),
# anomaly z-score. One wide CSV: one row per day, columns named `<channel>_<feature>` (e.g. `UVA_max`).
# Settings: `features_config` in env.yaml. The plots are made by the next task, `plots_features`.

# %% tags=["parameters"]
upstream = None          # set in pipeline.yaml from `data_source`: "clean" or "load" (raw)
product = None
config = None            # features_config from env.yaml
sentinel_values = []
cadence = "1min"
min_daily_coverage = 0.6

# %%
import numpy as np
import pandas as pd

selected = config.get("channels")
mask_sentinels = bool(config.get("mask_sentinels", True))

source = "clean" if "clean" in upstream else "load"
print(f"Input: {'cleaned' if source == 'clean' else 'RAW (uncleaned)'} data from task '{source}'")
df = pd.read_csv(upstream[source]["data"], index_col="timestamp", parse_dates=True)
available = [c for c in df.columns if c not in ("time_raw", "source_file", "was_filled")]
channels = list(selected) if selected else available
unknown = [c for c in channels if c not in available]
if unknown:
    raise KeyError(f"Channel(s) {unknown} not in the '{source}' output; available: {available}"
                   + ("  (the cleaned data holds only the cleaned channel; use data_source: load for all columns)"
                      if source == "clean" else ""))
step_s = pd.Timedelta(cadence).total_seconds()
samples_per_day = 86400 / step_s
print(f"Channels: {channels}")

# %% [markdown]
# ## Features per channel

# %%
def features_for(series):
    v = series.astype("float64")
    if mask_sentinels and source == "load" and sentinel_values:
        v = v.mask(v.isin(sentinel_values))
    v = v.groupby(level=0).mean().sort_index()        # one value per timestamp (raw data may repeat timestamps)
    g = v.resample("1D")
    d = pd.DataFrame({
        "mean": g.mean(),
        "max": g.max(),
        "p95": g.quantile(0.95),
        "dose_unit_hours": g.sum() * step_s / 3600,   # integral over the day: (value units) * hours
        "coverage": g.count() / samples_per_day,
    })
    before = len(d)
    d = d[d["coverage"] >= min_daily_coverage]
    kept = len(d)
    d = d.asfreq("1D")                                # reinstate dropped days explicitly as NaN rows
    for col in ["max", "dose_unit_hours"]:
        d[f"{col}_roll7"] = d[col].rolling(7, center=True, min_periods=4).mean()
        d[f"{col}_roll27"] = d[col].rolling(27, center=True, min_periods=14).mean()
    m = d["max"].rolling(27, center=True, min_periods=14).mean()
    sd = d["max"].rolling(27, center=True, min_periods=14).std()
    d["max_zscore"] = (d["max"] - m) / sd
    d["is_anomaly"] = d["max_zscore"].abs() > 3
    return d, before, kept

per_channel, rows = {}, []
for c in channels:
    d, before, kept = features_for(df[c])
    per_channel[c] = d
    rows.append({"channel": c, "days": before, "days_kept": kept, "anomaly_days": int(d["is_anomaly"].sum())})
pd.DataFrame(rows).set_index("channel")

# %%
wide = pd.concat({c: d for c, d in per_channel.items()}, axis=1)
wide.columns = [f"{c}_{f}" for c, f in wide.columns]
wide.index.name = "date"
wide.to_csv(product["data"])
print(f"Wrote {wide.shape[0]} days x {wide.shape[1]} columns")
