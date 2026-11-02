# %% [markdown]
# # 6. Analysis
# - Diurnal profile and seasonal cycle
# - Lomb-Scargle periodogram (tolerates gaps) to look for ~27-day solar-rotation modulation
# - Summary JSON for downstream use

# %% tags=["parameters"]
upstream = None          # set in pipeline.yaml: ["clean" or "load", "features"]
product = None
value_column = "uv"
value_units = "W/m^2"
period_min_days = 5
period_max_days = 120

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, lombscargle

source = "clean" if "clean" in upstream else "load"
print(f"Input: {'cleaned' if source == 'clean' else 'RAW (uncleaned)'} data from task '{source}'")
fine = pd.read_csv(upstream[source]["data"], index_col="timestamp", parse_dates=True)[value_column]
fine = fine.groupby(level=0).mean().sort_index()   # one value per timestamp (raw data may repeat timestamps)
daily = pd.read_csv(upstream["features"]["data"], index_col="date", parse_dates=True)
# features are stored as <channel>_<feature>; keep the columns of `value_column` under their plain names
pref = f"{value_column}_"
daily = daily[[c for c in daily.columns if c.startswith(pref)]].rename(columns=lambda c: c[len(pref):])

# %% [markdown]
# ## Diurnal profile and seasonal cycle

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
fine.groupby(fine.index.hour).mean().plot(ax=axes[0], marker="o")
axes[0].set(title="Mean diurnal profile", xlabel="hour", ylabel=value_units)
daily["max"].groupby(daily.index.month).mean().plot(ax=axes[1], kind="bar")
axes[1].set(title="Mean daily max by month", xlabel="month", ylabel=value_units)
plt.tight_layout();

# %% [markdown]
# ## Periodogram of daily maximum
# The slow seasonal cycle is removed with a centered 121-day rolling mean to expose shorter modulation.

# %%
detr = daily["max"] - daily["max"].rolling(121, center=True, min_periods=60).mean()
ok = detr.notna()
t_days = (detr.index[ok] - detr.index[0]).days.values.astype(float)
y = detr[ok].values - detr[ok].mean()

top_periods = []
if len(y) > 30:
    periods = np.linspace(period_min_days, period_max_days, 2000)
    power = lombscargle(t_days, y, 2 * np.pi / periods, normalize=True)
    peaks, _ = find_peaks(power, prominence=0.02)
    top = sorted(peaks, key=lambda i: power[i], reverse=True)[:5]
    top_periods = [(float(periods[i]), float(power[i])) for i in top]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(periods, power)
    ax.axvline(27, ls="--", c="gray", label="27 d (solar rotation)")
    ax.set(xlabel="period (days)", ylabel="normalized power", title="Lomb-Scargle periodogram")
    ax.legend()
else:
    print("Not enough valid days (need > 30) for a periodogram.")
top_periods

# %%
summary = {
    "units": value_units,
    "data_source": "cleaned" if source == "clean" else "raw",
    "period_start": str(fine.index.min()),
    "period_end": str(fine.index.max()),
    "valid_days": int(daily["max"].notna().sum()),
    "overall_max": float(fine.max()),
    "mean_daily_max": float(daily["max"].mean()),
    "mean_daily_dose_unit_hours": float(daily["dose_unit_hours"].mean()),
    "n_anomaly_days": int(daily["is_anomaly"].sum()),
    "anomaly_dates": [str(d.date()) for d in daily.index[daily["is_anomaly"]]],
    "top_periods_days": [{"period": p, "power": w} for p, w in top_periods],
}
with open(product["summary"], "w") as f:
    json.dump(summary, f, indent=2)
summary
