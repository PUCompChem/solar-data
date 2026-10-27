# %% [markdown]
# # 2. Clean
# Sentinels -> NaN, physical range check, duplicate merge, Hampel spike filter, regular resampling.

# %% tags=["parameters"]
upstream = ["load"]
product = None
value_column = "uv"
sentinel_values = [-999, -9999, 9999]
min_valid = 0.0
max_valid = None
hampel_window = 11
hampel_sigmas = 6.0
cadence = "1min"
max_gap_fill = 5

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(upstream["load"]["data"], parse_dates=["timestamp"])
s = df.groupby("timestamp")[value_column].mean()   # merges duplicate timestamps
s.index = pd.DatetimeIndex(s.index)
n0 = s.notna().sum()

# %%
s = s.mask(s.isin(sentinel_values))
if min_valid is not None:
    s = s.mask(s < min_valid)
if max_valid is not None:
    s = s.mask(s > max_valid)
n_range = s.notna().sum()

# %% [markdown]
# ### Hampel spike filter
# A point is a spike if it deviates from the rolling median by more than `hampel_sigmas` robust sigmas (MAD).
# Night-time MAD is ~0, so a small floor (1% of max) avoids flagging ordinary noise there.

# %%
def hampel(x, window, n_sigmas):
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * mad
    floor = 0.01 * x.max() if x.notna().any() else 0
    return (x - med).abs() > n_sigmas * np.maximum(sigma, floor)

spikes = hampel(s, hampel_window, hampel_sigmas)
s_clean = s.mask(spikes)
print(f"raw valid: {n0:,} | after range check: {n_range:,} | spikes removed: {int(spikes.sum()):,}")

# %%
reg = s_clean.resample(cadence).mean()
reg_filled = reg.interpolate(limit=max_gap_fill, limit_area="inside")
out = pd.DataFrame({value_column: reg_filled, "was_filled": reg.isna() & reg_filled.notna()})
print(f"Coverage on regular grid: {out[value_column].notna().mean():.1%}")

# %%
fig, ax = plt.subplots(figsize=(11, 3))
s.resample("1h").max().plot(ax=ax, lw=0.5, label="raw (range-checked), hourly max")
s_clean.resample("1h").max().plot(ax=ax, lw=0.5, label="cleaned, hourly max")
ax.legend(); ax.set_title("Cleaning overview");

# %%
out.index.name = "timestamp"
out.to_csv(product["data"])
