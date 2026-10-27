# %% [markdown]
# # 2. Clean (one target channel)
# For the target channel: sentinels -> NaN, physical range check, merge duplicate timestamps,
# Hampel spike filter, regular time grid, interpolation of SHORT gaps only.
#
# Outputs
# - `02_clean.csv`: `timestamp`, the cleaned channel, and `was_filled` (True where the value was interpolated)
# - `02_clean_report.csv`: one row with detailed statistics of every cleaning step
#
# The channel is `value_column` (env.yaml); the other settings come from `clean_config`.

# %% tags=["parameters"]
upstream = None
product = None
channel = None
config = None
cadence = "1min"

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(upstream["load"]["data"], parse_dates=["timestamp"])
available = [c for c in df.columns if c not in ("timestamp", "time_raw", "source_file")]
if channel not in available:
    raise KeyError(f"Target channel {channel!r} (value_column in env.yaml) is not in the load output; "
                   f"available: {available}")

sentinel_values = config["sentinel_values"]
min_valid = config.get("min_valid")
max_valid = config.get("max_valid")
hampel_window = config["hampel_window"]
hampel_sigmas = config["hampel_sigmas"]
hampel_floor_frac = config.get("hampel_floor_frac", 0.01)
max_gap_fill = config["max_gap_fill"]

n_rows = len(df)
n_unique_ts = int(df["timestamp"].nunique())
grid = pd.date_range(df["timestamp"].min().floor(cadence), df["timestamp"].max().floor(cadence), freq=cadence)
print(f"Channel: {channel} | {n_rows:,} rows, {n_unique_ts:,} unique timestamps, grid of {len(grid):,} slots at {cadence}")

# %% [markdown]
# ### Helpers
# **Hampel filter:** a point is a spike if it differs from the rolling median (`hampel_window` samples) by more
# than `hampel_sigmas` robust sigmas (1.4826 x MAD). At night the MAD is ~0, so a floor of
# `hampel_floor_frac` x channel maximum stops ordinary noise from being flagged.
#
# **Gap filling:** a gap of missing grid slots is interpolated linearly only if it is at most `max_gap_fill`
# slots long and has valid data on both sides. Longer gaps stay completely empty.

# %%
def hampel(x, window, n_sigmas, floor_frac):
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * mad
    floor = floor_frac * x.max() if x.notna().any() else 0.0
    return ((x - med).abs() > n_sigmas * np.maximum(sigma, floor)) & x.notna()


def fill_short_gaps(x, max_len):
    """Linearly fill NaN runs of length <= max_len that lie between valid values."""
    na = x.isna()
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    interp = x.interpolate(limit_area="inside")
    fill = na & (run_len <= max_len) & interp.notna()
    return x.where(~fill, interp), fill


def nan_runs(x):
    na = x.isna()
    run_id = (na != na.shift()).cumsum()
    runs = na.groupby(run_id).sum()
    return runs[runs > 0]

# %% [markdown]
# ## Clean

# %%
x = df[channel].copy()
raw_series = df.set_index("timestamp")[channel]
n_in_valid = int(x.notna().sum())

# --- row-level checks (before duplicates are merged, so a sentinel can never be averaged into a value)
is_sentinel = x.isin(sentinel_values)
x = x.mask(is_sentinel)
below = (x < min_valid) if min_valid is not None else pd.Series(False, index=x.index)
x = x.mask(below)
above = (x > max_valid) if max_valid is not None else pd.Series(False, index=x.index)
x = x.mask(above)
n_sentinel, n_below, n_above = int(is_sentinel.sum()), int(below.sum()), int(above.sum())
n_valid_rows = int(x.notna().sum())

# --- merge duplicate timestamps (mean of the valid values)
s = x.groupby(df["timestamp"]).mean()
n_after_merge = int(s.notna().sum())

# --- spikes
spikes = hampel(s, hampel_window, hampel_sigmas, hampel_floor_frac)
s_clean = s.mask(spikes)
n_spikes = int(spikes.sum())
n_kept = int(s_clean.notna().sum())
assert n_kept == n_after_merge - n_spikes

# --- regular grid and short-gap filling
reg = s_clean.resample(cadence).mean().reindex(grid)
n_grid_valid = int(reg.notna().sum())
filled, fill_flag = fill_short_gaps(reg, max_gap_fill)
n_filled = int(fill_flag.sum())
n_final = int(filled.notna().sum())
runs = nan_runs(filled)
longest = int(runs.max()) if len(runs) else 0
removed = n_sentinel + n_below + n_above + n_spikes

report = pd.DataFrame([{
    "channel": channel,
    "rows_in": n_rows,
    "values_missing_in_input": n_rows - n_in_valid,
    "values_valid_in_input": n_in_valid,
    "sentinel_removed": n_sentinel,
    "below_min_removed": n_below,
    "above_max_removed": n_above,
    "valid_rows_after_range_checks": n_valid_rows,
    "duplicate_rows_merged": n_rows - n_unique_ts,
    "values_after_merge": n_after_merge,
    "spikes_removed": n_spikes,
    "values_kept": n_kept,
    "removed_total": removed,
    "removed_pct_of_valid_input": round(100 * removed / n_in_valid, 3) if n_in_valid else np.nan,
    "grid_slots": len(grid),
    "grid_slots_with_data": n_grid_valid,
    "filled_by_interpolation": n_filled,
    "grid_slots_missing_after_fill": len(grid) - n_final,
    "coverage_pct": round(100 * n_final / len(grid), 3),
    "gaps_remaining": int(len(runs)),
    "longest_gap_samples": longest,
    "longest_gap": str(pd.Timedelta(cadence) * longest),
    "raw_min": float(raw_series.min()),
    "raw_max": float(raw_series.max()),
    "clean_min": float(filled.min()),
    "clean_max": float(filled.max()),
    "clean_mean": float(filled.mean()),
    "setting_sentinel_values": str(sentinel_values),
    "setting_min_valid": min_valid,
    "setting_max_valid": max_valid,
    "setting_hampel_window": hampel_window,
    "setting_hampel_sigmas": hampel_sigmas,
    "setting_hampel_floor_frac": hampel_floor_frac,
    "setting_max_gap_fill": max_gap_fill,
}])

# Transposed layout: one row per statistic, one column per channel (the saved CSV uses the same layout)
report_t = report.set_index("channel").T
report_t.index.name = "statistic"
report_t

# %% [markdown]
# ## Raw vs cleaned (hourly maximum)

# %%
fig, ax = plt.subplots(figsize=(11, 3))
raw_series.resample("1h").max().plot(ax=ax, lw=0.5, label="raw")
filled.resample("1h").max().plot(ax=ax, lw=0.5, label="cleaned")
ax.set_ylabel(channel); ax.legend(); ax.set_title(f"Cleaning overview: {channel}, hourly max");

# %%
out = pd.DataFrame({channel: filled, "was_filled": fill_flag})
out.index.name = "timestamp"
out.to_csv(product["data"])
report_t.to_csv(product["report"])
