# %% [markdown]
# # 1. Load the raw file
# Reads ONE file, `<folder_input>/<load_config.raw_file>`, using the explicit settings in `load_config`
# (env.yaml): delimiter, decimal mark, column order, time columns, time format, timezone.
#
# This task only parses. No values are altered or removed (cleaning is task 2), but it reports what
# looks wrong: unparsable rows, duplicate timestamps, gaps.
# Output: `timestamp` (naive UTC) + the remaining columns in the configured order.

# %% tags=["parameters"]
upstream = None
product = None
folder_input = None
config = None

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path(folder_input) / config["raw_file"]
delimiter = config["delimiter"]
decimal = config.get("decimal", ".")
comment_char = config.get("comment_char")
has_header = config["has_header"]
columns = config["columns"]
time_columns = config["time_columns"]
time_format = config["time_format"]
timezone = config.get("timezone", "UTC")
keep_raw_time = config.get("keep_raw_time", False)

# --- validate the configuration up front
if len(set(columns)) != len(columns):
    raise ValueError(f"`columns` contains duplicate names: {columns}")
unknown = [c for c in time_columns if c not in columns]
if unknown:
    raise ValueError(f"time_columns {unknown} are not in `columns` {columns}")
unix = time_format in ("unix_s", "unix_ms")
if unix and len(time_columns) != 1:
    raise ValueError("unix time formats need exactly one time column")
channels = [c for c in columns if c not in time_columns]

if not path.is_file():
    found = sorted(p.name for p in Path(folder_input).glob("*")) if Path(folder_input).is_dir() else "folder not found"
    raise FileNotFoundError(f"{path} does not exist. Files in {folder_input}: {found}")
print(f"File: {path}")
print(f"Time: {time_columns} as {time_format} | channels: {channels}")

# %% [markdown]
# ## Parse

# %%
def parse_time(df):
    parts = [df[c].str.strip() for c in time_columns]
    if unix:
        epoch = pd.to_numeric(parts[0], errors="coerce")
        return pd.to_datetime(epoch, unit="s" if time_format == "unix_s" else "ms", errors="coerce")
    joined = parts[0]
    for p in parts[1:]:
        joined = joined + " " + p
    ts = pd.to_datetime(joined, format=time_format, errors="coerce")
    if str(timezone).upper() != "UTC":
        ts = (ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
                .dt.tz_convert("UTC").dt.tz_localize(None))
    return ts

def raw_time(df):
    """Original time value as in the file: a number for unix formats, else the joined text."""
    parts = [df[c].str.strip() for c in time_columns]
    if unix:
        return pd.to_numeric(parts[0], errors="coerce")
    joined = parts[0]
    for p in parts[1:]:
        joined = joined + " " + p
    return joined

df = pd.read_csv(path, sep=delimiter, engine="python", dtype=str, on_bad_lines="skip",
                 comment=comment_char or None, header=0 if has_header else None)
if df.shape[1] != len(columns):
    raise ValueError(f"{path.name}: found {df.shape[1]} columns but `columns` lists {len(columns)}. "
                     f"Check `delimiter` ({delimiter!r}) and `columns` in env.yaml.")
df.columns = columns
n_rows = len(df)

ts = parse_time(df)
n_bad_time = int(ts.isna().sum())          # repeated header lines, blank or garbled rows
keep = ts.notna()
df, ts = df.loc[keep], ts[keep]
if len(ts) == 0:
    raise ValueError(f"{path.name}: no row has a valid time; check time_columns/time_format in env.yaml")
if not (pd.Timestamp("2000-01-01") <= ts.min() and ts.max() <= pd.Timestamp("2100-01-01")):
    raise ValueError(f"{path.name}: times span {ts.min()} .. {ts.max()}; check time_format in env.yaml")

data = pd.DataFrame({"timestamp": ts.values})
if keep_raw_time:
    data["time_raw"] = raw_time(df).values

n_bad_vals = 0
for c in channels:
    raw = df[c].str.strip()
    if decimal != ".":
        raw = raw.str.replace(decimal, ".", regex=False)
    num = pd.to_numeric(raw, errors="coerce")
    n_bad_vals += int((num.isna() & raw.notna()).sum())      # present but non-numeric
    data[c] = num.values
data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)

pd.DataFrame([{"file": path.name, "rows_read": n_rows, "bad_time_rows": n_bad_time,
               "non_numeric_values": n_bad_vals, "rows_kept": len(data),
               "start": data["timestamp"].min(), "end": data["timestamp"].max()}])

# %% [markdown]
# ## Sanity checks: duplicates, cadence, gaps

# %%
dup = data["timestamp"].duplicated(keep=False)
print(f"Rows sharing a timestamp: {int(dup.sum()):,} (identical duplicates: {int(data.duplicated().sum()):,})")

unique_ts = data["timestamp"].drop_duplicates().reset_index(drop=True)
steps = unique_ts.diff().dropna()
cadence = steps.median()
print(f"Median sampling step: {cadence}")
gaps = steps[steps > 2 * cadence]
print(f"Gaps longer than 2x cadence: {len(gaps)}; total missing time: {gaps.sum() - cadence * len(gaps)}")
top_gaps = pd.DataFrame(columns=["gap_ends_at", "gap_length"])
if len(gaps):
    top_gaps = (pd.DataFrame({"gap_ends_at": unique_ts.loc[gaps.index].values, "gap_length": gaps.values})
                .sort_values("gap_length", ascending=False).head(10))
top_gaps

# %% [markdown]
# ## Quick look at each channel (hourly max)

# %%
hourly = data.set_index("timestamp")[channels].resample("1h").max()
fig, axes = plt.subplots(len(channels), 1, figsize=(11, 2 * len(channels)), sharex=True, squeeze=False)
for ax, c in zip(axes[:, 0], channels):
    hourly[c].plot(ax=ax, lw=0.5); ax.set_ylabel(c)
axes[0, 0].set_title("Raw channels, hourly max");

# %%
Path(product["data"]).parent.mkdir(parents=True, exist_ok=True)
data.to_csv(product["data"], index=False)
