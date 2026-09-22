"""Pure analysis functions.

Nothing in this module imports FastF1 or Streamlit, so every function can be
unit-tested with small hand-made DataFrames.

Conventions
-----------
* Telemetry DataFrames have the columns ``Distance`` (m), ``Speed`` (km/h),
  ``Throttle`` (0-100), ``Brake`` (0/1), ``nGear``, ``X``, ``Y`` and ``TimeSec``
  (seconds since the start of the lap).
* Lap DataFrames have the columns listed in :data:`LAP_COLUMNS`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAP_COLUMNS = [
    "Driver", "Team", "LapNumber", "LapTimeSec", "Stint",
    "Compound", "TyreLife", "IsPitLap", "IsAccurate",
]
CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "X", "Y", "TimeSec"]


# --------------------------------------------------------------------------
# Standardising raw FastF1 objects
# --------------------------------------------------------------------------
def standardize_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a FastF1 ``session.laps`` table into the plain schema used here."""
    if "Deleted" in raw:
        deleted = raw["Deleted"].astype("boolean").fillna(False).astype(bool)
    else:
        deleted = pd.Series(False, index=raw.index)
    if "IsAccurate" in raw:
        accurate = raw["IsAccurate"].astype("boolean").fillna(False).astype(bool)
    else:
        accurate = pd.Series(True, index=raw.index)
    out = pd.DataFrame(
        {
            "Driver": raw["Driver"].astype(str),
            "Team": raw["Team"].astype(str),
            "LapNumber": raw["LapNumber"].astype(float).astype(int),
            "LapTimeSec": raw["LapTime"].dt.total_seconds(),
            "Stint": raw["Stint"].astype(float),
            "Compound": raw["Compound"].fillna("UNKNOWN").astype(str).str.upper(),
            "TyreLife": raw["TyreLife"].astype(float),
            "IsPitLap": raw["PitInTime"].notna() | raw["PitOutTime"].notna(),
            "IsAccurate": accurate & ~deleted,
        }
    )
    return out[LAP_COLUMNS].reset_index(drop=True)


def standardize_telemetry(tel: pd.DataFrame) -> pd.DataFrame:
    """Convert FastF1 lap telemetry (from ``Lap.get_telemetry()``) to plain columns."""
    out = pd.DataFrame(
        {
            "Distance": tel["Distance"].to_numpy(dtype=float),
            "Speed": tel["Speed"].to_numpy(dtype=float),
            "Throttle": tel["Throttle"].to_numpy(dtype=float),
            "Brake": tel["Brake"].astype(float).to_numpy(),
            "nGear": tel["nGear"].to_numpy(dtype=float),
            "X": tel["X"].to_numpy(dtype=float),
            "Y": tel["Y"].to_numpy(dtype=float),
            "TimeSec": tel["Time"].dt.total_seconds().to_numpy(),
        }
    )
    out = out.dropna(subset=["Distance", "Speed", "TimeSec"]).reset_index(drop=True)
    out["TimeSec"] = out["TimeSec"] - out["TimeSec"].iloc[0]
    return out


# --------------------------------------------------------------------------
# Comparing laps along distance
# --------------------------------------------------------------------------
def to_distance_grid(tel: pd.DataFrame, n: int = 1000, length: float | None = None) -> pd.DataFrame:
    """Resample a lap onto ``n`` evenly spaced distance points.

    Telemetry is sampled in time, not distance, so two laps cannot be compared
    sample by sample. Interpolating both onto the same distance grid solves that.
    """
    dist = np.maximum.accumulate(tel["Distance"].to_numpy(dtype=float))
    end = float(length) if length is not None else float(dist[-1])
    grid = np.linspace(0.0, end, n)
    cols = {"Distance": grid}
    for ch in CHANNELS:
        cols[ch] = np.interp(grid, dist, tel[ch].to_numpy(dtype=float))
    out = pd.DataFrame(cols)
    out["Brake"] = (out["Brake"] > 0.5).astype(int)
    out["nGear"] = out["nGear"].round().astype(int)
    out["TimeSec"] = out["TimeSec"] - out["TimeSec"].iloc[0]
    return out


def align_laps(tels: dict[str, pd.DataFrame], n: int = 1000) -> dict[str, pd.DataFrame]:
    """Put several laps on one shared distance grid (cut to the shortest lap)."""
    length = min(float(np.nanmax(t["Distance"])) for t in tels.values())
    return {name: to_distance_grid(t, n=n, length=length) for name, t in tels.items()}


def compute_delta(ref: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Cumulative time gap (s) along the lap. Positive means ``other`` is behind ``ref``."""
    return other["TimeSec"].to_numpy() - ref["TimeSec"].to_numpy()


def minisector_winners(grids: dict[str, pd.DataFrame], n_sectors: int = 25) -> pd.DataFrame:
    """Find which driver was fastest through each mini-sector.

    The lap is cut into ``n_sectors`` slices of equal distance. The driver with
    the smallest elapsed time through a slice wins it.
    """
    n = len(next(iter(grids.values())))
    edges = np.linspace(0, n - 1, n_sectors + 1).astype(int)
    rows = []
    for i in range(n_sectors):
        a, b = edges[i], edges[i + 1]
        times = {k: g["TimeSec"].iloc[b] - g["TimeSec"].iloc[a] for k, g in grids.items()}
        ranked = sorted(times.items(), key=lambda kv: kv[1])
        margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else 0.0
        rows.append({"Sector": i + 1, "Start": a, "End": b, "Winner": ranked[0][0], "Margin": margin})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Lap tables
# --------------------------------------------------------------------------
def fastest_lap_number(laps: pd.DataFrame, driver: str) -> int | None:
    """Lap number of a driver's quickest valid lap (falls back to any timed lap)."""
    d = laps[(laps["Driver"] == driver) & laps["LapTimeSec"].notna()]
    valid = d[d["IsAccurate"]]
    d = valid if not valid.empty else d
    if d.empty:
        return None
    return int(d.loc[d["LapTimeSec"].idxmin(), "LapNumber"])


def clean_race_laps(laps: pd.DataFrame, threshold: float = 1.07) -> pd.DataFrame:
    """Keep representative laps: timed, accurate, no pit in/out, within 107 % of the best."""
    ok = laps[laps["IsAccurate"] & ~laps["IsPitLap"] & laps["LapTimeSec"].notna()]
    if ok.empty:
        return ok
    return ok[ok["LapTimeSec"] <= threshold * ok["LapTimeSec"].min()]


def stint_table(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver and stint with compound, first/last lap and length."""
    g = laps.dropna(subset=["Stint"]).groupby(["Driver", "Stint"], as_index=False)
    out = g.agg(
        Compound=("Compound", "first"),
        StartLap=("LapNumber", "min"),
        EndLap=("LapNumber", "max"),
    )
    out["Laps"] = out["EndLap"] - out["StartLap"] + 1
    return out


def pace_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Median and best clean lap per driver, sorted by median pace."""
    if clean.empty:
        return pd.DataFrame(columns=["Driver", "MedianSec", "BestSec", "Laps", "GapSec"])
    out = clean.groupby("Driver", as_index=False).agg(
        MedianSec=("LapTimeSec", "median"),
        BestSec=("LapTimeSec", "min"),
        Laps=("LapNumber", "count"),
    )
    out = out.sort_values("MedianSec").reset_index(drop=True)
    out["GapSec"] = out["MedianSec"] - out["MedianSec"].iloc[0]
    return out


def format_laptime(seconds: float | None) -> str:
    """Format seconds as m:ss.mmm. Missing values become a plain hyphen."""
    if seconds is None or not np.isfinite(seconds):
        return "-"
    minutes, rest = divmod(float(seconds), 60.0)
    return f"{int(minutes)}:{rest:06.3f}"
