"""Data sources for the dashboard.

Two implementations share one small interface (:class:`SessionData`):

* :class:`FastF1Session` wraps a real session loaded with the FastF1 library.
* :class:`SyntheticSession` generates a fictional circuit and race so the app
  (and the test suite) can run without any network access.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import analysis

FALLBACK_COLORS = ["#4CC9F0", "#F72585", "#B5E48C", "#FFB703", "#9D4EDD", "#F4A261", "#2A9D8F", "#E5989B"]


class SessionData:
    """Minimal interface the dashboard needs from a session."""

    title: str = ""
    is_synthetic: bool = False
    laps: pd.DataFrame  # columns: analysis.LAP_COLUMNS
    drivers: pd.DataFrame  # columns: Driver, Team, Color, FullName

    def lap_telemetry(self, driver: str, lap_number: int) -> pd.DataFrame:
        """Telemetry of one lap in the plain schema described in ``analysis``."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Real data (FastF1)
# --------------------------------------------------------------------------
def enable_cache(path: str = ".f1cache") -> None:
    """FastF1 downloads are slow; keep them on disk between runs."""
    import fastf1

    os.makedirs(path, exist_ok=True)
    fastf1.Cache.enable_cache(path)


def _already_happened(ts) -> bool:
    if pd.isna(ts):
        return True
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t <= pd.Timestamp.now(tz="UTC")
    except Exception:
        return True


def list_events(year: int) -> list[dict]:
    """Race weekends of a season with the sessions that have already taken place."""
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    events = []
    for _, row in schedule.iterrows():
        if "F1ApiSupport" in row and not bool(row["F1ApiSupport"]):
            continue
        sessions = []
        for i in range(1, 6):
            name = row.get(f"Session{i}")
            if isinstance(name, str) and name not in ("", "None") and _already_happened(row.get(f"Session{i}DateUtc")):
                sessions.append(name)
        if sessions:
            events.append(
                {
                    "round": int(row["RoundNumber"]),
                    "name": str(row["EventName"]),
                    "country": str(row["Country"]),
                    "sessions": sessions,
                }
            )
    return events


class FastF1Session(SessionData):
    def __init__(self, session):
        self._session = session
        self.is_synthetic = False
        self.title = f"{session.event.year} {session.event['EventName']} - {session.name}"
        self.laps = analysis.standardize_laps(session.laps)
        self.drivers = self._driver_table(session, self.laps)

    @staticmethod
    def _driver_table(session, laps: pd.DataFrame) -> pd.DataFrame:
        present = set(laps["Driver"])
        rows = []
        for _, r in session.results.iterrows():
            abbr = r.get("Abbreviation")
            if abbr not in present:
                continue
            color = r.get("TeamColor")
            color = f"#{color}" if isinstance(color, str) and len(color) == 6 else None
            rows.append(
                {
                    "Driver": abbr,
                    "Team": r.get("TeamName", ""),
                    "Color": color or FALLBACK_COLORS[len(rows) % len(FALLBACK_COLORS)],
                    "FullName": r.get("FullName", abbr),
                }
            )
        return pd.DataFrame(rows, columns=["Driver", "Team", "Color", "FullName"])

    def lap_telemetry(self, driver: str, lap_number: int) -> pd.DataFrame:
        laps = self._session.laps.pick_drivers(driver).pick_laps(int(lap_number))
        if len(laps) == 0:
            raise ValueError(f"No lap {lap_number} found for {driver}.")
        return analysis.standardize_telemetry(laps.iloc[0].get_telemetry())


def load_fastf1_session(year: int, round_number: int, session_name: str, cache_dir: str = ".f1cache") -> FastF1Session:
    import fastf1

    enable_cache(cache_dir)
    session = fastf1.get_session(year, round_number, session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return FastF1Session(session)


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
_N_POINTS = 800
_HARMONICS = [(0.22, 2, 0.0), (0.14, 3, 0.6), (0.08, 5, 0.3), (0.05, 7, 0.0), (0.04, 9, 1.0), (0.025, 13, 0.4)]

# (abbreviation, team, colour, grip, power)
_SYNTH_DRIVERS = [
    ("AAA", "Team Alpha", "#4CC9F0", 1.000, 1.000),
    ("BBB", "Team Alpha", "#4CC9F0", 0.988, 1.000),
    ("CCC", "Team Beta", "#F72585", 1.006, 0.985),
    ("DDD", "Team Beta", "#F72585", 0.978, 0.990),
]
_SYNTH_STRATEGY = {
    "AAA": [("MEDIUM", 12), ("HARD", 12)],
    "BBB": [("SOFT", 9), ("MEDIUM", 15)],
    "CCC": [("MEDIUM", 11), ("HARD", 13)],
    "DDD": [("SOFT", 8), ("HARD", 16)],
}
_COMPOUND_OFFSET = {"SOFT": -0.55, "MEDIUM": 0.0, "HARD": 0.45}
_COMPOUND_DEG = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}


def _build_circuit():
    theta = np.linspace(0, 2 * np.pi, _N_POINTS, endpoint=False)
    radius = 780 * (1 + sum(a * np.cos(k * theta + p) for a, k, p in _HARMONICS))
    x, y = radius * np.cos(theta), radius * np.sin(theta)
    dx = (np.roll(x, -1) - np.roll(x, 1)) / 2
    dy = (np.roll(y, -1) - np.roll(y, 1)) / 2
    ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
    ds = np.hypot(dx, dy)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(ds**3, 1e-9)
    return x, y, ds, curvature


def _speed_profile(curvature, ds, grip, power, a_lat=22.0, a_acc=7.0, a_brk=42.0):
    """Quasi-steady-state speed limit: cornering limit, then acceleration and braking passes."""
    v = np.minimum(92.0 * power, np.sqrt(a_lat * grip / np.maximum(curvature, 1e-4)))
    n = len(v)
    for _ in range(2):  # two passes so the start/finish line wraps consistently
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_acc * power * ds[i]))
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_brk * ds[j]))
    return v


class SyntheticSession(SessionData):
    """A fictional 4-driver, 24-lap race on a generated circuit. Not real F1 data."""

    def __init__(self, seed: int = 7, n_laps: int = 24):
        self.is_synthetic = True
        self.title = "Synthetic demo race (generated data, not real F1 data)"
        rng = np.random.default_rng(seed)
        x, y, ds, curv = _build_circuit()
        self._x, self._y = x * 10, y * 10  # FastF1 reports X/Y in 1/10 m
        self._distance = np.concatenate([[0.0], np.cumsum(ds[:-1])])
        self._profiles = {}
        for abbr, _team, _color, grip, power in _SYNTH_DRIVERS:
            v = _speed_profile(curv, ds, grip, power)
            time = np.concatenate([[0.0], np.cumsum(ds[:-1] / v[:-1])])
            base = float(np.sum(ds / v))
            self._profiles[abbr] = {"v": v, "ds": ds, "time": time, "base": base}
        self.drivers = pd.DataFrame(
            [{"Driver": a, "Team": t, "Color": c, "FullName": f"Driver {a}"} for a, t, c, _g, _p in _SYNTH_DRIVERS]
        )
        self.laps = self._build_laps(rng, n_laps)

    def _build_laps(self, rng, n_laps: int) -> pd.DataFrame:
        rows = []
        for abbr, team, *_ in _SYNTH_DRIVERS:
            base = self._profiles[abbr]["base"]
            lap_no = 0
            stints = _SYNTH_STRATEGY[abbr]
            for stint_idx, (compound, length) in enumerate(stints):
                for life in range(1, length + 1):
                    lap_no += 1
                    is_out = stint_idx > 0 and life == 1
                    is_in = stint_idx < len(stints) - 1 and life == length
                    t = base + _COMPOUND_OFFSET[compound] + _COMPOUND_DEG[compound] * (life - 1)
                    t += -0.04 * (lap_no - 1) + rng.normal(0, 0.12)
                    t += 5.0 if lap_no == 1 else 0.0
                    t += 2.5 if is_in else 0.0
                    t += 16.0 if is_out else 0.0
                    rows.append(
                        {
                            "Driver": abbr, "Team": team, "LapNumber": lap_no, "LapTimeSec": t,
                            "Stint": float(stint_idx + 1), "Compound": compound, "TyreLife": float(life),
                            "IsPitLap": bool(is_in or is_out), "IsAccurate": not (is_in or is_out or lap_no == 1),
                        }
                    )
        return pd.DataFrame(rows)[analysis.LAP_COLUMNS]

    def lap_telemetry(self, driver: str, lap_number: int) -> pd.DataFrame:
        lap = self.laps[(self.laps["Driver"] == driver) & (self.laps["LapNumber"] == lap_number)]
        if lap.empty:
            raise ValueError(f"No lap {lap_number} found for {driver}.")
        p = self._profiles[driver]
        factor = float(lap["LapTimeSec"].iloc[0]) / p["base"]  # slower lap = stretched time axis
        v, ds = p["v"], p["ds"]
        accel = (np.roll(v, -1) ** 2 - v**2) / (2 * ds)
        braking = accel < -3.0
        throttle = np.where(braking, 0.0, np.where((accel > 1.0) | (v >= 0.98 * v.max()), 100.0, 45.0))
        gear = np.digitize(v * 3.6, [70, 110, 150, 190, 230, 270, 300]) + 1
        return pd.DataFrame(
            {
                "Distance": self._distance,
                "Speed": v * 3.6 / factor,
                "Throttle": throttle,
                "Brake": braking.astype(float),
                "nGear": gear.astype(float),
                "X": self._x,
                "Y": self._y,
                "TimeSec": p["time"] * factor,
            }
        )


# --------------------------------------------------------------------------
# Entry point used by the app
# --------------------------------------------------------------------------
def load_session(kind: str, year: int | None = None, round_number: int | None = None, session_name: str | None = None) -> SessionData:
    """Load a session. ``kind`` is ``"fastf1"`` or ``"synthetic"``."""
    if kind == "synthetic":
        return SyntheticSession()
    if kind == "fastf1":
        return load_fastf1_session(int(year), int(round_number), str(session_name))
    raise ValueError(f"Unknown data source: {kind!r}")
