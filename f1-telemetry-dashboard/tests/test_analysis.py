import numpy as np
import pandas as pd
import pytest

from f1dash import analysis


def make_tel(length=1000.0, n=200, lap_time=80.0):
    """Constant-speed lap: time grows linearly with distance."""
    dist = np.linspace(0, length, n)
    return pd.DataFrame(
        {
            "Distance": dist,
            "Speed": np.full(n, length / lap_time * 3.6),
            "Throttle": np.full(n, 100.0),
            "Brake": np.zeros(n),
            "nGear": np.full(n, 6.0),
            "X": dist,
            "Y": np.zeros(n),
            "TimeSec": dist / length * lap_time,
        }
    )


def test_distance_grid_has_requested_size_and_is_monotonic():
    grid = analysis.to_distance_grid(make_tel(), n=500)
    assert len(grid) == 500
    assert grid["Distance"].is_monotonic_increasing
    assert grid["TimeSec"].iloc[0] == 0
    assert set(grid["Brake"].unique()) <= {0, 1}


def test_align_laps_cuts_to_shortest_lap():
    grids = analysis.align_laps({"A": make_tel(length=1000), "B": make_tel(length=980)}, n=100)
    assert grids["A"]["Distance"].iloc[-1] == pytest.approx(980)
    assert grids["B"]["Distance"].iloc[-1] == pytest.approx(980)


def test_delta_is_zero_for_identical_laps():
    g = analysis.to_distance_grid(make_tel())
    assert np.allclose(analysis.compute_delta(g, g), 0)


def test_delta_grows_linearly_for_a_uniformly_slower_lap():
    ref = analysis.to_distance_grid(make_tel(lap_time=80.0))
    slow = analysis.to_distance_grid(make_tel(lap_time=81.6))
    delta = analysis.compute_delta(ref, slow)
    assert delta[-1] == pytest.approx(1.6, abs=1e-6)
    assert np.all(np.diff(delta) >= -1e-9)


def test_minisector_winners_picks_the_faster_driver_per_sector():
    n = 101
    dist = np.linspace(0, 100, n)

    def grid(rates):
        t = np.concatenate([[0.0], np.cumsum(rates)])
        return pd.DataFrame({"Distance": dist, "TimeSec": t})

    a = grid([0.9] * 50 + [1.1] * 50)  # fast first half, slow second half
    b = grid([1.0] * 100)
    res = analysis.minisector_winners({"A": a, "B": b}, n_sectors=2)
    assert list(res["Winner"]) == ["A", "B"]
    assert list(res["Margin"].round(6)) == [5.0, 5.0]


def _laps():
    return pd.DataFrame(
        {
            "Driver": ["AAA"] * 5 + ["BBB"] * 3,
            "Team": ["T"] * 8,
            "LapNumber": [1, 2, 3, 4, 5, 1, 2, 3],
            "LapTimeSec": [95.0, 80.0, 79.0, 90.0, 79.5, 81.0, 80.5, np.nan],
            "Stint": [1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            "Compound": ["SOFT", "SOFT", "SOFT", "HARD", "HARD", "MEDIUM", "MEDIUM", "MEDIUM"],
            "TyreLife": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            "IsPitLap": [False, False, True, True, False, False, False, False],
            "IsAccurate": [False, True, False, False, True, True, True, False],
        }
    )


def test_fastest_lap_ignores_inaccurate_laps():
    laps = _laps()
    assert analysis.fastest_lap_number(laps, "AAA") == 5  # lap 3 (79.0 s) is faster but flagged inaccurate
    assert analysis.fastest_lap_number(laps, "BBB") == 2
    assert analysis.fastest_lap_number(laps, "ZZZ") is None


def test_clean_race_laps_removes_pit_laps_and_slow_outliers():
    clean = analysis.clean_race_laps(_laps())
    assert not clean["IsPitLap"].any()
    assert clean["LapTimeSec"].max() <= 1.07 * clean["LapTimeSec"].min()
    assert set(zip(clean["Driver"], clean["LapNumber"])) == {("AAA", 2), ("AAA", 5), ("BBB", 1), ("BBB", 2)}


def test_stint_table_and_pace_summary():
    stints = analysis.stint_table(_laps())
    a = stints[stints["Driver"] == "AAA"].sort_values("Stint")
    assert list(a["Compound"]) == ["SOFT", "HARD"]
    assert list(a["Laps"]) == [3, 2]
    pace = analysis.pace_summary(analysis.clean_race_laps(_laps()))
    assert pace["MedianSec"].is_monotonic_increasing
    assert pace["GapSec"].iloc[0] == 0


@pytest.mark.parametrize("seconds,expected", [(83.456, "1:23.456"), (59.9, "0:59.900"), (None, "-"), (float("nan"), "-")])
def test_format_laptime(seconds, expected):
    assert analysis.format_laptime(seconds) == expected


def test_standardize_laps_handles_fastf1_style_columns():
    raw = pd.DataFrame(
        {
            "Driver": ["VER", "VER"],
            "Team": ["Red Bull Racing"] * 2,
            "LapNumber": [1.0, 2.0],
            "LapTime": pd.to_timedelta([np.nan, 83.5], unit="s"),
            "Stint": [1.0, 1.0],
            "Compound": ["soft", None],
            "TyreLife": [1.0, 2.0],
            "PitInTime": pd.to_timedelta([np.nan, np.nan], unit="s"),
            "PitOutTime": pd.to_timedelta([100.0, np.nan], unit="s"),
            "IsAccurate": [False, True],
            "Deleted": [None, True],
        }
    )
    out = analysis.standardize_laps(raw)
    assert list(out.columns) == analysis.LAP_COLUMNS
    assert out["LapTimeSec"].iloc[1] == pytest.approx(83.5)
    assert list(out["Compound"]) == ["SOFT", "UNKNOWN"]
    assert list(out["IsPitLap"]) == [True, False]
    assert list(out["IsAccurate"]) == [False, False]  # deleted lap is not accurate


def test_standardize_telemetry_zeroes_time_and_casts_brake():
    tel = pd.DataFrame(
        {
            "Time": pd.to_timedelta([5.0, 5.2, 5.4], unit="s"),
            "Distance": [0.0, 10.0, 20.0],
            "Speed": [200.0, 210.0, 220.0],
            "Throttle": [100, 100, 90],
            "Brake": [False, False, True],
            "nGear": [6, 6, 7],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 1.0, 2.0],
        }
    )
    out = analysis.standardize_telemetry(tel)
    assert out["TimeSec"].iloc[0] == 0
    assert out["TimeSec"].iloc[-1] == pytest.approx(0.4)
    assert list(out["Brake"]) == [0.0, 0.0, 1.0]
