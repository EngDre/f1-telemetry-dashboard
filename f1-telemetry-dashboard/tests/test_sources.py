import numpy as np

from f1dash import analysis, plots, sources


def test_synthetic_session_is_deterministic():
    a, b = sources.SyntheticSession(seed=3), sources.SyntheticSession(seed=3)
    assert a.laps.equals(b.laps)


def test_synthetic_laps_follow_the_standard_schema():
    s = sources.SyntheticSession()
    assert list(s.laps.columns) == analysis.LAP_COLUMNS
    assert set(s.laps["Driver"]) == set(s.drivers["Driver"])
    assert s.laps["LapNumber"].max() == 24


def test_synthetic_telemetry_matches_lap_time():
    s = sources.SyntheticSession()
    for driver in s.drivers["Driver"]:
        lap = analysis.fastest_lap_number(s.laps, driver)
        tel = s.lap_telemetry(driver, lap)
        lap_time = s.laps[(s.laps["Driver"] == driver) & (s.laps["LapNumber"] == lap)]["LapTimeSec"].iloc[0]
        assert 60 < lap_time < 120
        assert abs(tel["TimeSec"].iloc[-1] - lap_time) < 0.5  # last sample is one step before the line
        assert tel["Distance"].is_monotonic_increasing
        assert set(tel["Brake"].unique()) <= {0.0, 1.0}


def test_full_pipeline_runs_on_synthetic_data():
    s = sources.SyntheticSession()
    picks = ["AAA", "CCC"]
    tels = {d: s.lap_telemetry(d, analysis.fastest_lap_number(s.laps, d)) for d in picks}
    grids = analysis.align_laps(tels)
    winners = analysis.minisector_winners(grids)
    styles = plots.driver_styles(s.drivers, picks)
    assert set(winners["Winner"]) <= set(picks)
    for fig in (
        plots.telemetry_figure(grids, "AAA", styles),
        plots.dominance_figure(grids["AAA"], winners, styles),
        plots.speed_map_figure(grids["AAA"], "AAA"),
        plots.lap_times_figure(analysis.clean_race_laps(s.laps), styles),
        plots.stint_figure(analysis.stint_table(s.laps), list(s.drivers["Driver"])),
    ):
        assert len(fig.data) > 0


def test_teammates_get_different_line_styles():
    s = sources.SyntheticSession()
    styles = plots.driver_styles(s.drivers, ["AAA", "BBB"])
    assert styles["AAA"]["color"] == styles["BBB"]["color"]
    assert styles["AAA"]["dash"] != styles["BBB"]["dash"]


def test_softer_tyres_degrade_faster_in_synthetic_data():
    s = sources.SyntheticSession()
    d = s.laps[(s.laps["Driver"] == "BBB") & (~s.laps["IsPitLap"]) & (s.laps["LapNumber"] > 1)]
    slope = {}
    for comp, g in d.groupby("Compound"):
        # add back the fuel effect (0.04 s per lap) before fitting tyre age
        y = g["LapTimeSec"] + 0.04 * (g["LapNumber"] - 1)
        slope[comp] = np.polyfit(g["TyreLife"], y, 1)[0]
    assert slope["SOFT"] > slope["MEDIUM"]
