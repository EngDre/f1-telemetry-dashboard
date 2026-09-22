"""Smoke tests: run the Streamlit script headlessly with no network access."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from f1dash import sources

APP = str(Path(__file__).resolve().parents[1] / "app.py")
EVENTS = [{"round": 1, "name": "Test Grand Prix", "country": "Nowhere", "sessions": ["Qualifying", "Race"]}]


def _patch(monkeypatch):
    monkeypatch.setattr(sources, "list_events", lambda year: EVENTS)
    monkeypatch.setattr(sources, "load_fastf1_session", lambda *a, **k: sources.SyntheticSession())


def test_app_shows_empty_state_first(monkeypatch):
    _patch(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Compare laps" in h.value for h in at.header)


def test_synthetic_demo_renders_all_tabs(monkeypatch):
    _patch(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio(key="source").set_value("Synthetic demo").run()
    at.sidebar.button(key="load").click().run()
    assert not at.exception
    assert len(at.tabs) == 3
    assert len(at.multiselect) == 1


def test_real_data_branch_uses_the_fastf1_loader(monkeypatch):
    _patch(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox(key="session").set_value("Race")
    at.sidebar.button(key="load").click().run()
    assert not at.exception
    assert len(at.tabs) == 3
