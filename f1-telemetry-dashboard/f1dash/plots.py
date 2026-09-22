"""Plotly figures. Every function takes plain DataFrames and returns a ``go.Figure``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import analysis

BG = "#0F1319"
PANEL = "#171D26"
GRID = "#2A3340"
TEXT = "#E8ECF1"

COMPOUND_COLORS = {
    "SOFT": "#E5484D",
    "MEDIUM": "#F5C542",
    "HARD": "#E8ECF1",
    "INTERMEDIATE": "#46B26E",
    "WET": "#3D7FD9",
    "UNKNOWN": "#7A8594",
}


def driver_styles(drivers: pd.DataFrame, selected: list[str]) -> dict[str, dict]:
    """Colour and dash per driver. Team-mates share a colour, so the second one is dashed."""
    styles, used = {}, set()
    for name in selected:
        row = drivers[drivers["Driver"] == name]
        color = row["Color"].iloc[0] if not row.empty else "#4CC9F0"
        styles[name] = {"color": color, "dash": "dash" if color in used else "solid"}
        used.add(color)
    return styles


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=12),
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def _time_ticks(lo: float, hi: float, step: float = 1.0):
    vals = np.arange(np.floor(lo), np.ceil(hi) + step, step)
    return vals, [analysis.format_laptime(v) for v in vals]


def telemetry_figure(grids: dict[str, pd.DataFrame], ref: str, styles: dict[str, dict]) -> go.Figure:
    """Delta, speed, throttle, brake and gear against distance for the chosen laps."""
    titles = [f"Gap to {ref} (s)", "Speed (km/h)", "Throttle (%)", "Brake", "Gear"]
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.16, 0.34, 0.16, 0.10, 0.16], subplot_titles=titles,
    )
    for name, g in grids.items():
        st = dict(color=styles[name]["color"], dash=styles[name]["dash"], width=2)
        dist = g["Distance"]
        if name != ref:
            fig.add_trace(go.Scatter(x=dist, y=analysis.compute_delta(grids[ref], g), name=name, line=st, legendgroup=name, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=dist, y=g["Speed"], name=name, line=st, legendgroup=name), row=2, col=1)
        fig.add_trace(go.Scatter(x=dist, y=g["Throttle"], name=name, line=st, legendgroup=name, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=dist, y=g["Brake"], name=name, line=dict(st, shape="hv"), legendgroup=name, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=dist, y=g["nGear"], name=name, line=dict(st, shape="hv"), legendgroup=name, showlegend=False), row=5, col=1)
    fig.add_hline(y=0, line_color=GRID, row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1], row=4, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=5, col=1)
    fig.update_annotations(font=dict(size=12, color=TEXT), x=0.0, xanchor="left")
    return _style(fig, 860)


def dominance_figure(ref_grid: pd.DataFrame, winners: pd.DataFrame, styles: dict[str, dict]) -> go.Figure:
    """Track map where each mini-sector is coloured by the fastest driver through it."""
    fig = go.Figure()
    seen = set()
    for _, s in winners.iterrows():
        seg = ref_grid.iloc[int(s["Start"]): int(s["End"]) + 1]
        w = s["Winner"]
        fig.add_trace(
            go.Scatter(
                x=seg["X"], y=seg["Y"], mode="lines", name=w, legendgroup=w, showlegend=w not in seen,
                line=dict(color=styles[w]["color"], width=8),
                hovertemplate=f"Sector {int(s['Sector'])}<br>Fastest: {w}<br>Margin: {s['Margin']:.3f} s<extra></extra>",
            )
        )
        seen.add(w)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _style(fig, 560)


def speed_map_figure(grid: pd.DataFrame, driver: str) -> go.Figure:
    """Track map coloured by speed for one lap."""
    fig = go.Figure(
        go.Scatter(
            x=grid["X"], y=grid["Y"], mode="markers",
            marker=dict(size=6, color=grid["Speed"], colorscale="Viridis", colorbar=dict(title="km/h")),
            hovertemplate="%{marker.color:.0f} km/h<extra></extra>", name=driver,
        )
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _style(fig, 560)


def lap_times_figure(clean: pd.DataFrame, styles: dict[str, dict]) -> go.Figure:
    """Clean lap times against lap number."""
    fig = go.Figure()
    for name, d in clean.groupby("Driver"):
        if name not in styles:
            continue
        st = styles[name]
        fig.add_trace(
            go.Scatter(
                x=d["LapNumber"], y=d["LapTimeSec"], mode="lines+markers", name=name,
                line=dict(color=st["color"], dash=st["dash"], width=1.5), marker=dict(size=6),
                customdata=np.stack([d["Compound"], d["TyreLife"], d["LapTimeSec"].map(analysis.format_laptime)], axis=-1),
                hovertemplate="%{customdata[2]}<br>%{customdata[0]}, tyre age %{customdata[1]:.0f}<extra>" + name + "</extra>",
            )
        )
    if not clean.empty:
        vals, text = _time_ticks(clean["LapTimeSec"].min(), clean["LapTimeSec"].max(), 2.0)
        fig.update_yaxes(tickvals=vals, ticktext=text)
    fig.update_xaxes(title_text="Lap")
    fig.update_yaxes(title_text="Lap time")
    fig.update_layout(hovermode="closest")
    return _style(fig, 460)


def stint_figure(stints: pd.DataFrame, order: list[str]) -> go.Figure:
    """Tyre strategy: one bar per stint, coloured by compound."""
    fig = go.Figure()
    shown = set()
    for _, s in stints.iterrows():
        c = s["Compound"]
        fig.add_trace(
            go.Bar(
                y=[s["Driver"]], x=[s["Laps"]], base=[s["StartLap"] - 1], orientation="h",
                marker=dict(color=COMPOUND_COLORS.get(c, COMPOUND_COLORS["UNKNOWN"]), line=dict(color=BG, width=2)),
                name=c.title(), legendgroup=c, showlegend=c not in shown,
                hovertemplate=f"{c.title()}: laps {int(s['StartLap'])}-{int(s['EndLap'])}<extra>{s['Driver']}</extra>",
            )
        )
        shown.add(c)
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
    fig.update_xaxes(title_text="Lap")
    fig.update_layout(barmode="overlay", hovermode="closest")
    return _style(fig, max(260, 34 * len(order) + 120))
