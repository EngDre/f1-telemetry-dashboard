# F1 Telemetry Explorer

An interactive dashboard for comparing Formula 1 laps, corner by corner. Pick a season, Grand Prix and session, choose two to four drivers, and see where the time is gained or lost along the lap.

<!-- After you deploy, add your live link and a screenshot here:
**Live demo:** https://YOUR-APP-NAME.streamlit.app
![Lap comparison](docs/lap-comparison.png)
-->

Built with Python, [FastF1](https://docs.fastf1.dev), pandas, NumPy, Plotly and Streamlit.

## What it shows

| Tab | What you get |
|---|---|
| **Lap comparison** | Speed, throttle, brake and gear traces against distance, plus the cumulative time gap to a reference driver. |
| **Track map** | The circuit with each mini-sector coloured by the fastest driver through it, next to a speed map of the reference lap. |
| **Race pace and tyres** | Clean lap times against lap number, median and best pace per driver, and the tyre strategy of every driver. |

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/f1-telemetry-dashboard.git
cd f1-telemetry-dashboard
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first time you load a real session, FastF1 downloads the data and stores it in `.f1cache/`. This can take a minute; later loads of the same session are fast.

No internet connection, or the F1 data service is down? Switch **Data source** to **Synthetic demo** in the sidebar. It runs the whole app on a generated circuit and race, and shows a warning so it is never mistaken for real data.

## How it works

1. **Load.** FastF1 provides lap timing and car telemetry (about 4 to 5 samples per second per car). `f1dash/sources.py` wraps a session behind a small interface so the rest of the app never touches FastF1 directly.
2. **Standardise.** Raw FastF1 tables are converted to plain numeric schemas (`analysis.standardize_laps`, `analysis.standardize_telemetry`).
3. **Align by distance.** Telemetry is sampled in time, so two laps cannot be compared row by row. Each lap is interpolated onto the same evenly spaced distance grid (`analysis.align_laps`).
4. **Compare.**
   - *Gap:* the difference in elapsed time at each distance point.
   - *Mini-sector winners:* the lap is cut into 25 slices of equal length, and the driver with the smallest elapsed time through a slice wins it.
   - *Clean laps:* valid laps that are not in or out laps and are within 107 % of the fastest lap, the same idea as FastF1's quick-lap filter.

## Project structure

```
app.py               Streamlit interface
f1dash/
  sources.py         FastF1 wrapper and a synthetic session generator
  analysis.py        Pure functions: alignment, gap, mini-sectors, lap tables
  plots.py           Plotly figures
tests/               pytest suite (no network needed)
.streamlit/          Dashboard theme
.github/workflows/   CI: runs the tests on every push
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the analysis functions with small hand-made laps (for example, a lap that is uniformly 2 % slower must produce a gap that grows to exactly that amount), the conversion of FastF1-style tables, and a headless run of the Streamlit app. They use the synthetic session, so they run without internet access. The real FastF1 loading path is exercised only when you run the app yourself.

## Deploy for free on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app**.
3. Select the repository, branch `main` and the file `app.py`, then deploy.
4. Copy the app URL into the top of this README.

The app keeps at most two loaded sessions in memory, because a session with telemetry is large and the free tier has limited RAM.

## Limitations

- Comparisons use each driver's fastest valid lap, which can be set on different tyres, fuel loads and track conditions. The table above the chart shows the compound and tyre age so you can judge this.
- Telemetry is sampled a few times per second, and position data is interpolated, so the gap trace is an estimate with some noise, not a timing-loop measurement.
- Mini-sector winners describe one lap each. A single lap says little about a driver's general strength.
- The data comes from the unofficial F1 live timing API through FastF1. Availability and format can change, and the API only covers seasons from 2018.

## Disclaimer

This is an unofficial project and is not associated with Formula 1 companies. F1, FORMULA ONE, FORMULA 1 and related marks are trademarks of Formula One Licensing B.V.

## License

MIT, see [LICENSE](LICENSE).
