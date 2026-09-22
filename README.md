# Airspace Pulse — Live Mobility Analyzer

A live flight-tracking dashboard built with [Streamlit](https://streamlit.io). It polls [ADSB.lol](https://adsb.lol) for aircraft near ~50 major US airports, enriches each one with real airline/aircraft-type data from OpenFlights and the OpenSky Network, and renders it as a live map plus a set of analytics and experimental visualizations.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads two small/medium reference datasets automatically (OpenFlights airline codes, the OpenSky aircraft database) — no manual setup needed.

## What's in here

- `app.py` — the dashboard
- `scripts/ingest_flights.py` — pulls live positions from ADSB.lol
- `scripts/aircraft_registry.py` / `scripts/metadata.py` — airline/aircraft enrichment
- `scripts/` (the rest) — each analytics/visualization feature is its own small module; see their docstrings for what they do and, where a technique is more research-flavored than production-grade, an honest note on the tradeoff

## Keeping data fresh

The dashboard refreshes itself while someone has it open in a browser. For continuous background collection independent of viewers, schedule `scripts/ingest_flights.py` to run every few minutes (e.g. via cron) on any always-on machine.

Note: on ephemeral hosting (like Streamlit Community Cloud's free tier), collected history resets whenever the app's container restarts — there's no persistent disk. It self-heals by re-collecting on the next visit, but it isn't a durable historical archive without adding external storage.
