"""Local aircraft metadata registry, sourced from the OpenSky Network database.

Previously, registration/manufacturer/model/operator for any aircraft with no
broadcast callsign-based airline came from a per-hex, on-demand call to
hexdb.io (scripts.metadata.lookup_hexdb) at ingestion time. That has a low
fill rate in practice — one HTTP round trip per unknown aircraft, subject to
timeouts, rate limits, and 404s for anything hexdb's smaller index doesn't
carry — and was quietly leaving ~88% of "unknown operator" aircraft with no
registration or type at all.

The OpenSky Network publishes a free, complete snapshot of the global
aircraft register (~520k tail numbers) as a downloadable CSV, keyed by the
ICAO24 hex address every ADS-B message already carries. Joining against a
local copy of that file is instant, works offline, and covers essentially
every registered aircraft — a strictly more accurate and more available data
source than per-request hexdb.io lookups, which are now used only as a
last-resort fallback for the (rare) hex codes this file doesn't have.

Download once with:
    curl -L -o data/opensky_aircraft_database.csv \\
        https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv
"""
from __future__ import annotations

import csv
from pathlib import Path
from urllib.request import Request, urlopen

OPENSKY_DATABASE_URL = "https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv"

REGISTRY_COLUMNS = [
    "icao24",
    "registration",
    "manufacturername",
    "model",
    "typecode",
    "operator",
    "operatorcallsign",
    "owner",
    "categoryDescription",
]

# OpenSky's categoryDescription is the same ADS-B emitter-category taxonomy
# broadcast live (scripts.wake_vortex.CATEGORY_WAKE), just always-present
# (looked up from the registered airframe) instead of only-when-transmitted.
WAKE_CLASS_FROM_DESCRIPTION = {
    "Light (< 15500 lbs)": ("Light", 0.35),
    "Small (15500 to 75000 lbs)": ("Small", 0.55),
    "Large (75000 to 300000 lbs)": ("Large", 0.75),
    "High Vortex Large (aircraft such as B-757)": ("Large", 0.85),
    "Heavy (> 300000 lbs)": ("Heavy", 1.0),
    "Rotorcraft": ("Rotorcraft", 0.4),
    "Glider / sailplane": ("Light", 0.2),
    "Ultralight / hang-glider / paraglider": ("Light", 0.15),
}


def download_opensky_database(destination: Path, timeout: int = 60) -> Path:
    """Fetch the ~95MB OpenSky aircraft database CSV.

    Not committed to git (it's gitignored under data/, and 95MB is a poor
    fit for a repo that gets cloned on every deploy) — a fresh checkout
    (e.g. a Streamlit Community Cloud deploy) calls this once at startup
    instead. The registry degrades to empty (see load_registry) if this
    hasn't run yet or fails, so a slow/blocked download never crashes the
    dashboard, just leaves aircraft type/registration less complete.
    """
    request = Request(OPENSKY_DATABASE_URL, headers={"User-Agent": "airspace-pulse/1.0"})
    with urlopen(request, timeout=timeout) as response:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.read())
    return destination


def load_registry(csv_path: Path) -> dict[str, dict]:
    """Load the OpenSky aircraft database into an in-memory icao24 -> record map.

    Returns {} if the file hasn't been downloaded yet, so callers degrade the
    same way they did before this registry existed (falling through to
    hexdb.io / "Unknown").
    """
    if not csv_path.exists():
        return {}

    registry: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            icao24 = (row.get("icao24") or "").strip().lower()
            if not icao24:
                continue
            registry[icao24] = {
                "registration": (row.get("registration") or "").strip() or None,
                "manufacturer": (row.get("manufacturername") or "").strip() or None,
                "model": (row.get("model") or "").strip() or None,
                "typecode": (row.get("typecode") or "").strip() or None,
                "operator": (row.get("operator") or row.get("operatorcallsign") or "").strip() or None,
                "owner": (row.get("owner") or "").strip() or None,
                "category_description": (row.get("categoryDescription") or "").strip() or None,
            }
    return registry


def lookup_aircraft(icao24_hex: str, registry: dict[str, dict]) -> dict:
    """Look up one aircraft's registry record, or {} if it isn't in the file."""
    normalized = (icao24_hex or "").strip().lower()
    return registry.get(normalized, {})


def wake_class_from_registry(record: dict) -> tuple[str, float] | None:
    """Map a registry record's category description to a wake-turbulence class."""
    description = record.get("category_description")
    if not description:
        return None
    return WAKE_CLASS_FROM_DESCRIPTION.get(description)
