"""Re-enrich already-collected bronze_flights.parquet with better metadata.

Two data sources referenced by scripts.metadata and scripts.aircraft_registry
were previously unavailable when this project's earlier snapshots were
ingested: data/airlines.dat (OpenFlights) hadn't been downloaded, so every
callsign-based airline lookup fell through to "Unidentified operator", and
scripts/aircraft_registry.py (the local OpenSky aircraft database) didn't
exist yet, so 88%+ of non-airline aircraft had no registration/type at all.

Re-running ingestion only fixes *future* snapshots. This script re-resolves
airline/registration/type/manufacturer for every row already on disk, using
whichever of those two sources are now present, without re-querying ADSB.lol
or hexdb.io — a pure local join, safe to re-run any time either reference
file is refreshed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aircraft_registry import load_registry, lookup_aircraft
from scripts.metadata import NON_AIRLINE_LABELS, load_openflights_prefixes, resolve_callsign_airline


def _load_hexdb_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def backfill(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    airline_prefixes = load_openflights_prefixes(PROJECT_ROOT / "data" / "airlines.dat")
    registry = load_registry(PROJECT_ROOT / "data" / "opensky_aircraft_database.csv")
    hexdb_cache = _load_hexdb_cache(PROJECT_ROOT / "data" / "hexdb_cache.json")

    for column in ("wake_category_description", "manufacturer", "registration", "aircraft_type"):
        if column not in df.columns:
            df[column] = None

    def is_missing(value) -> bool:
        # A plain `not value` is wrong here: pandas represents a missing
        # string cell as float('nan'), and bool(float('nan')) is True, so
        # truthy checks silently treat "missing" as "present" and never fire.
        return value is None or (isinstance(value, float) and pd.isna(value)) or value == ""

    airline_updates = registration_updates = type_updates = manufacturer_updates = 0

    for index, row in df.iterrows():
        icao_hex = str(row.get("icao_hex") or "").strip().lower()
        registry_record = lookup_aircraft(icao_hex, registry) if icao_hex else {}
        hex_record = hexdb_cache.get(icao_hex, {}) if icao_hex else {}

        current_airline = row.get("airline")
        if current_airline in NON_AIRLINE_LABELS or is_missing(current_airline):
            resolved = resolve_callsign_airline(str(row.get("flight_id") or ""), airline_prefixes)
            if resolved not in NON_AIRLINE_LABELS:
                df.at[index, "airline"] = resolved
                airline_updates += 1
            elif registry_record.get("operator") or registry_record.get("owner"):
                df.at[index, "airline"] = registry_record.get("operator") or registry_record.get("owner")
                airline_updates += 1

        if is_missing(row.get("registration")):
            registration = registry_record.get("registration") or hex_record.get("Registration")
            if registration:
                df.at[index, "registration"] = registration
                registration_updates += 1

        if is_missing(row.get("aircraft_type")):
            aircraft_type = (
                registry_record.get("model") or registry_record.get("typecode") or hex_record.get("Type")
            )
            if aircraft_type:
                df.at[index, "aircraft_type"] = aircraft_type
                type_updates += 1

        if is_missing(row.get("manufacturer")) and registry_record.get("manufacturer"):
            df.at[index, "manufacturer"] = registry_record["manufacturer"]
            manufacturer_updates += 1

        if is_missing(row.get("wake_category_description")) and registry_record.get("category_description"):
            df.at[index, "wake_category_description"] = registry_record["category_description"]

    print(
        f"Backfilled {airline_updates} airlines, {registration_updates} registrations, "
        f"{type_updates} aircraft types, {manufacturer_updates} manufacturers "
        f"across {len(df)} rows."
    )
    return df


if __name__ == "__main__":
    target = PROJECT_ROOT / "data" / "bronze_flights.parquet"
    if not target.exists():
        raise SystemExit(f"No parquet file found at {target}")
    enriched = backfill(target)
    enriched.to_parquet(target, engine="pyarrow", index=False)
    print(f"Wrote enriched data back to {target}.")
