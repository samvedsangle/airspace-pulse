import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for candidate in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from scripts.aircraft_registry import load_registry, lookup_aircraft
    from scripts.airports import AIRPORTS, distance_km, nearest_airport
    from scripts.metadata import NON_AIRLINE_LABELS, load_openflights_prefixes, lookup_hexdb, resolve_callsign_airline
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from aircraft_registry import load_registry, lookup_aircraft
    from airports import AIRPORTS, distance_km, nearest_airport
    from metadata import NON_AIRLINE_LABELS, load_openflights_prefixes, lookup_hexdb, resolve_callsign_airline

SEARCH_RADIUS_NM = 100
# Querying 50 airports with too much concurrency reliably draws HTTP 429s
# from ADSB.lol's free tier — a run with FETCH_WORKERS=6 and no retry
# silently lost 33 of 50 airports (66% of the network) to rate limiting in a
# single cycle. Lower concurrency plus one short retry recovers genuinely
# transient blips. Deliberately NOT an aggressive multi-retry backoff: a
# *sustained* rate limit (this whole IP throttled for minutes, not one
# request) doesn't get better within a single cycle, and a first attempt at
# 4 retries with exponential backoff made one ingestion run take 12+ minutes
# under sustained throttling — worse than just letting the next 5-minute
# cron cycle pick up whatever this one missed.
FETCH_WORKERS = 3
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 2.0


def fetch_aircraft(latitude, longitude):
    url = (
        f"https://api.adsb.lol/v2/lat/{latitude}/lon/{longitude}"
        f"/dist/{SEARCH_RADIUS_NM}"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "real-mobility-analyzer/1.0",
        },
    )
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=20) as response:
                return json.loads(response.read())
        except HTTPError as error:
            if error.code == 429 and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                continue
            raise


def summarize_live_flights(df):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "flight_date",
                "destination_city",
                "center_lat",
                "center_lon",
                "total_aircraft",
            ]
        )

    working = df.copy()
    working["flight_date"] = pd.to_datetime(working["timestamp"], utc=True).dt.date
    summary = (
        working.groupby(["flight_date", "destination_city"], as_index=False)
        .agg(
            center_lat=("lat", "mean"),
            center_lon=("lon", "mean"),
            total_aircraft=("flight_id", "nunique"),
        )
        .sort_values(["flight_date", "total_aircraft"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def fetch_airport_snapshot(item):
    airport_code, airport = item
    try:
        return airport_code, fetch_aircraft(airport["lat"], airport["lon"]), None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        return airport_code, None, error


def ingest_adsb_lol_data():
    print("Loading live aircraft data from ADSB.lol...")
    aircraft_by_hex = {}
    airline_prefixes = load_openflights_prefixes(PROJECT_ROOT / "data" / "airlines.dat")
    metadata_cache = PROJECT_ROOT / "data" / "hexdb_cache.json"
    registry = load_registry(PROJECT_ROOT / "data" / "opensky_aircraft_database.csv")

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = [executor.submit(fetch_airport_snapshot, item) for item in AIRPORTS.items()]
        for future in as_completed(futures):
            airport_code, payload, error = future.result()
            if error:
                print(f"Could not load {airport_code}: {error}")
                continue

            source_timestamp = payload.get("now", time.time())
            timestamp_unit = "ms" if source_timestamp > 100_000_000_000 else "s"
            for aircraft in payload.get("ac", []):
                latitude = aircraft.get("lat")
                longitude = aircraft.get("lon")
                aircraft_hex = aircraft.get("hex")
                if not aircraft_hex or latitude is None or longitude is None:
                    continue

                nearby = nearest_airport(latitude, longitude)
                if nearby is None:
                    continue

                nearest_code, nearest_meta = nearby
                flight_id = (aircraft.get("flight") or aircraft_hex).strip()
                operator = (aircraft.get("ownOp") or aircraft.get("operator") or "").strip()
                if not operator:
                    operator = resolve_callsign_airline(flight_id, airline_prefixes)

                # The local OpenSky registry (520k tail numbers, no network
                # round trip) is tried first; hexdb.io is now only a
                # last-resort fallback for the hex codes it doesn't carry.
                registry_record = lookup_aircraft(aircraft_hex, registry)
                needs_fallback = operator in NON_AIRLINE_LABELS and not registry_record.get("registration")
                hex_metadata = lookup_hexdb(aircraft_hex, metadata_cache) if needs_fallback else {}

                resolved_airline = (
                    operator
                    or registry_record.get("operator")
                    or registry_record.get("owner")
                    or hex_metadata.get("RegisteredOwners")
                    or "Unknown airline"
                )
                aircraft_by_hex[aircraft_hex] = {
                    "flight_id": flight_id,
                    "icao_hex": aircraft_hex,
                    "airline": resolved_airline,
                    "timestamp": pd.to_datetime(source_timestamp, unit=timestamp_unit, utc=True),
                    "destination_city": nearest_meta["city"],
                    "lat": float(latitude),
                    "lon": float(longitude),
                    "airport_code": nearest_code,
                    "on_ground": bool(aircraft.get("on_ground", False)),
                    "altitude_ft": aircraft.get("alt_baro") if isinstance(aircraft.get("alt_baro"), (int, float)) else None,
                    "speed_knots": aircraft.get("gs"),
                    "heading_deg": aircraft.get("track"),
                    "squawk": str(aircraft.get("squawk") or "").strip(),
                    "aircraft_category": aircraft.get("category") or "Unknown",
                    "wake_category_description": registry_record.get("category_description"),
                    "registration": registry_record.get("registration") or hex_metadata.get("Registration"),
                    "aircraft_type": (
                        registry_record.get("model")
                        or registry_record.get("typecode")
                        or hex_metadata.get("Type")
                    ),
                    "manufacturer": registry_record.get("manufacturer"),
                }

    if not aircraft_by_hex:
        raise RuntimeError("ADSB.lol returned no aircraft with valid positions")

    output_path = PROJECT_ROOT / "data" / "bronze_flights.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(aircraft_by_hex.values())
    final_df = _merge_and_write_atomically(output_path, new_df)
    print(f"Success! {len(final_df)} live aircraft snapshots saved to {output_path}.")
    return final_df


def dedupe_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse near-duplicate observations of the same aircraft to one per minute.

    Each ADSB.lol response carries its own server timestamp, so refreshing
    the dashboard more than once inside a minute — an eager auto-refresh
    cycle, multiple open tabs, someone mashing "Refresh now" — produces rows
    a few seconds apart with distinct millisecond timestamps. An exact
    (flight_id, timestamp, airport_code) dedup key never catches those, so
    they silently accumulate as if they were independent minutes-apart
    samples: hourly counts get inflated, and pairwise analyses (e.g.
    scripts.advanced_analytics.wake_separation_alerts) can even match an
    aircraft against its own near-identical snapshot as a false "encounter".
    Keying on the icao hex, airport, and the timestamp floored to the minute
    keeps at most one (the latest) observation per aircraft per minute.
    """
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["_dedup_minute"] = working["timestamp"].dt.floor("min")
    dedup_key = ["icao_hex", "airport_code", "_dedup_minute"] if "icao_hex" in working else ["flight_id", "airport_code", "_dedup_minute"]
    working = working.sort_values("timestamp").drop_duplicates(subset=dedup_key, keep="last")
    return working.drop(columns=["_dedup_minute"]).reset_index(drop=True)


def _merge_and_write_atomically(output_path: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Read-modify-write the bronze parquet under a lock, with an atomic rename.

    Streamlit can trigger ingestion from several places at once (auto-refresh,
    the manual refresh button, multiple open tabs), so without a lock two
    concurrent runs can race on the read-then-overwrite and silently drop one
    run's data.
    """
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX fallback
            pass
        try:
            final_df = new_df
            if output_path.exists():
                previous_df = pd.read_parquet(output_path)
                final_df = pd.concat([previous_df, new_df], ignore_index=True)
                final_df = dedupe_snapshots(final_df)
            tmp_path = output_path.with_suffix(f".tmp{os.getpid()}.parquet")
            final_df.to_parquet(tmp_path, engine="pyarrow", index=False)
            os.replace(tmp_path, output_path)
            return final_df
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                pass


if __name__ == "__main__":
    ingest_adsb_lol_data()
