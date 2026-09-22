import argparse
import gzip
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from airports import build_aircraft_record, nearest_airport
else:
    from scripts.airports import build_aircraft_record, nearest_airport


def read_trace(stream):
    with gzip.open(stream, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def iter_trace_files(archive_path):
    if archive_path == "-":
        archive = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
        try:
            for member in archive:
                if member.isfile() and member.name.endswith(".json.gz"):
                    stream = archive.extractfile(member)
                    if stream is not None:
                        yield stream
        finally:
            archive.close()
        return

    trace_files = sorted(Path(archive_path).rglob("*.json.gz"))
    if not trace_files:
        raise FileNotFoundError(
            f"No .json.gz trace files found under {archive_path}. "
            "Extract an ADSB.lol daily release first."
        )
    yield from trace_files


def write_chunk(records, output_path, chunk_number):
    chunk_path = Path(output_path).with_name(
        f"{Path(output_path).stem}_part{chunk_number:04d}.parquet"
    )
    pd.DataFrame(records).drop_duplicates(
        subset=["flight_id", "timestamp", "lat", "lon"]
    ).to_parquet(chunk_path, engine="pyarrow", index=False)


def ingest_history(archive_path, output_path):
    records = []
    chunk_number = 0
    total_records = 0
    chunk_size = 10_000
    for trace_path in iter_trace_files(archive_path):
        try:
            aircraft = read_trace(trace_path)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping unreadable trace {trace_path}: {error}")
            continue

        base_timestamp = aircraft.get("timestamp")
        if not base_timestamp:
            continue

        for point in aircraft.get("trace", []):
            if len(point) < 3 or point[1] is None or point[2] is None:
                continue
            try:
                latitude, longitude = float(point[1]), float(point[2])
            except (TypeError, ValueError):
                continue
            nearby_airport = nearest_airport(latitude, longitude)
            if nearby_airport is None:
                continue

            airport_code, airport = nearby_airport
            records.append(
                build_aircraft_record(
                    flight_id=(aircraft.get("flight") or aircraft.get("icao") or "UNKNOWN"),
                    airline=(
                        aircraft.get("ownOp")
                        or aircraft.get("operator")
                        or "Unknown airline"
                    ),
                    timestamp=pd.to_datetime(base_timestamp + point[0], unit="s", utc=True),
                    airport_code=airport_code,
                    airport=airport,
                    latitude=latitude,
                    longitude=longitude,
                    on_ground=point[3] == "ground",
                    aircraft_category=aircraft.get("category") or "Unknown",
                )
            )

            if len(records) >= chunk_size:
                write_chunk(records, output_path, chunk_number)
                total_records += len(records)
                chunk_number += 1
                records = []

    if not records:
        if total_records == 0:
            raise RuntimeError(
                "No aircraft trace points intersected the monitored airports"
            )
    else:
        write_chunk(records, output_path, chunk_number)
        total_records += len(records)
    print(f"Success! {total_records} historical aircraft points saved as Parquet chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process extracted ADSB.lol history")
    parser.add_argument("archive_path", help="Extracted directory or '-' for a tar stream")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "bronze_flights.parquet"),
    )
    args = parser.parse_args()
    ingest_history(args.archive_path, args.output)