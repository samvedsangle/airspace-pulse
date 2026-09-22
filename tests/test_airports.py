import sys
import importlib
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from airports import AIRPORTS, RADIUS_KM, distance_km, nearest_airport
from ingest_flights import summarize_live_flights


def test_distance_zero_for_identical_points():
    assert distance_km(25.7959, -80.2870, 25.7959, -80.2870) == pytest.approx(0.0)


def test_scripts_package_imports_without_top_level_airports_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    sys.modules.pop("scripts.ingest_flights", None)
    sys.modules.pop("scripts.airports", None)

    module = importlib.import_module("scripts.ingest_flights")

    assert hasattr(module, "ingest_adsb_lol_data")
    assert hasattr(module, "summarize_live_flights")


def test_distance_is_symmetric():
    forward = distance_km(25.7959, -80.2870, 36.0840, -115.1537)
    backward = distance_km(36.0840, -115.1537, 25.7959, -80.2870)
    assert forward == pytest.approx(backward)


def test_distance_miami_to_las_vegas_matches_reference():
    # Great-circle distance MIA <-> LAS is ~3500 km.
    distance = distance_km(25.7959, -80.2870, 36.0840, -115.1537)
    assert distance == pytest.approx(3500, abs=150)


def test_distance_one_degree_latitude_is_about_111km():
    assert distance_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.5)


def test_nearest_airport_returns_exact_airport_at_its_coordinates():
    result = nearest_airport(25.7959, -80.2870)
    assert result is not None
    code, meta = result
    assert code == "MIA"
    assert meta["city"] == "Miami"


def test_nearest_airport_returns_none_when_outside_radius():
    # Mid-Pacific point, far from every monitored airport.
    assert nearest_airport(0.0, -150.0) is None


def test_nearest_airport_boundary_just_inside_and_outside():
    airport_code = "DEN"
    airport = AIRPORTS[airport_code]

    # A point RADIUS_KM due north of DEN should be treated as inside the radius,
    # and one slightly beyond as outside.
    inside_lat = airport["lat"] + (RADIUS_KM / 111.19) * 0.99
    outside_lat = airport["lat"] + (RADIUS_KM / 111.19) * 1.01

    assert nearest_airport(inside_lat, airport["lon"]) is not None
    assert nearest_airport(outside_lat, airport["lon"]) is None


def test_nearest_airport_respects_custom_radius():
    # Slightly offset from MIA so the distance is clearly greater than 0.
    result = nearest_airport(25.9, -80.4, radius_km=0)
    assert result is None


def test_summarize_live_flights_groups_by_airport_and_day():
    df = pd.DataFrame(
        [
            {
                "flight_id": "AA1",
                "timestamp": "2026-09-14T12:00:00Z",
                "destination_city": "Miami",
                "lat": 25.8,
                "lon": -80.3,
            },
            {
                "flight_id": "AA1",
                "timestamp": "2026-09-14T12:05:00Z",
                "destination_city": "Miami",
                "lat": 25.9,
                "lon": -80.2,
            },
            {
                "flight_id": "BB1",
                "timestamp": "2026-09-14T12:00:00Z",
                "destination_city": "Orlando",
                "lat": 28.4,
                "lon": -81.3,
            },
        ]
    )

    summary = summarize_live_flights(df)

    assert list(summary.columns) == [
        "flight_date",
        "destination_city",
        "center_lat",
        "center_lon",
        "total_aircraft",
    ]
    assert len(summary) == 2
    assert summary["flight_date"].nunique() == 1
    assert summary["flight_date"].tolist() == [
        pd.Timestamp("2026-09-14").date(),
        pd.Timestamp("2026-09-14").date(),
    ]
    assert summary.loc[summary["destination_city"] == "Miami", "total_aircraft"].iat[0] == 1
    assert summary.loc[summary["destination_city"] == "Orlando", "total_aircraft"].iat[0] == 1
