"""Wake-vortex trail geometry from real aircraft state.

Every aircraft sheds a pair of counter-rotating trailing vortices whose
circulation strength scales with weight and inversely with speed and
wingspan, and which decay roughly exponentially with time/age in the free
atmosphere. ICAO's wake-turbulence categories (Light/Small/Large/Heavy) are
exactly a coarse proxy for that circulation strength, and the ADS-B emitter
"category" field we already ingest maps onto them.

We do not have real wind fields or an aircraft weight/wingspan database, so
this is deliberately a *simplified* physics-flavoured visualization, not a
CFD-grade wake model: circulation strength comes from the ICAO category
lookup, decay follows the standard exponential form

    Gamma(t) = Gamma_0 * exp(-t / tau)

and the trail is drawn as a chain of shrinking, fading discs behind the
aircraft's reciprocal heading, spaced by how far the aircraft actually
travelled between "puffs" (ground speed * dt).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

METERS_PER_DEGREE_LAT = 111_320.0
KNOTS_TO_MPS = 0.514444
DECAY_TAU_SECONDS = 100.0
TRAIL_STEPS = 10
STEP_SECONDS = 4.0
MIN_SPEED_KNOTS = 40.0

# ADS-B emitter "category" -> (ICAO wake class, relative circulation strength).
CATEGORY_WAKE = {
    "A1": ("Light", 0.35),
    "A2": ("Small", 0.55),
    "A3": ("Large", 0.75),
    "A4": ("Large", 0.8),
    "A5": ("Heavy", 1.0),
    "A6": ("Heavy", 1.0),
    "A7": ("Rotorcraft", 0.4),
}
DEFAULT_WAKE = ("Unknown", 0.6)


def meters_per_degree_lon(latitude: float) -> float:
    return METERS_PER_DEGREE_LAT * max(math.cos(math.radians(latitude)), 1e-6)


@dataclass
class WakeTrail:
    flight_id: str
    airline: str
    wake_class: str
    puffs: list[dict]


def wake_class_for(category: str | None, category_description: str | None = None) -> tuple[str, float]:
    """Resolve a wake class, preferring the live ADS-B emitter category.

    Many aircraft (especially GA) never transmit an emitter category over
    ADS-B, in which case `category` is missing/"Unknown" — the registered
    airframe's category from scripts.aircraft_registry (looked up by ICAO24,
    always present for anything in the OpenSky database) is then used
    instead of silently defaulting to the generic "Unknown" wake class.
    """
    live = CATEGORY_WAKE.get((category or "").upper())
    if live is not None:
        return live
    if category_description:
        try:
            from scripts.aircraft_registry import WAKE_CLASS_FROM_DESCRIPTION
        except ModuleNotFoundError:  # pragma: no cover - direct script fallback
            from aircraft_registry import WAKE_CLASS_FROM_DESCRIPTION

        registry_class = WAKE_CLASS_FROM_DESCRIPTION.get(category_description)
        if registry_class is not None:
            return registry_class
    return DEFAULT_WAKE


def _trail_for_row(row) -> WakeTrail | None:
    speed_knots = row.speed_knots
    heading_deg = row.heading_deg
    if pd.isna(speed_knots) or pd.isna(heading_deg) or speed_knots < MIN_SPEED_KNOTS:
        return None

    wake_class, circulation0 = wake_class_for(
        getattr(row, "aircraft_category", None),
        getattr(row, "wake_category_description", None),
    )
    speed_mps = float(speed_knots) * KNOTS_TO_MPS
    reciprocal_heading = math.radians((float(heading_deg) + 180) % 360)
    meters_lon = meters_per_degree_lon(float(row.lat))
    altitude_m = (float(row.altitude_ft) * 0.3048) if pd.notna(getattr(row, "altitude_ft", None)) else 0.0

    puffs = []
    lat, lon = float(row.lat), float(row.lon)
    for step in range(1, TRAIL_STEPS + 1):
        age_seconds = step * STEP_SECONDS
        distance_m = speed_mps * STEP_SECONDS
        lat += (distance_m * math.cos(reciprocal_heading)) / METERS_PER_DEGREE_LAT
        lon += (distance_m * math.sin(reciprocal_heading)) / meters_lon
        circulation = circulation0 * math.exp(-age_seconds / DECAY_TAU_SECONDS)
        puffs.append(
            {
                "lat": lat,
                "lon": lon,
                "altitude_m": altitude_m,
                "age_seconds": age_seconds,
                "circulation": circulation,
                "radius_m": 25.0 + 60.0 * circulation,
            }
        )
    return WakeTrail(
        flight_id=str(row.flight_id),
        airline=str(getattr(row, "airline", "Unknown airline")),
        wake_class=wake_class,
        puffs=puffs,
    )


def compute_wake_trails(df: pd.DataFrame, max_aircraft: int = 200) -> list[WakeTrail]:
    """Build decaying wake-vortex trails for the fastest-moving aircraft in the snapshot."""
    if df.empty:
        return []
    working = df.dropna(subset=["lat", "lon"]).copy()
    if "speed_knots" not in working or "heading_deg" not in working:
        return []
    working = working.sort_values("speed_knots", ascending=False).head(max_aircraft)

    trails = []
    for row in working.itertuples():
        trail = _trail_for_row(row)
        if trail is not None:
            trails.append(trail)
    return trails


def trails_to_frame(trails: list[WakeTrail]) -> pd.DataFrame:
    """Flatten wake trails into one row per puff, for a pydeck ScatterplotLayer."""
    rows = []
    for trail in trails:
        for puff in trail.puffs:
            intensity = puff["circulation"]
            rows.append(
                {
                    "flight_id": trail.flight_id,
                    "airline": trail.airline,
                    "wake_class": trail.wake_class,
                    "lat": puff["lat"],
                    "lon": puff["lon"],
                    "altitude_m": puff["altitude_m"],
                    "age_seconds": puff["age_seconds"],
                    "radius_m": puff["radius_m"],
                    "fill_color": [
                        int(255 * min(intensity * 1.4, 1.0)),
                        int(140 + 100 * (1 - intensity)),
                        255,
                        int(180 * intensity),
                    ],
                }
            )
    return pd.DataFrame(rows)
