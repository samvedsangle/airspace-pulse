"""Per-airport star glyphs for multivariate cartography.

A single dot per airport hides the multi-dimensional state of that field.
Bertin/Tufte-style star (radar) glyphs render one spoke per normalised metric
directly at each airport's map location, so several metrics can be compared
across the whole network at a glance without leaving the map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scripts.airports import AIRPORTS
    from scripts.analytics import normalize_columns
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from airports import AIRPORTS
    from analytics import normalize_columns

GLYPH_RADIUS_KM = 70.0
METERS_PER_DEGREE_LAT = 111_320.0
METRIC_NAMES = (
    "activity",
    "airline_diversity",
    "airborne_share",
    "altitude_mix",
    "anomaly",
)
METRIC_DESCRIPTIONS = {
    "activity": "Distinct aircraft observed near the field.",
    "airline_diversity": "Distinct operators seen near the field.",
    "airborne_share": "Share of observations that are airborne.",
    "altitude_mix": "Mean reported altitude (normalised).",
    "anomaly": "Absolute deviation of activity from the network mean.",
}


@dataclass
class StarGlyph:
    """A single multivariate star glyph anchored at an airport."""

    airport_code: str
    destination_city: str
    center_lon: float
    center_lat: float
    metrics: dict[str, float]
    polygon: list[list[float]]
    spokes: list[list[list[float]]]
    axis_points: list[list[float]]


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = float(np.min(finite)), float(np.max(finite))
    if math.isclose(low, high):
        return np.where(np.isfinite(values), 0.5, 0.0)
    scaled = (values - low) / (high - low)
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0)


def _airport_metrics(df: pd.DataFrame) -> pd.DataFrame:
    working = normalize_columns(df)
    if "airport_code" not in working.columns or working.empty:
        return pd.DataFrame(columns=["airport_code", *METRIC_NAMES])

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    grouped = working.groupby("airport_code").agg(
        aircraft=("flight_id", "nunique"),
        airlines=("airline", lambda values: values.nunique()),
        airborne=("on_ground", lambda values: float(1 - np.mean(values))),
        mean_altitude=("altitude_ft", "mean"),
    ).reset_index()

    grouped["activity"] = _normalize(grouped["aircraft"].to_numpy(dtype=float))
    grouped["airline_diversity"] = _normalize(grouped["airlines"].to_numpy(dtype=float))
    grouped["airborne_share"] = _normalize(grouped["airborne"].to_numpy(dtype=float))
    grouped["altitude_mix"] = _normalize(grouped["mean_altitude"].to_numpy(dtype=float))

    counts = grouped["aircraft"].to_numpy(dtype=float)
    spread = float(np.std(counts)) or 1.0
    deviation = np.abs(counts - float(np.mean(counts))) / spread
    grouped["anomaly"] = _normalize(deviation)
    return grouped[["airport_code", *METRIC_NAMES]]


def _glyph_geometry(
    latitude: float, longitude: float, metrics: dict[str, float], radius_km: float
) -> tuple[list[list[float]], list[list[list[float]]], list[list[float]]]:
    """Build the star polygon, spokes, and axis vertices for one glyph."""
    axes = list(METRIC_NAMES)
    delta_lat = radius_km / (METERS_PER_DEGREE_LAT / 1000)
    delta_lon = radius_km / (
        METERS_PER_DEGREE_LAT / 1000 * max(math.cos(math.radians(latitude)), 1e-6)
    )

    axis_points: list[list[float]] = []
    polygon: list[list[float]] = []
    for index, axis in enumerate(axes):
        angle = 2 * math.pi * index / len(axes) - math.pi / 2
        value = float(metrics.get(axis, 0.0))
        vertex_lat = latitude + value * delta_lat * math.sin(angle) * -1.0
        vertex_lon = longitude + value * delta_lon * math.cos(angle)
        axis_points.append([vertex_lon, vertex_lat])
        polygon.append([vertex_lon, vertex_lat])

    # Axis vertices at full scale (the glyph's arms) for the spoke lines.
    full_points: list[list[float]] = []
    for index in range(len(axes)):
        angle = 2 * math.pi * index / len(axes) - math.pi / 2
        full_points.append(
            [
                longitude + delta_lon * math.cos(angle),
                latitude + delta_lat * math.sin(angle) * -1.0,
            ]
        )

    center = [longitude, latitude]
    spokes = [[center, point] for point in full_points]
    polygon.append(polygon[0])
    return polygon, spokes, full_points


def compute_star_glyphs(
    df: pd.DataFrame, radius_km: float = GLYPH_RADIUS_KM
) -> list[StarGlyph]:
    """Compute a star glyph for every airport present in the snapshot."""
    metrics_table = _airport_metrics(df)
    if metrics_table.empty:
        return []

    glyphs: list[StarGlyph] = []
    for row in metrics_table.itertuples():
        airport = AIRPORTS.get(row.airport_code)
        if airport is None:
            continue
        metric_values = {name: float(getattr(row, name)) for name in METRIC_NAMES}
        polygon, spokes, axis_points = _glyph_geometry(
            airport["lat"], airport["lon"], metric_values, radius_km
        )
        glyphs.append(
            StarGlyph(
                airport_code=row.airport_code,
                destination_city=airport["city"],
                center_lon=airport["lon"],
                center_lat=airport["lat"],
                metrics=metric_values,
                polygon=polygon,
                spokes=spokes,
                axis_points=axis_points,
            )
        )
    return sorted(glyphs, key=lambda glyph: glyph.metrics["activity"], reverse=True)


def glyphs_to_records(glyphs: list[StarGlyph]) -> list[dict]:
    """Serialise star glyphs for the web/map export payload."""
    return [
        {
            "airport_code": glyph.airport_code,
            "destination_city": glyph.destination_city,
            "center_lon": glyph.center_lon,
            "center_lat": glyph.center_lat,
            "metrics": glyph.metrics,
            "polygon": glyph.polygon,
            "spokes": glyph.spokes,
            "axis_points": glyph.axis_points,
        }
        for glyph in glyphs
    ]
