"""Uncertainty-aware visualisation helpers.

Turns model uncertainty into honest, renderable geometry: a Kalman filter's
position-covariance matrix becomes a confidence ellipse, and a forecast's
residual spread becomes a fan chart of credible intervals. Instead of drawing
every aircraft as a dimensionless dot, the map shows how confident the track
actually is.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

METERS_PER_DEGREE_LAT = 111_320.0
DEFAULT_CONFIDENCE = 0.95

# Chi-square quantiles for 2 degrees of freedom. Multiplying the one-sigma
# standard-deviation ellipse by these scales turns it into an ellipsoidal
# confidence region of the given coverage.
CHI2_SCALE = {0.5: 1.1774, 0.8: 1.7941, 0.95: 2.4477, 0.99: 3.0349}

UNCERTAINTY_COLUMNS = [
    "timestamp",
    "filtered_lat",
    "filtered_lon",
    "semi_major_m",
    "semi_minor_m",
    "orientation_deg",
    "confidence_radius_m",
    "confidence",
]


def meters_per_degree_lon(latitude: float) -> float:
    """Length of one degree of longitude at a given latitude, in metres."""
    return METERS_PER_DEGREE_LAT * max(math.cos(math.radians(latitude)), 1e-6)


def covariance_to_ellipse(
    cov_lat_deg2: float,
    cov_lon_deg2: float,
    cov_latlon_deg2: float,
    latitude: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float, float]:
    """Convert a lat/lon covariance (deg^2) to ellipse axes in metres.

    Returns ``(semi_major_m, semi_minor_m, orientation_deg)`` where the
    orientation is the compass bearing of the semi-major axis.
    """
    meters_lat = METERS_PER_DEGREE_LAT
    meters_lon = meters_per_degree_lon(latitude)
    cov = np.array(
        [
            [cov_lat_deg2 * meters_lat * meters_lat, cov_latlon_deg2 * meters_lat * meters_lon],
            [cov_latlon_deg2 * meters_lat * meters_lon, cov_lon_deg2 * meters_lon * meters_lon],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0, 0.0
    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scale = CHI2_SCALE.get(confidence, CHI2_SCALE[DEFAULT_CONFIDENCE])
    semi_major = float(math.sqrt(eigenvalues[0]) * scale)
    semi_minor = float(math.sqrt(eigenvalues[1]) * scale)
    major_vector = eigenvectors[:, 0]
    orientation = math.degrees(math.atan2(major_vector[0], major_vector[1])) % 360
    return semi_major, semi_minor, orientation


def ellipse_ring(
    latitude: float,
    longitude: float,
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    segments: int = 40,
) -> list[list[float]]:
    """Return a closed ``[lon, lat]`` ring for a rotated confidence ellipse."""
    ring: list[list[float]] = []
    orientation = math.radians(orientation_deg)
    meters_lon = meters_per_degree_lon(latitude)
    for step in range(segments):
        theta = 2 * math.pi * step / segments
        north = semi_major_m * math.sin(theta)
        east = semi_minor_m * math.cos(theta)
        rotated_north = north * math.cos(orientation) + east * math.sin(orientation)
        rotated_east = -north * math.sin(orientation) + east * math.cos(orientation)
        ring.append(
            [
                longitude + rotated_east / meters_lon,
                latitude + rotated_north / METERS_PER_DEGREE_LAT,
            ]
        )
    ring.append(ring[0])
    return ring


def trajectory_uncertainty(
    smoothed: pd.DataFrame, confidence: float = DEFAULT_CONFIDENCE
) -> pd.DataFrame:
    """Build confidence ellipses for a Kalman-smoothed trajectory."""
    required = {"cov_lat_deg2", "cov_lon_deg2", "cov_latlon_deg2", "filtered_lat"}
    if smoothed.empty or not required.issubset(smoothed.columns):
        return pd.DataFrame(columns=UNCERTAINTY_COLUMNS + ["ring"])

    rows = []
    for point in smoothed.itertuples():
        semi_major, semi_minor, orientation = covariance_to_ellipse(
            point.cov_lat_deg2,
            point.cov_lon_deg2,
            point.cov_latlon_deg2,
            point.filtered_lat,
            confidence,
        )
        rows.append(
            {
                "timestamp": point.timestamp,
                "filtered_lat": point.filtered_lat,
                "filtered_lon": point.filtered_lon,
                "semi_major_m": semi_major,
                "semi_minor_m": semi_minor,
                "orientation_deg": orientation,
                "confidence_radius_m": semi_major,
                "confidence": confidence,
                "ring": ellipse_ring(
                    point.filtered_lat,
                    point.filtered_lon,
                    semi_major,
                    semi_minor,
                    orientation,
                ),
            }
        )
    return pd.DataFrame(rows)
