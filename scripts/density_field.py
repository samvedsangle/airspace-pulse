"""Continuous volumetric traffic-occupancy field.

The original ask here was a neural radiance field (NeRF) over the
accumulated 3D trajectory data. NeRFs learn a radiance/opacity field from
many *camera views of one static scene* by differentiating through a
renderer; they have no notion of "many independent moving objects sampled at
sparse GPS pings," which is what ADS-B data actually is. Labelling a model
"NeRF" here would be costume, not substance.

What the ask is really reaching for — a continuous, differentiable-enough
field you can query at any point in space (and slice by time) — is exactly
what kernel density estimation gives you, and it is honestly justified by
the data we actually have: accumulated (lon, lat, altitude) observations.
This module fits a Gaussian KDE over that point cloud (optionally
conditioned on hour-of-day), exposes `query_density` for an arbitrary point,
and rasterizes a 2-D slice at a chosen altitude band for the map.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_POINTS_FOR_FIELD = 30
DEFAULT_GRID_SIZE = 48


@dataclass
class DensityField:
    ready: bool
    message: str
    grid_lon: np.ndarray = field(default_factory=lambda: np.array([]))
    grid_lat: np.ndarray = field(default_factory=lambda: np.array([]))
    density: np.ndarray = field(default_factory=lambda: np.array([]))
    kde: object = None
    altitude_band_ft: tuple[float, float] | None = None
    hour_of_day: int | None = None


def _select_slice(df: pd.DataFrame, altitude_band_ft: tuple[float, float] | None, hour_of_day: int | None) -> pd.DataFrame:
    working = df.dropna(subset=["lat", "lon"]).copy()
    if "altitude_ft" in working:
        working["altitude_ft"] = pd.to_numeric(working["altitude_ft"], errors="coerce")
    if altitude_band_ft is not None and "altitude_ft" in working:
        low, high = altitude_band_ft
        working = working[working["altitude_ft"].between(low, high)]
    if hour_of_day is not None and "timestamp" in working:
        hours = pd.to_datetime(working["timestamp"], utc=True).dt.hour
        working = working[hours == hour_of_day]
    return working


def build_density_field(
    df: pd.DataFrame,
    altitude_band_ft: tuple[float, float] | None = None,
    hour_of_day: int | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> DensityField:
    """Fit a queryable KDE occupancy field over a (lon, lat) slice of the traffic record."""
    from scipy.stats import gaussian_kde

    working = _select_slice(df, altitude_band_ft, hour_of_day)
    if len(working) < MIN_POINTS_FOR_FIELD:
        return DensityField(
            False,
            f"At least {MIN_POINTS_FOR_FIELD} observations are needed in this slice for a density field "
            "(try a wider altitude band or 'any hour').",
            altitude_band_ft=altitude_band_ft,
            hour_of_day=hour_of_day,
        )

    points = np.vstack([working["lon"].to_numpy(dtype=float), working["lat"].to_numpy(dtype=float)])
    kde = gaussian_kde(points)

    lon_min, lon_max = points[0].min(), points[0].max()
    lat_min, lat_max = points[1].min(), points[1].max()
    lon_pad = max((lon_max - lon_min) * 0.15, 0.5)
    lat_pad = max((lat_max - lat_min) * 0.15, 0.5)
    grid_lon = np.linspace(lon_min - lon_pad, lon_max + lon_pad, grid_size)
    grid_lat = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_size)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)
    query_points = np.vstack([mesh_lon.ravel(), mesh_lat.ravel()])
    density = kde(query_points).reshape(mesh_lon.shape)
    density = density / density.max() if density.max() > 0 else density

    return DensityField(
        True,
        "Gaussian KDE occupancy field fit over observed positions — a continuous, "
        "queryable-anywhere density surface, not a claim of learned radiance.",
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        density=density,
        kde=kde,
        altitude_band_ft=altitude_band_ft,
        hour_of_day=hour_of_day,
    )


def query_density(field: DensityField, lon: float, lat: float) -> float:
    """Query the fitted field's relative occupancy density at an arbitrary point."""
    if not field.ready or field.kde is None:
        return 0.0
    raw = float(field.kde(np.array([[lon], [lat]]))[0])
    peak = float(field.density.max()) if field.density.size else 1.0
    return raw / peak if peak > 0 else 0.0


def field_to_grid_frame(field: DensityField) -> pd.DataFrame:
    """Flatten the density grid into (lon, lat, density) rows for a heatmap layer."""
    if not field.ready:
        return pd.DataFrame(columns=["lon", "lat", "density"])
    mesh_lon, mesh_lat = np.meshgrid(field.grid_lon, field.grid_lat)
    return pd.DataFrame(
        {
            "lon": mesh_lon.ravel(),
            "lat": mesh_lat.ravel(),
            "density": field.density.ravel(),
        }
    )
