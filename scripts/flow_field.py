"""Continuous traffic flow field.

Instead of drawing discrete arcs between airports, aggregate every aircraft's
instantaneous heading and ground speed into a regular lat/lon grid, producing a
continuous vector field (east/north components in knots). The field is consumed
by the browser as (a) a Line Integral Convolution texture and (b) a GPU particle
advection simulation, so the airspace reads as a flowing fabric rather than a
set of isolated lines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Continental-US default extent (west, south, east, north) in degrees.
CONUS_BOUNDS = (-125.0, 24.0, -66.0, 50.0)
DEFAULT_COLUMNS = 64
DEFAULT_ROWS = 32
SPLAT_SIGMA_CELLS = 1.4
MIN_SPEED_KNOTS = 5.0


@dataclass
class FlowField:
    """A regular vector field of aggregate aircraft motion."""

    west: float
    south: float
    east: float
    north: float
    columns: int
    rows: int
    east_component: list[list[float]]
    north_component: list[list[float]]
    speed: list[list[float]]
    sample_counts: list[list[int]]

    @property
    def bounds(self) -> dict:
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }


def _kernel_offsets(sigma: float):
    """Yield (delta_row, delta_column, weight) for a Gaussian splat kernel."""
    radius = int(math.ceil(3 * sigma))
    for delta_row in range(-radius, radius + 1):
        row_weight = math.exp(-0.5 * (delta_row / sigma) ** 2)
        for delta_column in range(-radius, radius + 1):
            column_weight = math.exp(-0.5 * (delta_column / sigma) ** 2)
            yield delta_row, delta_column, row_weight * column_weight


def compute_flow_field(
    df: pd.DataFrame,
    columns: int = DEFAULT_COLUMNS,
    rows: int = DEFAULT_ROWS,
    bounds: tuple[float, float, float, float] | None = None,
) -> FlowField:
    """Aggregate aircraft heading/speed into a continuous vector field."""
    if bounds is None:
        bounds = CONUS_BOUNDS
    west, south, east, north = bounds

    weight_grid = np.zeros((rows, columns), dtype=float)
    east_grid = np.zeros((rows, columns), dtype=float)
    north_grid = np.zeros((rows, columns), dtype=float)
    count_grid = np.zeros((rows, columns), dtype=int)
    offsets = list(_kernel_offsets(SPLAT_SIGMA_CELLS))

    usable = pd.DataFrame()
    if not df.empty:
        usable = df.copy()
        for column in ("lat", "lon", "heading_deg", "speed_knots"):
            if column not in usable.columns:
                usable[column] = np.nan
        usable = usable.dropna(subset=["lat", "lon", "heading_deg", "speed_knots"])

    lon_span = (east - west) or 1.0
    lat_span = (north - south) or 1.0

    for aircraft in usable.itertuples():
        longitude = float(aircraft.lon)
        latitude = float(aircraft.lat)
        speed = float(aircraft.speed_knots)
        if not (west <= longitude <= east and south <= latitude <= north):
            continue
        if speed < MIN_SPEED_KNOTS:
            continue
        column_index = int(round((longitude - west) / lon_span * (columns - 1)))
        row_index = int(round((latitude - south) / lat_span * (rows - 1)))
        count_grid[row_index, column_index] += 1

        heading = math.radians(float(aircraft.heading_deg))
        east_speed = speed * math.sin(heading)
        north_speed = speed * math.cos(heading)
        for delta_row, delta_column, kernel in offsets:
            target_row = row_index + delta_row
            target_column = column_index + delta_column
            if 0 <= target_row < rows and 0 <= target_column < columns:
                weight_grid[target_row, target_column] += kernel
                east_grid[target_row, target_column] += kernel * east_speed
                north_grid[target_row, target_column] += kernel * north_speed

    with np.errstate(invalid="ignore", divide="ignore"):
        east_component = np.where(weight_grid > 0, east_grid / weight_grid, 0.0)
        north_component = np.where(weight_grid > 0, north_grid / weight_grid, 0.0)
    speed_grid = np.sqrt(east_component**2 + north_component**2)

    return FlowField(
        west=west,
        south=south,
        east=east,
        north=north,
        columns=columns,
        rows=rows,
        east_component=east_component.tolist(),
        north_component=north_component.tolist(),
        speed=speed_grid.tolist(),
        sample_counts=count_grid.tolist(),
    )


def flow_to_payload(flow: FlowField) -> dict:
    """Serialise the flow field for the browser as flat typed arrays."""
    flat_east: list[float] = []
    flat_north: list[float] = []
    flat_speed: list[float] = []
    flat_counts: list[int] = []
    for row in range(flow.rows):
        for column in range(flow.columns):
            flat_east.append(round(flow.east_component[row][column], 3))
            flat_north.append(round(flow.north_component[row][column], 3))
            flat_speed.append(round(flow.speed[row][column], 3))
            flat_counts.append(int(flow.sample_counts[row][column]))
    return {
        "bounds": flow.bounds,
        "columns": flow.columns,
        "rows": flow.rows,
        "east": flat_east,
        "north": flat_north,
        "speed": flat_speed,
        "counts": flat_counts,
        "max_speed": max(flat_speed) if flat_speed else 0.0,
    }
