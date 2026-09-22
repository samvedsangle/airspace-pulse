"""FAA-style controlled-airspace volume model (approximate).

Class B/C/D airspace is modelled as real translucent 3D volumes (stacked
cylinders per published airspace structure) rather than an abstract radius
circle, so aircraft can be seen passing through controlled-airspace geometry.

Geometry here is an APPROXIMATION of published airspace, synthesised from the
airspace class assigned to each monitored field. It is for situational
awareness and visualisation only. It is NOT for navigation and must not be
used for flight planning or separation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from scripts.airports import AIRPORTS
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from airports import AIRPORTS

NM_TO_M = 1852.0
FEET_TO_METERS = 0.3048

# Airspace ceilings. Class B ceilings are published in feet MSL; Class C and D
# ceilings are published in feet AGL and are converted using field elevation.
CLASS_B_CEILING_FT_MSL = 10_000
CLASS_C_CEILING_FT_AGL = 4_000
CLASS_D_CEILING_FT_AGL = 2_500
CLASS_B_SHELF_FLOOR_FT_AGL = 1_500
CLASS_C_SHELF_FLOOR_FT_AGL = 1_200

# Cylinder radii (nautical miles) for the surface core and the outer shelf.
CLASS_CORE_RADIUS_NM = {"B": 10.0, "C": 5.0, "D": 4.0}
CLASS_SHELF_RADIUS_NM = {"B": 20.0, "C": 10.0}

# A few fields publish a higher Class B ceiling than the 10,000 ft default.
CLASS_B_CEILING_OVERRIDES_FT_MSL = {
    "ATL": 12_500,
    "DEN": 12_000,
    "DFW": 11_000,
    "SLC": 11_000,
    "MSP": 10_000,
    "LAS": 10_000,
}

# Approximate field elevation (feet MSL) used to convert AGL ceilings to MSL.
FIELD_ELEVATION_FT = {
    "ATL": 1026, "LAX": 125, "ORD": 672, "DFW": 607, "DEN": 5434,
    "JFK": 13, "SFO": 13, "SEA": 433, "LAS": 2181, "MCO": 96,
    "MIA": 8, "CLT": 748, "PHX": 1135, "IAH": 97, "BOS": 20,
    "EWR": 18, "MSP": 841, "DTW": 645, "PHL": 36, "LGA": 21,
    "BWI": 146, "SLC": 4227, "DCA": 15, "IAD": 313, "SAN": 17,
    "TPA": 26, "BNA": 599, "AUS": 542, "FLL": 65, "RDU": 435,
    "PDX": 31, "STL": 618, "MCI": 1026, "MSY": 4, "HNL": 13,
    "OAK": 9, "SMF": 27, "SJC": 62, "CLE": 791, "IND": 797,
    "PIT": 1203, "CMH": 815, "CVG": 896, "MKE": 723, "JAX": 30,
    "RSW": 30, "SAT": 809, "BDL": 173, "PBI": 19, "SNA": 56,
    "BUR": 778,
}

# Airspace class assigned to each monitored field (approximate).
AIRSPACE_CLASS = {
    "ATL": "B", "LAX": "B", "ORD": "B", "DFW": "B", "DEN": "B",
    "JFK": "B", "SFO": "B", "SEA": "B", "LAS": "B", "MCO": "B",
    "MIA": "B", "CLT": "B", "PHX": "B", "IAH": "B", "BOS": "B",
    "EWR": "B", "MSP": "B", "DTW": "B", "PHL": "B", "LGA": "B",
    "BWI": "B", "SLC": "B", "DCA": "B", "IAD": "B", "SAN": "B",
    "TPA": "B", "HNL": "B",
    "BNA": "C", "AUS": "C", "FLL": "C", "RDU": "C", "PDX": "C",
    "STL": "C", "MCI": "C", "MSY": "C", "OAK": "C", "SMF": "C",
    "SJC": "C", "CLE": "C", "IND": "C", "PIT": "C", "CMH": "C",
    "CVG": "C", "MKE": "C", "JAX": "C", "RSW": "C", "SAT": "C",
    "BDL": "C", "PBI": "C", "SNA": "C", "BUR": "C",
}

AIRSPACE_CLASS_COLORS = {
    "B": [95, 158, 255],
    "C": [255, 176, 84],
    "D": [150, 230, 168],
}

AIRSPACE_CLASS_DESCRIPTIONS = {
    "B": "Class B — busy terminal area, layered shelves to the published ceiling.",
    "C": "Class C — surface core with an outer shelf from 1,200 ft AGL.",
    "D": "Class D — small surface cylinder around a controlled field.",
}


@dataclass(frozen=True)
class AirspaceVolume:
    """A single stacked cylinder of controlled airspace."""

    airport_code: str
    city: str
    airspace_class: str
    layer: str
    center_lat: float
    center_lon: float
    radius_m: float
    floor_ft_msl: float
    ceiling_ft_msl: float

    @property
    def floor_m(self) -> float:
        return self.floor_ft_msl * FEET_TO_METERS

    @property
    def ceiling_m(self) -> float:
        return self.ceiling_ft_msl * FEET_TO_METERS

    @property
    def color(self) -> list[int]:
        return AIRSPACE_CLASS_COLORS.get(self.airspace_class, [200, 200, 200])

    @property
    def label(self) -> str:
        return (
            f"{self.airport_code} Class {self.airspace_class} "
            f"{self.layer} • {self.floor_ft_msl:,.0f}–{self.ceiling_ft_msl:,.0f} ft MSL"
        )


def field_elevation_ft(airport_code: str) -> float:
    return float(FIELD_ELEVATION_FT.get(airport_code, 0))


def airspace_class(airport_code: str) -> str:
    return AIRSPACE_CLASS.get(airport_code, "D")


def _class_b_ceiling(airport_code: str) -> float:
    return float(
        CLASS_B_CEILING_OVERRIDES_FT_MSL.get(
            airport_code, CLASS_B_CEILING_FT_MSL
        )
    )


def _volumes_for_airport(airport_code: str) -> list[AirspaceVolume]:
    airport = AIRPORTS.get(airport_code)
    if airport is None:
        return []

    kind = airspace_class(airport_code)
    elevation = field_elevation_ft(airport_code)
    core_radius_m = CLASS_CORE_RADIUS_NM[kind] * NM_TO_M

    if kind == "B":
        ceiling = _class_b_ceiling(airport_code)
        core_floor = elevation
        shelf_floor = elevation + CLASS_B_SHELF_FLOOR_FT_AGL
    elif kind == "C":
        ceiling = elevation + CLASS_C_CEILING_FT_AGL
        core_floor = elevation
        shelf_floor = elevation + CLASS_C_SHELF_FLOOR_FT_AGL
    else:
        ceiling = elevation + CLASS_D_CEILING_FT_AGL
        core_floor = elevation
        shelf_floor = None

    volumes = [
        AirspaceVolume(
            airport_code=airport_code,
            city=airport["city"],
            airspace_class=kind,
            layer="surface core",
            center_lat=airport["lat"],
            center_lon=airport["lon"],
            radius_m=core_radius_m,
            floor_ft_msl=core_floor,
            ceiling_ft_msl=ceiling,
        )
    ]

    shelf_radius_nm = CLASS_SHELF_RADIUS_NM.get(kind)
    if shelf_radius_nm is not None and shelf_floor is not None:
        volumes.append(
            AirspaceVolume(
                airport_code=airport_code,
                city=airport["city"],
                airspace_class=kind,
                layer="outer shelf",
                center_lat=airport["lat"],
                center_lon=airport["lon"],
                radius_m=shelf_radius_nm * NM_TO_M,
                floor_ft_msl=shelf_floor,
                ceiling_ft_msl=ceiling,
            )
        )
    return volumes


def airspace_volumes(codes: list[str] | None = None) -> list[AirspaceVolume]:
    """Return the controlled-airspace volumes for the monitored network."""
    selected = codes if codes is not None else sorted(AIRPORTS)
    volumes: list[AirspaceVolume] = []
    for code in selected:
        volumes.extend(_volumes_for_airport(code))
    return volumes


def volume_ring(volume: AirspaceVolume, segments: int = 64) -> list[list[float]]:
    """Return a closed [lon, lat] ring approximating the volume's footprint."""
    lat_scale = math.cos(math.radians(volume.center_lat)) or 1e-6
    ring: list[list[float]] = []
    for step in range(segments):
        bearing = 2 * math.pi * step / segments
        delta_lat = (volume.radius_m / 111_320) * math.cos(bearing)
        delta_lon = (volume.radius_m / (111_320 * lat_scale)) * math.sin(bearing)
        ring.append(
            [volume.center_lon + delta_lon, volume.center_lat + delta_lat]
        )
    ring.append(ring[0])
    return ring


def volumes_to_records(volumes: list[AirspaceVolume]) -> list[dict]:
    """Serialise airspace volumes for the web/globe export payload."""
    records = []
    for volume in volumes:
        records.append(
            {
                "airport_code": volume.airport_code,
                "city": volume.city,
                "airspace_class": volume.airspace_class,
                "layer": volume.layer,
                "center_lat": volume.center_lat,
                "center_lon": volume.center_lon,
                "radius_m": volume.radius_m,
                "floor_ft_msl": volume.floor_ft_msl,
                "ceiling_ft_msl": volume.ceiling_ft_msl,
                "floor_m": volume.floor_m,
                "ceiling_m": volume.ceiling_m,
                "color": volume.color,
                "label": volume.label,
            }
        )
    return records
