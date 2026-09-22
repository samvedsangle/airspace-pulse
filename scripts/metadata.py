from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

OPENFLIGHTS_AIRLINES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat"
HEXDB_AIRCRAFT_URL = "https://hexdb.io/api/v1/aircraft/{hex_code}"
NON_AIRLINE_LABELS = {
    "Unknown airline",
    "Unidentified aircraft",
    "Unidentified operator",
    "Private / General Aviation",
}
COMMON_AIRLINE_PREFIXES = {
    "AAL": "American Airlines",
    "UAL": "United Airlines",
    "DAL": "Delta Air Lines",
    "SWA": "Southwest Airlines",
    "JBU": "JetBlue Airways",
    "ASA": "Alaska Airlines",
    "ALK": "Alaska Airlines",
    "FFT": "Frontier Airlines",
    "AAY": "Allegiant Air",
    "ACA": "Air Canada",
    "AFR": "Air France",
    "BAW": "British Airways",
    "KAL": "Korean Air",
    "QFA": "Qantas",
    "VIR": "Virgin Atlantic",
    "VOI": "Volaris",
    "VIV": "Viva Aerobus",
    "AMX": "Aeromexico",
    "ABX": "ABX Air",
    "BOX": "European Air Transport Leipzig",
    "CMP": "Copa Airlines",
    "CKS": "Kalitta Air",
    "EDV": "Endeavor Air",
    "JIA": "PSA Airlines",
    "RPA": "Republic Airways",
    "SCX": "Sun Country Airlines",
    "WGN": "Western Global Airlines",
    "EJA": "NetJets",
    "VJT": "VistaJet",
    "DHK": "DHL Aviation",
    "CLX": "Cargolux",
    "CPA": "Cathay Pacific",
}


def load_openflights_prefixes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    prefixes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if len(fields) < 8:
            continue
        name = fields[1].strip('"')
        iata = fields[4].strip('"').upper()
        icao = fields[5].strip('"').upper()
        if name and name != r"\N":
            for prefix in (icao, iata):
                if prefix and prefix != r"\N":
                    prefixes[prefix] = name
    return prefixes


def download_openflights_airlines(destination: Path) -> Path:
    request = Request(OPENFLIGHTS_AIRLINES_URL, headers={"User-Agent": "airspace-pulse/1.0"})
    with urlopen(request, timeout=20) as response:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.read())
    return destination


def resolve_callsign_airline(callsign: str, prefixes: dict[str, str]) -> str:
    normalized = (callsign or "").strip().upper()
    prefixes = {**COMMON_AIRLINE_PREFIXES, **prefixes}
    for length in (4, 3, 2):
        prefix = normalized[:length]
        if prefix in prefixes:
            return prefixes[prefix]
    if len(normalized) == 6 and all(character in "0123456789ABCDEF" for character in normalized):
        return "Unidentified aircraft"
    if normalized.startswith("N") and normalized[1:].replace("-", "").isalnum():
        return "Private / General Aviation"
    return "Unidentified operator"


def lookup_hexdb(hex_code: str, cache_path: Path) -> dict:
    normalized = (hex_code or "").strip().lower()
    if not normalized:
        return {}
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    if normalized in cache:
        return cache[normalized]
    if os.getenv("AIRSPACE_ENABLE_HEXDB", "1") != "1":
        return {}
    request = Request(
        HEXDB_AIRCRAFT_URL.format(hex_code=normalized),
        headers={"Accept": "application/json", "User-Agent": "airspace-pulse/1.0"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read())
    except Exception:
        return {}
    if result.get("status") == "404":
        return {}
    cache[normalized] = result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    return result