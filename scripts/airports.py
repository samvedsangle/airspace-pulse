import math


AIRPORTS = {
    "ATL": {"city": "Atlanta", "lat": 33.6407, "lon": -84.4277},
    "LAX": {"city": "Los Angeles", "lat": 33.9416, "lon": -118.4085},
    "ORD": {"city": "Chicago", "lat": 41.9742, "lon": -87.9073},
    "DFW": {"city": "Dallas-Fort Worth", "lat": 32.8998, "lon": -97.0403},
    "DEN": {"city": "Denver", "lat": 39.8561, "lon": -104.6737},
    "JFK": {"city": "New York", "lat": 40.6413, "lon": -73.7781},
    "SFO": {"city": "San Francisco", "lat": 37.6213, "lon": -122.3790},
    "SEA": {"city": "Seattle", "lat": 47.4502, "lon": -122.3088},
    "LAS": {"city": "Las Vegas", "lat": 36.0840, "lon": -115.1537},
    "MCO": {"city": "Orlando", "lat": 28.4312, "lon": -81.3081},
    "MIA": {"city": "Miami", "lat": 25.7959, "lon": -80.2870},
    "CLT": {"city": "Charlotte", "lat": 35.2140, "lon": -80.9431},
    "PHX": {"city": "Phoenix", "lat": 33.4342, "lon": -112.0116},
    "IAH": {"city": "Houston", "lat": 29.9902, "lon": -95.3368},
    "BOS": {"city": "Boston", "lat": 42.3656, "lon": -71.0096},
    "EWR": {"city": "Newark", "lat": 40.6895, "lon": -74.1745},
    "MSP": {"city": "Minneapolis", "lat": 44.8848, "lon": -93.2223},
    "DTW": {"city": "Detroit", "lat": 42.2124, "lon": -83.3534},
    "PHL": {"city": "Philadelphia", "lat": 39.8744, "lon": -75.2424},
    "LGA": {"city": "New York", "lat": 40.7769, "lon": -73.8740},
    "BWI": {"city": "Baltimore", "lat": 39.1774, "lon": -76.6684},
    "SLC": {"city": "Salt Lake City", "lat": 40.7899, "lon": -111.9791},
    "DCA": {"city": "Washington, DC", "lat": 38.8512, "lon": -77.0402},
    "IAD": {"city": "Washington, DC", "lat": 38.9531, "lon": -77.4565},
    "SAN": {"city": "San Diego", "lat": 32.7338, "lon": -117.1933},
    "TPA": {"city": "Tampa", "lat": 27.9755, "lon": -82.5332},
    "BNA": {"city": "Nashville", "lat": 36.1263, "lon": -86.6774},
    "AUS": {"city": "Austin", "lat": 30.1975, "lon": -97.6664},
    "FLL": {"city": "Fort Lauderdale", "lat": 26.0742, "lon": -80.1506},
    "RDU": {"city": "Raleigh", "lat": 35.8801, "lon": -78.7880},
    "PDX": {"city": "Portland", "lat": 45.5898, "lon": -122.5951},
    "STL": {"city": "St. Louis", "lat": 38.7487, "lon": -90.3700},
    "MCI": {"city": "Kansas City", "lat": 39.2976, "lon": -94.7139},
    "MSY": {"city": "New Orleans", "lat": 29.9934, "lon": -90.2580},
    "HNL": {"city": "Honolulu", "lat": 21.3187, "lon": -157.9225},
    "OAK": {"city": "Oakland", "lat": 37.7213, "lon": -122.2208},
    "SMF": {"city": "Sacramento", "lat": 38.6951, "lon": -121.5908},
    "SJC": {"city": "San Jose", "lat": 37.3626, "lon": -121.9290},
    "CLE": {"city": "Cleveland", "lat": 41.4117, "lon": -81.8498},
    "IND": {"city": "Indianapolis", "lat": 39.7173, "lon": -86.2944},
    "PIT": {"city": "Pittsburgh", "lat": 40.4915, "lon": -80.2329},
    "CMH": {"city": "Columbus", "lat": 39.9980, "lon": -82.8919},
    "CVG": {"city": "Cincinnati", "lat": 39.0488, "lon": -84.6678},
    "MKE": {"city": "Milwaukee", "lat": 42.9472, "lon": -87.8966},
    "JAX": {"city": "Jacksonville", "lat": 30.4941, "lon": -81.6879},
    "RSW": {"city": "Fort Myers", "lat": 26.5362, "lon": -81.7552},
    "SAT": {"city": "San Antonio", "lat": 29.5337, "lon": -98.4698},
    "BDL": {"city": "Hartford", "lat": 41.9389, "lon": -72.6832},
    "PBI": {"city": "West Palm Beach", "lat": 26.6832, "lon": -80.0956},
    "SNA": {"city": "Orange County", "lat": 33.6757, "lon": -117.8682},
    "BUR": {"city": "Burbank", "lat": 34.2007, "lon": -118.3585},
}

RADIUS_KM = 160


def distance_km(latitude_a, longitude_a, latitude_b, longitude_b):
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)
    origin_latitude = math.radians(latitude_a)
    target_latitude = math.radians(latitude_b)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(origin_latitude)
        * math.cos(target_latitude)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(haversine))


def nearest_airport(latitude, longitude, radius_km=RADIUS_KM):
    airport_code, airport = min(
        AIRPORTS.items(),
        key=lambda item: distance_km(
            latitude,
            longitude,
            item[1]["lat"],
            item[1]["lon"],
        ),
    )
    distance = distance_km(latitude, longitude, airport["lat"], airport["lon"])
    return (airport_code, airport) if distance <= radius_km else None
def build_aircraft_record(
    flight_id,
    timestamp,
    airport_code,
    airport,
    latitude,
    longitude,
    on_ground=False,
    airline="Unknown airline",
    altitude_ft=None,
    speed_knots=None,
    heading_deg=None,
    squawk="",
    aircraft_category="Unknown",
):
    """Build the canonical per-aircraft record shared by both ingest scripts."""
    return {
        "flight_id": (flight_id or "UNKNOWN").strip(),
        "airline": (airline or "Unknown airline").strip(),
        "timestamp": timestamp,
        "destination_city": airport["city"],
        "lat": float(latitude),
        "lon": float(longitude),
        "airport_code": airport_code,
        "on_ground": bool(on_ground),
        "altitude_ft": altitude_ft,
        "speed_knots": speed_knots,
        "heading_deg": heading_deg,
        "squawk": (squawk or "").strip(),
        "aircraft_category": aircraft_category or "Unknown",
    }
