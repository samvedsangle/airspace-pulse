from __future__ import annotations

import math

import numpy as np
import pandas as pd


def kalman_smooth_trajectory(df: pd.DataFrame, flight_id: str) -> pd.DataFrame:
    """Smooth latitude/longitude with a constant-velocity Kalman filter."""
    points = df[df["flight_id"] == flight_id].copy()
    if points.empty:
        return pd.DataFrame()
    points["timestamp"] = pd.to_datetime(points["timestamp"], utc=True)
    points = points.sort_values("timestamp").drop_duplicates("timestamp")
    if len(points) < 2:
        return points.assign(
            raw_lat=points["lat"],
            raw_lon=points["lon"],
            filtered_lat=points["lat"],
            filtered_lon=points["lon"],
            velocity_latitude_per_hour=0.0,
            velocity_longitude_per_hour=0.0,
            cov_lat_deg2=0.0,
            cov_lon_deg2=0.0,
            cov_latlon_deg2=0.0,
        )[[
            "flight_id",
            "timestamp",
            "raw_lat",
            "raw_lon",
            "filtered_lat",
            "filtered_lon",
            "velocity_latitude_per_hour",
            "velocity_longitude_per_hour",
            "cov_lat_deg2",
            "cov_lon_deg2",
            "cov_latlon_deg2",
        ]]

    state = np.array([points.iloc[0]["lat"], points.iloc[0]["lon"], 0.0, 0.0], dtype=float)
    covariance = np.eye(4) * 0.01
    measurement_noise = np.eye(2) * 0.000025
    filtered = []
    previous_time = points.iloc[0]["timestamp"]

    for point in points.itertuples():
        delta_seconds = max((point.timestamp - previous_time).total_seconds(), 1.0)
        delta_hours = delta_seconds / 3600
        transition = np.array(
            [[1, 0, delta_hours, 0], [0, 1, 0, delta_hours], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        process_noise = np.diag([1e-7, 1e-7, 1e-5, 1e-5]) * min(delta_seconds, 300)
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise

        measurement = np.array([point.lat, point.lon], dtype=float)
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        innovation = measurement - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + measurement_noise
        gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        state = state + gain @ innovation
        covariance = (np.eye(4) - gain @ observation) @ covariance
        position_covariance = covariance[:2, :2]
        filtered.append(
            {
                "flight_id": flight_id,
                "timestamp": point.timestamp,
                "raw_lat": point.lat,
                "raw_lon": point.lon,
                "filtered_lat": state[0],
                "filtered_lon": state[1],
                "velocity_latitude_per_hour": state[2],
                "velocity_longitude_per_hour": state[3],
                "cov_lat_deg2": float(position_covariance[0, 0]),
                "cov_lon_deg2": float(position_covariance[1, 1]),
                "cov_latlon_deg2": float(position_covariance[0, 1]),
            }
        )
        previous_time = point.timestamp
    return pd.DataFrame(filtered)


def predict_trajectory(smoothed: pd.DataFrame, minutes: int = 10) -> pd.DataFrame:
    if smoothed.empty:
        return pd.DataFrame(columns=["timestamp", "predicted_lat", "predicted_lon"])
    latest = smoothed.iloc[-1]
    future_times = pd.date_range(
        smoothed["timestamp"].iloc[-1] + pd.Timedelta(minutes=1), periods=minutes, freq="min"
    )
    elapsed_hours = np.arange(1, minutes + 1) / 60
    return pd.DataFrame(
        {
            "timestamp": future_times,
            "predicted_lat": latest["filtered_lat"] + latest["velocity_latitude_per_hour"] * elapsed_hours,
            "predicted_lon": latest["filtered_lon"] + latest["velocity_longitude_per_hour"] * elapsed_hours,
        }
    )


def shannon_entropy(df: pd.DataFrame, group_column: str = "airline") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["airport_code", "entropy_bits"])
    counts = df.groupby(["airport_code", group_column]).size().rename("count").reset_index()
    counts["share"] = counts["count"] / counts.groupby("airport_code")["count"].transform("sum")
    entropy = (
        counts.assign(component=-counts["share"] * np.log2(counts["share"]))
        .groupby("airport_code", as_index=False)["component"]
        .sum()
        .rename(columns={"component": "entropy_bits"})
    )
    return entropy.sort_values("entropy_bits", ascending=False)


def hawkes_burst_score(df: pd.DataFrame, decay_hours: float = 2.0) -> pd.DataFrame:
    """Calculate a Hawkes-style exponential self-excitation score per airport-hour."""
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True).dt.floor("h")
    counts = working.groupby(["airport_code", "timestamp"])["flight_id"].nunique().reset_index(name="events")
    scores = []
    for airport, group in counts.groupby("airport_code"):
        group = group.sort_values("timestamp")
        baseline = max(group["events"].mean(), 0.01)
        for row in group.itertuples():
            previous = group[group["timestamp"] < row.timestamp]
            elapsed = (row.timestamp - previous["timestamp"]).dt.total_seconds() / 3600
            excitation = float((previous["events"] * np.exp(-elapsed / decay_hours)).sum())
            scores.append({"airport_code": airport, "timestamp": row.timestamp, "events": row.events, "hawkes_intensity": baseline + excitation})
    return pd.DataFrame(scores)


def transfer_entropy(df: pd.DataFrame, source: str, target: str, lag_hours: int = 1) -> float | None:
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True).dt.floor("h")
    counts = working.groupby(["timestamp", "airport_code"])["flight_id"].nunique().unstack(fill_value=0)
    if source not in counts or target not in counts or len(counts) <= lag_hours + 2:
        return None
    source_values = counts[source].to_numpy()[:-lag_hours]
    target_current = counts[target].to_numpy()[:-lag_hours]
    target_future = counts[target].to_numpy()[lag_hours:]
    source_bins = pd.qcut(source_values, q=min(3, len(np.unique(source_values))), labels=False, duplicates="drop")
    target_bins = pd.qcut(target_current, q=min(3, len(np.unique(target_current))), labels=False, duplicates="drop")
    future_bins = pd.qcut(target_future, q=min(3, len(np.unique(target_future))), labels=False, duplicates="drop")
    frame = pd.DataFrame({"source": source_bins, "target": target_bins, "future": future_bins}).dropna()
    if frame.empty:
        return None
    joint = frame.value_counts(normalize=True)
    target_joint = frame[["target", "future"]].value_counts(normalize=True)
    source_target = frame[["source", "target"]].value_counts(normalize=True)
    target_prob = frame["target"].value_counts(normalize=True)
    value = 0.0
    for (source_bin, target_bin, future_bin), probability in joint.items():
        conditional = probability / source_target.loc[(source_bin, target_bin)]
        baseline = target_joint.loc[(target_bin, future_bin)] / target_prob.loc[target_bin]
        value += probability * math.log2(max(conditional / baseline, 1e-12))
    return max(float(value), 0.0)


def pooled_airport_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes Gamma-Poisson partial pooling for daily airport counts."""
    working = df.copy()
    working["date"] = pd.to_datetime(working["timestamp"], utc=True).dt.date
    daily = working.groupby(["date", "airport_code"])["flight_id"].nunique().reset_index(name="count")
    if daily.empty:
        return pd.DataFrame()
    network_mean = daily["count"].mean()
    prior_strength = 5.0
    result = daily.groupby("airport_code").agg(total_count=("count", "sum"), days=("count", "size")).reset_index()
    result["pooled_daily_rate"] = (result["total_count"] + prior_strength * network_mean) / (result["days"] + prior_strength)
    result["pooling_weight"] = result["days"] / (result["days"] + prior_strength)
    return result.sort_values("pooled_daily_rate", ascending=False)


def cep_alerts(df: pd.DataFrame, window_minutes: int = 2) -> pd.DataFrame:
    """Lightweight streaming rule engine for emergency and local burst events."""
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    alerts = []
    emergency = working[working.get("squawk", "").isin(["7500", "7600", "7700"])] if "squawk" in working else pd.DataFrame()
    for row in emergency.itertuples():
        alerts.append({"timestamp": row.timestamp, "alert_type": "emergency_squawk", "flight_id": row.flight_id, "detail": row.squawk})
    for airport, group in working.groupby("airport_code"):
        grouped = group.set_index("timestamp").resample(f"{window_minutes}min")["flight_id"].nunique()
        for timestamp, count in grouped[grouped >= 3].items():
            alerts.append({"timestamp": timestamp, "alert_type": "local_burst", "flight_id": airport, "detail": f"{count} aircraft"})
    return pd.DataFrame(
        alerts,
        columns=["timestamp", "alert_type", "flight_id", "detail"],
    )


# Simplified ICAO/FAA-style radar wake-turbulence separation minima, in
# nautical miles, keyed by (leader_class, follower_class). This is the
# standard "Heavy/Large/Small" table (JO 7110.65-style); the newer RECAT
# system has finer categories, and real ATC separation also depends on
# approach phase and aircraft-specific RECAT groupings this simplified model
# doesn't have. Any pair missing from this table falls back to the ordinary
# 3 nm non-wake radar minimum.
WAKE_SEPARATION_MINIMA_NM = {
    ("Heavy", "Heavy"): 4.0,
    ("Heavy", "Large"): 5.0,
    ("Heavy", "Small"): 6.0,
    ("Heavy", "Light"): 6.0,
    ("Large", "Small"): 4.0,
    ("Large", "Light"): 4.0,
}
NM_PER_KM = 1.0 / 1.852


KM_PER_DEGREE_LAT = 111.32
MAX_CROSS_TRACK_NM = 0.75


def wake_separation_alerts(
    df: pd.DataFrame,
    heading_tolerance_deg: float = 25.0,
    max_check_distance_nm: float = 8.0,
    time_window_minutes: int = 3,
) -> pd.DataFrame:
    """Flag pairs trailing closer than the real ICAO-style wake minimum for their classes.

    This only became possible to do honestly once scripts.aircraft_registry
    gave most aircraft a real wake-turbulence category instead of a coarse
    default — see scripts/wake_vortex.py. "Trailing" is decomposed into
    along-track distance (behind the leader, along its heading) and
    cross-track distance (sideways offset): two aircraft on similar headings
    but in different lanes — parallel runways, offset approach paths — have
    a large cross-track offset and are correctly not in each other's wake,
    even though a plain straight-line distance check would flag them.

    Only leader/follower class pairs with an *elevated* wake minimum
    (WAKE_SEPARATION_MINIMA_NM) are checked — same-class pairs (Light
    following Light, Small following Small) fall back to the ordinary 3 nm
    IFR radar minimum, which isn't a wake hazard at all and, on
    multi-minute snapshots with no IFR/VFR or approach-phase context, would
    mostly just flag normal, safely-separated traffic. Unknown-class
    aircraft are skipped for the same reason: no responsible risk call can
    be made without knowing what's actually following what. This is still a
    situational-awareness heuristic over noisy multi-minute snapshots, not a
    verified encounter — real ATC separation uses continuous track data and
    finer RECAT categories.
    """
    from scripts.wake_vortex import wake_class_for

    working = df.dropna(subset=["lat", "lon", "heading_deg", "speed_knots"]).copy()
    if "on_ground" in working:
        working = working[~working["on_ground"]]
    columns = ["timestamp", "leader", "follower", "along_track_nm", "required_nm", "leader_class", "follower_class"]
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["time_bucket"] = working["timestamp"].dt.floor(f"{time_window_minutes}min")

    alerts = []
    for (_airport, _bucket), group in working.groupby(["airport_code", "time_bucket"]):
        records = list(group.itertuples())
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                heading_diff = abs((a.heading_deg - b.heading_deg + 180) % 360 - 180)
                if heading_diff > heading_tolerance_deg:
                    continue

                lat_scale = KM_PER_DEGREE_LAT
                lon_scale = KM_PER_DEGREE_LAT * max(math.cos(math.radians(a.lat)), 1e-6)
                north_km = (b.lat - a.lat) * lat_scale
                east_km = (b.lon - a.lon) * lon_scale
                straight_line_nm = math.hypot(north_km, east_km) * NM_PER_KM
                if straight_line_nm > max_check_distance_nm:
                    continue

                heading_rad = math.radians(a.heading_deg)
                heading_north, heading_east = math.cos(heading_rad), math.sin(heading_rad)
                along_track_km = north_km * heading_north + east_km * heading_east
                cross_track_km = north_km * -heading_east + east_km * heading_north
                if abs(cross_track_km) * NM_PER_KM > MAX_CROSS_TRACK_NM:
                    continue  # different lane, not in-trail — not a wake encounter

                along_track_nm = abs(along_track_km) * NM_PER_KM
                leader, follower = (a, b) if along_track_km > 0 else (b, a)

                leader_class, _ = wake_class_for(
                    getattr(leader, "aircraft_category", None), getattr(leader, "wake_category_description", None)
                )
                follower_class, _ = wake_class_for(
                    getattr(follower, "aircraft_category", None), getattr(follower, "wake_category_description", None)
                )
                required_nm = WAKE_SEPARATION_MINIMA_NM.get((leader_class, follower_class))
                if required_nm is not None and along_track_nm < required_nm:
                    alerts.append(
                        {
                            "timestamp": leader.timestamp,
                            "leader": leader.flight_id,
                            "follower": follower.flight_id,
                            "along_track_nm": round(along_track_nm, 2),
                            "required_nm": required_nm,
                            "leader_class": leader_class,
                            "follower_class": follower_class,
                        }
                    )
    return pd.DataFrame(alerts, columns=columns).sort_values("timestamp", ascending=False) if alerts else pd.DataFrame(
        columns=columns
    )
