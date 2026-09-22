from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AnalyticsResult:
    ready: bool
    message: str
    data: pd.DataFrame | None = None


@dataclass
class FleetComposition:
    ready: bool
    message: str
    by_manufacturer: pd.DataFrame | None = None
    by_type: pd.DataFrame | None = None
    by_wake_class: pd.DataFrame | None = None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    defaults = {
        "airline": "Unknown airline",
        "altitude_ft": np.nan,
        "aircraft_category": "Unknown",
        "on_ground": False,
        "speed_knots": np.nan,
    }
    for column, default in defaults.items():
        if column not in working:
            working[column] = default
    return working


def hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "airport_code", "aircraft_count"])
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    return (
        working.assign(timestamp=working["timestamp"].dt.floor("h"))
        .groupby(["timestamp", "airport_code"], as_index=False)["flight_id"]
        .nunique()
        .rename(columns={"flight_id": "aircraft_count"})
    )


def forecast_airport(df: pd.DataFrame, airport_code: str, horizon: int = 12) -> AnalyticsResult:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    counts = hourly_counts(df)
    series = counts[counts["airport_code"] == airport_code].set_index("timestamp")["aircraft_count"]
    series = series.asfreq("h", fill_value=0)
    if len(series) < 48:
        return AnalyticsResult(False, "At least 48 hourly observations are needed for forecasting.")
    seasonal_periods = 24 if len(series) >= 72 else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)
    future_index = pd.date_range(series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    forecast = model.forecast(horizon).clip(lower=0)
    result = pd.DataFrame({"timestamp": future_index, "predicted_aircraft": forecast.values})
    return AnalyticsResult(True, "Forecast generated with Holt-Winters exponential smoothing.", result)


def stl_decomposition(df: pd.DataFrame, airport_code: str) -> AnalyticsResult:
    from statsmodels.tsa.seasonal import STL

    counts = hourly_counts(df)
    series = counts[counts["airport_code"] == airport_code].set_index("timestamp")["aircraft_count"].asfreq("h", fill_value=0)
    if len(series) < 48:
        return AnalyticsResult(False, "At least 48 hourly observations are needed for daily seasonality.")
    result = STL(series, period=24, robust=True).fit()
    decomposition = pd.DataFrame(
        {"observed": series, "trend": result.trend, "seasonal": result.seasonal, "residual": result.resid}
    ).reset_index()
    return AnalyticsResult(True, "STL decomposition separates trend, daily seasonality, and residual noise.", decomposition)


def residual_anomalies(df: pd.DataFrame, airport_code: str) -> AnalyticsResult:
    counts = hourly_counts(df)
    series = counts[counts["airport_code"] == airport_code].set_index("timestamp")["aircraft_count"].asfreq("h", fill_value=0)
    if len(series) < 48:
        return AnalyticsResult(False, "At least 48 hourly observations are needed for residual anomalies.")
    baseline = series.shift(24)
    residual = series - baseline
    scale = residual.rolling(24, min_periods=12).std().replace(0, np.nan)
    result = pd.DataFrame({"timestamp": series.index, "actual": series.values, "expected": baseline.values, "residual": residual.values})
    result["residual_score"] = (result["residual"] / scale.values).replace([np.inf, -np.inf], np.nan)
    result["anomaly"] = result["residual_score"].abs() >= 3
    return AnalyticsResult(True, "Residuals compare traffic with the same hour on the prior day.", result)


def multivariate_anomalies(df: pd.DataFrame) -> AnalyticsResult:
    working = normalize_columns(df)
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["date"] = working["timestamp"].dt.date
    features = working.groupby(["date", "airport_code"]).agg(
        aircraft_count=("flight_id", "nunique"),
        airline_diversity=("airline", "nunique"),
        median_altitude=("altitude_ft", "median"),
        category_diversity=("aircraft_category", "nunique"),
        airborne_ratio=("on_ground", lambda values: 1 - values.mean()),
    ).reset_index()
    numeric = ["aircraft_count", "airline_diversity", "median_altitude", "category_diversity", "airborne_ratio"]
    features[numeric] = features[numeric].fillna(0)
    if features["date"].nunique() < 20:
        return AnalyticsResult(False, "At least 20 distinct days are needed for multivariate anomaly detection.", features)
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(random_state=42, contamination="auto")
    features["anomaly_score"] = model.fit_predict(features[numeric])
    features["is_anomaly"] = features["anomaly_score"] == -1
    return AnalyticsResult(True, "Isolation Forest combines count, altitude, airline, type, and airborne mix.", features)


def change_points(df: pd.DataFrame, airport_code: str) -> AnalyticsResult:
    import ruptures as rpt

    counts = hourly_counts(df)
    series = counts[counts["airport_code"] == airport_code].set_index("timestamp")["aircraft_count"].asfreq("h", fill_value=0)
    if len(series) < 72:
        return AnalyticsResult(False, "At least 72 hourly observations are needed for change-point detection.")
    breakpoints = rpt.Pelt(model="rbf").fit(series.to_numpy()).predict(pen=3)
    points = [series.index[index - 1] for index in breakpoints[:-1]]
    return AnalyticsResult(True, "PELT detected structural shifts in the hourly traffic signal.", pd.DataFrame({"change_point": points}))


def airport_benchmark(df: pd.DataFrame, bootstrap_runs: int = 300) -> AnalyticsResult:
    working = df.copy()
    working["date"] = pd.to_datetime(working["timestamp"], utc=True).dt.date
    daily = working.groupby(["date", "airport_code"])["flight_id"].nunique().reset_index(name="aircraft_count")
    if daily["date"].nunique() < 3:
        return AnalyticsResult(False, "At least 3 collected days are needed for confidence intervals.", daily)
    rng = np.random.default_rng(42)
    rows = []
    for airport, group in daily.groupby("airport_code"):
        values = group["aircraft_count"].to_numpy()
        means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(bootstrap_runs)]
        rows.append({"airport_code": airport, "mean_daily_aircraft": values.mean(), "ci_low": np.percentile(means, 2.5), "ci_high": np.percentile(means, 97.5)})
    return AnalyticsResult(True, "Bootstrap intervals quantify uncertainty in airport comparisons.", pd.DataFrame(rows).sort_values("mean_daily_aircraft", ascending=False))


def inferred_network(df: pd.DataFrame) -> AnalyticsResult:
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working = working.sort_values(["flight_id", "timestamp"])
    transitions = []
    for _, group in working.groupby("flight_id"):
        group = group.drop_duplicates("airport_code")
        for first, second in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            hours = (second.timestamp - first.timestamp).total_seconds() / 3600
            if first.airport_code != second.airport_code and 0 < hours <= 6:
                transitions.append({"origin": first.airport_code, "destination": second.airport_code, "observations": 1})
    edges = pd.DataFrame(transitions)
    if edges.empty:
        return AnalyticsResult(False, "No plausible same-aircraft airport transitions are available yet.", edges)
    edges = edges.groupby(["origin", "destination"], as_index=False)["observations"].sum()
    return AnalyticsResult(True, "Edges are inferred from the same callsign near two airports within six hours, not confirmed routes.", edges)


def movement_signatures(df: pd.DataFrame) -> AnalyticsResult:
    working = normalize_columns(df)
    if working["speed_knots"].notna().sum() < 10 or working["altitude_ft"].notna().sum() < 10:
        return AnalyticsResult(False, "Speed and altitude telemetry are needed for movement signature clustering.")
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    features = working.groupby("flight_id").agg(
        mean_speed=("speed_knots", "mean"),
        mean_altitude=("altitude_ft", "mean"),
        altitude_range=("altitude_ft", lambda values: values.max() - values.min()),
        position_count=("timestamp", "size"),
    ).fillna(0)
    if len(features) < 10:
        return AnalyticsResult(False, "At least 10 aircraft with telemetry are needed for signature clustering.", features.reset_index())
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feature_names = ["mean_speed", "mean_altitude", "altitude_range", "position_count"]
    scaled = StandardScaler().fit_transform(features[feature_names])
    cluster_count = min(4, len(features), max(features[feature_names].drop_duplicates().shape[0], 1))
    if cluster_count < 2:
        return AnalyticsResult(False, "Movement signatures are identical in the current sample.", features.reset_index())
    features["signature_cluster"] = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(scaled)
    return AnalyticsResult(True, "K-means groups aircraft by observed movement signature; labels are exploratory.", features.reset_index())


# Quantiles used to widen forecast bands in the fan chart.
FORECAST_INTERVAL_QUANTILES = {0.5: 0.6745, 0.8: 1.2816, 0.95: 1.9600}


def forecast_with_intervals(
    df: pd.DataFrame,
    airport_code: str,
    horizon: int = 12,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> AnalyticsResult:
    """Forecast with widening credible bands, for a fan chart.

    The predictive spread grows with the square root of the forecast step (a
    random-walk fan), scaled by the in-sample residual standard error of the
    Holt-Winters fit. This is a transparent heuristic for visualisation, not an
    exact analytic prediction interval.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    counts = hourly_counts(df)
    series = counts[counts["airport_code"] == airport_code].set_index("timestamp")["aircraft_count"]
    series = series.asfreq("h", fill_value=0)
    if len(series) < 48:
        return AnalyticsResult(False, "At least 48 hourly observations are needed for forecast intervals.")

    seasonal_periods = 24 if len(series) >= 72 else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)

    point_forecast = model.forecast(horizon).clip(lower=0)
    residual_scale = float(np.nanstd(np.asarray(model.resid, dtype=float)))
    if not np.isfinite(residual_scale) or residual_scale == 0:
        residual_scale = max(float(series.std()), 1.0)

    future_index = pd.date_range(series.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    steps = np.arange(1, horizon + 1)
    width = residual_scale * np.sqrt(steps)

    result = pd.DataFrame({"timestamp": future_index, "point_forecast": point_forecast.values})
    for level in levels:
        z = FORECAST_INTERVAL_QUANTILES[level]
        label = int(round(level * 100))
        result[f"lower_{label}"] = np.clip(point_forecast.values - z * width, 0, None)
        result[f"upper_{label}"] = np.clip(point_forecast.values + z * width, 0, None)
    return AnalyticsResult(
        True,
        "Fan chart: credible bands widen with the square root of the forecast step.",
        result,
    )


def fleet_composition(df: pd.DataFrame) -> FleetComposition:
    """Break down observed traffic by real manufacturer/type/wake class.

    Only meaningful once scripts.aircraft_registry has been downloaded and
    scripts/backfill_metadata.py has run — before that, `manufacturer` and
    `aircraft_type` are mostly missing and this reports "not ready" rather
    than a fleet mix built from a handful of lucky hexdb.io hits.
    """
    from scripts.wake_vortex import wake_class_for

    if df.empty or "manufacturer" not in df.columns:
        return FleetComposition(False, "No manufacturer/aircraft-type data is available yet in this snapshot.")

    working = df.dropna(subset=["manufacturer"]).copy()
    if working.empty:
        return FleetComposition(
            False,
            "No aircraft in this snapshot have registry-backed manufacturer data yet — "
            "download data/opensky_aircraft_database.csv and run scripts/backfill_metadata.py.",
        )

    by_manufacturer = (
        working.groupby("manufacturer")["flight_id"].nunique().sort_values(ascending=False).reset_index(name="aircraft_count")
    )
    by_type = (
        working.dropna(subset=["aircraft_type"])
        .groupby("aircraft_type")["flight_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="aircraft_count")
    )
    wake_classes = working.apply(
        lambda row: wake_class_for(row.get("aircraft_category"), row.get("wake_category_description"))[0], axis=1
    )
    by_wake_class = (
        wake_classes.value_counts().rename_axis("wake_class").reset_index(name="aircraft_count")
    )
    coverage = len(working) / len(df)
    return FleetComposition(
        True,
        f"Real manufacturer/type data covers {coverage:.0%} of this snapshot "
        "(scripts/aircraft_registry.py, OpenSky Network aircraft database).",
        by_manufacturer=by_manufacturer,
        by_type=by_type,
        by_wake_class=by_wake_class,
    )
