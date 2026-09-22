"""Digital-twin traffic replay: a calibrated counterfactual, not a full airport DES.

The ask was a discrete-event simulation of arrivals/departures/taxi/gate
operations. Building a faithful one needs runway configurations, gate maps,
and taxi-time distributions we do not have for 50+ airports; a "digital
twin" that fabricates those would just be decorative. What we *do* have,
honestly, is each airport's observed hourly arrival-rate profile
(scripts.advanced_analytics.pooled_airport_rates supplies the smoothed daily
rate; this module derives the hour-of-day shape directly from history).

So this is a calibrated non-homogeneous Poisson arrival simulator: for each
hour of the day we estimate lambda(hour) from observed history, then sample
a synthetic count ~ Poisson(lambda(hour) * capacity_multiplier). Set the
multiplier below 1.0 to play out a simplified "what if capacity were cut by
X%" scenario (a stand-in for a runway closure or ATC ground stop) and
compare the simulated counterfactual to what was actually observed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_HOURS_FOR_CALIBRATION = 24


@dataclass
class DigitalTwinResult:
    ready: bool
    message: str
    hourly_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    simulation: pd.DataFrame = field(default_factory=pd.DataFrame)


def _hourly_profile(df: pd.DataFrame, airport_code: str) -> pd.DataFrame:
    working = df[df["airport_code"] == airport_code].copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["hour"] = working["timestamp"].dt.floor("h")
    counts = working.groupby("hour")["flight_id"].nunique().rename("aircraft_count").reset_index()
    counts["hour_of_day"] = counts["hour"].dt.hour
    profile = counts.groupby("hour_of_day")["aircraft_count"].mean().reindex(range(24), fill_value=0)
    return profile.rename("lambda_hourly").reset_index().rename(columns={"index": "hour_of_day"})


def simulate_digital_twin(
    df: pd.DataFrame,
    airport_code: str,
    capacity_multiplier: float = 1.0,
    horizon_hours: int = 24,
    random_seed: int = 42,
) -> DigitalTwinResult:
    """Simulate a capacity-scaled counterfactual traffic day against observed history."""
    if df.empty or "airport_code" not in df:
        return DigitalTwinResult(False, "No airport telemetry is available yet to calibrate a digital twin.")

    subset = df[df["airport_code"] == airport_code]
    if len(subset) < MIN_HOURS_FOR_CALIBRATION:
        return DigitalTwinResult(
            False,
            f"At least {MIN_HOURS_FOR_CALIBRATION} observations at {airport_code} are needed to calibrate hourly rates.",
        )

    profile = _hourly_profile(df, airport_code)
    rng = np.random.default_rng(random_seed)
    rows = []
    for step in range(horizon_hours):
        hour_of_day = step % 24
        lam = float(profile.loc[profile["hour_of_day"] == hour_of_day, "lambda_hourly"].iloc[0])
        scaled_lambda = max(lam * capacity_multiplier, 0.0)
        simulated = int(rng.poisson(scaled_lambda))
        rows.append(
            {
                "step": step,
                "hour_of_day": hour_of_day,
                "baseline_lambda": lam,
                "capacity_multiplier": capacity_multiplier,
                "simulated_aircraft": simulated,
            }
        )

    return DigitalTwinResult(
        True,
        "Non-homogeneous Poisson twin calibrated on the observed hour-of-day arrival profile "
        "— a simplified capacity-scenario replay, not a full gate/taxi operations model.",
        hourly_profile=profile,
        simulation=pd.DataFrame(rows),
    )
