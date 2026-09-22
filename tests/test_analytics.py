from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.analytics import forecast_airport, multivariate_anomalies, stl_decomposition


def make_history(days=3, hours_per_day=24):
    rows = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for hour in range(days * hours_per_day):
        timestamp = start + timedelta(hours=hour)
        for aircraft_number in range(3 + hour % 4):
            rows.append(
                {
                    "flight_id": f"A{aircraft_number}",
                    "timestamp": timestamp,
                    "airport_code": "ATL",
                    "airline": "Test Air",
                    "altitude_ft": 10000 + aircraft_number * 1000,
                    "aircraft_category": "A3",
                    "on_ground": False,
                }
            )
    return pd.DataFrame(rows)


def test_forecast_and_stl_require_and_use_hourly_history():
    history = make_history()

    forecast = forecast_airport(history, "ATL")
    decomposition = stl_decomposition(history, "ATL")

    assert forecast.ready
    assert len(forecast.data) == 12
    assert decomposition.ready
    assert {"trend", "seasonal", "residual"}.issubset(decomposition.data.columns)


def test_multivariate_anomaly_requires_distinct_days():
    history = make_history(days=19)
    result = multivariate_anomalies(history)
    assert not result.ready

    result = multivariate_anomalies(make_history(days=20))
    assert result.ready
    assert "is_anomaly" in result.data