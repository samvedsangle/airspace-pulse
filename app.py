import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from scripts.aircraft_registry import download_opensky_database
from scripts.airports import AIRPORTS, distance_km
from scripts.airspace import airspace_volumes, volume_ring
from scripts.analytics import (
    airport_benchmark,
    change_points,
    fleet_composition,
    forecast_airport,
    inferred_network,
    movement_signatures,
    multivariate_anomalies,
    residual_anomalies,
    stl_decomposition,
)
from scripts.advanced_analytics import (
    cep_alerts,
    hawkes_burst_score,
    kalman_smooth_trajectory,
    pooled_airport_rates,
    predict_trajectory,
    shannon_entropy,
    transfer_entropy,
    wake_separation_alerts,
)
from scripts.density_field import build_density_field, field_to_grid_frame
from scripts.digital_twin import simulate_digital_twin
from scripts.flow_field import compute_flow_field
from scripts.generative_art import render_generative_art
from scripts.glyphs import compute_star_glyphs
from scripts.hyperbolic_layout import build_hyperbolic_layout
from scripts.ingest_flights import dedupe_snapshots, ingest_adsb_lol_data, summarize_live_flights
from scripts.metadata import NON_AIRLINE_LABELS, download_openflights_airlines, resolve_callsign_airline
from scripts.mood_theme import compute_mood_score, mood_css, mood_label
from scripts.topology import build_day_fingerprints, mapper_graph, persistence_diagram, project_days
from scripts.trajectory_manifold import build_trajectory_manifold, decode_trajectory_at
from scripts.uncertainty import trajectory_uncertainty
from scripts.wake_vortex import compute_wake_trails, trails_to_frame

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"


@st.cache_resource(show_spinner="Fetching airline and aircraft reference data (first run only)...")
def bootstrap_reference_data():
    """Fetch the reference datasets a fresh clone won't have.

    data/ is entirely gitignored (bronze_flights.parquet grows without
    bound, and the OpenSky database alone is ~95MB — a poor fit for a repo
    that's re-cloned on every deploy). A machine that's been running this
    dashboard for a while already has these; a brand new deploy (e.g.
    Streamlit Community Cloud) does not, so fetch them once per container
    lifetime here. Failures are swallowed deliberately: airline/aircraft
    lookups already degrade to "Unknown" without these files (see
    scripts/metadata.py and scripts/aircraft_registry.py), so a blocked or
    slow network on first boot should never crash the dashboard itself.
    """
    airlines_path = DATA_DIRECTORY / "airlines.dat"
    if not airlines_path.exists():
        try:
            download_openflights_airlines(airlines_path)
        except Exception:
            pass

    registry_path = DATA_DIRECTORY / "opensky_aircraft_database.csv"
    if not registry_path.exists():
        try:
            download_opensky_database(registry_path)
        except Exception:
            pass
    return True


st.set_page_config(layout="wide", page_title="Live Mobility Analyzer", page_icon="🛫")

bootstrap_reference_data()

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at 12% 0%, rgba(28, 94, 109, 0.32), transparent 32%), linear-gradient(180deg, #06131b 0%, #0b1d24 100%);
            color: #edf6ff;
        }
        div[data-testid="stSidebar"] {
            background: #091a2a;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.7rem;
            font-weight: 700;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(38, 184, 117, 0.15);
            border: 1px solid rgba(38, 184, 117, 0.4);
            color: #6ef0a7;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #6ef0a7;
            box-shadow: 0 0 12px rgba(110, 240, 167, 0.9);
            animation: pulse 1.6s infinite ease-in-out;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.7; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.7; }
        }
        .top-card {
            background: linear-gradient(180deg, rgba(20, 48, 57, 0.94), rgba(11, 29, 35, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 0.9rem;
            padding: 1rem;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .top-card:hover {
            transform: translateY(-2px);
            border-color: rgba(110, 240, 167, 0.6);
            box-shadow: 0 0 18px rgba(110, 240, 167, 0.18);
        }
        .top-card.best {
            background: linear-gradient(180deg, rgba(24, 73, 70, 0.94), rgba(14, 42, 42, 0.94));
            border-color: rgba(110, 240, 167, 0.7);
            box-shadow: 0 0 18px rgba(110, 240, 167, 0.2);
        }
        .top-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.8rem;
            color: #9eb7cf;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .top-card .value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #f8fbff;
        }
        .top-card.best .value {
            color: #8af5b4;
        }
        .stDataFrame {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 0.8rem;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Airspace Pulse")
st.markdown(
    '<div class="status-pill"><span class="status-dot"></span> LIVE</div> Aircraft activity around monitored airports',
    unsafe_allow_html=True,
)
st.caption(
    "A situational view of aircraft observed within 160 km of the monitored airport network. "
    "This is proximity activity, not confirmed arrivals or destinations."
)


@st.cache_data(ttl=180)
def read_snapshot_data():
    snapshot_paths = sorted(DATA_DIRECTORY.glob("bronze_flights*.parquet"))
    if not snapshot_paths:
        return pd.DataFrame()

    snapshots = [pd.read_parquet(path) for path in snapshot_paths]
    combined = dedupe_snapshots(pd.concat(snapshots, ignore_index=True))
    if "airline" not in combined:
        combined["airline"] = "Unknown airline"
    if "registration" not in combined:
        combined["registration"] = None
    combined["airline"] = combined["airline"].fillna("Unknown airline").replace("", "Unknown airline")
    unknown_airlines = combined["airline"].eq("Unknown airline")
    combined.loc[unknown_airlines, "airline"] = combined.loc[unknown_airlines, "flight_id"].map(
        lambda flight_id: resolve_callsign_airline(flight_id, {})
    )
    # Registration/type/manufacturer enrichment happens once, offline, at
    # ingestion time (scripts/ingest_flights.py, backed by the local OpenSky
    # registry) and via scripts/backfill_metadata.py for older rows. This
    # function used to also re-enrich here via live hexdb.io calls — up to
    # 50 serial network requests with a 10s timeout each, so a single cold
    # cache render could block for minutes. Removed: it duplicated work the
    # ingestion pipeline already does reliably, without the network risk.
    defaults = {
        "icao_hex": "",
        "altitude_ft": None,
        "speed_knots": None,
        "heading_deg": None,
        "squawk": "",
        "aircraft_category": "Unknown",
        "registration": None,
        "aircraft_type": None,
        "manufacturer": None,
        "wake_category_description": None,
    }
    for column, default in defaults.items():
        if column not in combined:
            combined[column] = default
    return combined


@st.cache_data(ttl=180)
def get_data():
    load_error = None
    if not list(DATA_DIRECTORY.glob("bronze_flights*.parquet")):
        try:
            ingest_adsb_lol_data()
        except Exception as error:
            load_error = str(error)

    if not list(DATA_DIRECTORY.glob("bronze_flights*.parquet")):
        return pd.DataFrame(), load_error or "No local flight snapshot is available."

    try:
        df = read_snapshot_data()
    except Exception as error:
        return pd.DataFrame(), f"Could not read the local flight snapshot: {error}"

    summary = summarize_live_flights(df)
    if summary.empty:
        return summary, load_error

    summary["flight_date"] = pd.to_datetime(summary["flight_date"], utc=True).dt.date
    return (
        summary.sort_values(["flight_date", "total_aircraft"], ascending=[True, False]).reset_index(drop=True),
        load_error,
    )


@st.cache_data(ttl=180)
def get_aircraft_snapshot():
    try:
        return read_snapshot_data()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_inferred_network(snapshot):
    # Shared across get_analytics_bundle and get_hyperbolic_layout so this
    # O(n^2)-ish per-flight_id pass over the whole snapshot runs once per
    # cache window instead of once per caller (it was ~5s each on 6k+ rows).
    return inferred_network(snapshot)


@st.cache_data(ttl=300, show_spinner=False)
def get_analytics_bundle(snapshot, airport_code):
    return {
        "forecast": forecast_airport(snapshot, airport_code),
        "decomposition": stl_decomposition(snapshot, airport_code),
        "residuals": residual_anomalies(snapshot, airport_code),
        "multivariate": multivariate_anomalies(snapshot),
        "change": change_points(snapshot, airport_code),
        "network": get_inferred_network(snapshot),
        "signatures": movement_signatures(snapshot),
        "benchmark": airport_benchmark(snapshot),
        "fleet": fleet_composition(snapshot),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_advanced_bundle(snapshot):
    return {
        "entropy": shannon_entropy(snapshot),
        "hawkes": hawkes_burst_score(snapshot),
        "pooled_rates": pooled_airport_rates(snapshot),
        "cep": cep_alerts(snapshot),
        "wake_separation": wake_separation_alerts(snapshot),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_smoothed_trajectory(snapshot, flight_id):
    return kalman_smooth_trajectory(snapshot, flight_id)


@st.cache_data(ttl=300, show_spinner=False)
def get_topology_bundle(snapshot):
    fingerprints = build_day_fingerprints(snapshot)
    return {
        "fingerprints": fingerprints,
        "embedding": project_days(fingerprints),
        "mapper": mapper_graph(fingerprints),
        "persistence": persistence_diagram(fingerprints),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_hyperbolic_layout(snapshot):
    network = get_inferred_network(snapshot)
    return build_hyperbolic_layout(network.data)


@st.cache_resource(ttl=300, show_spinner=False)
def get_trajectory_manifold(snapshot):
    return build_trajectory_manifold(snapshot)


if "refresh_seconds" not in st.session_state:
    st.session_state.refresh_seconds = 180
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "last_refresh_ts" not in st.session_state:
    st.session_state.last_refresh_ts = time.monotonic()

with st.sidebar:
    st.header("Explore the network")
    st.session_state.auto_refresh = st.toggle("Auto-refresh live feed", value=st.session_state.auto_refresh)
    st.session_state.refresh_seconds = st.slider(
        "Refresh every",
        min_value=60,
        max_value=600,
        value=st.session_state.refresh_seconds,
        step=30,
    )
    map_options = ["Airport activity", "Altitude bands", "Aircraft positions", "Flight trails"]
    map_options += ["3D altitude", "3D flight trails", "Network arcs"]
    map_options += ["Star glyphs", "Controlled airspace", "Wake vortex trails", "Density field (KDE)", "Flow field"]
    requested_map = st.query_params.get("map")
    map_mode = st.selectbox(
        "Map layer",
        map_options,
        index=map_options.index(requested_map) if requested_map in map_options else 0,
    )
    st.caption("Source: ADSB.lol. Counts are unique aircraft callsigns per day.")

    if st.button("Refresh now", width="stretch"):
        try:
            ingest_adsb_lol_data()
        except Exception as error:
            st.error(f"Refresh failed: {error}")
        else:
            st.cache_data.clear()
            st.session_state.last_refresh_ts = time.monotonic()
            st.rerun()

summary_df, data_error = get_data()

if data_error:
    st.warning(f"Live feed unavailable. Showing the latest local snapshot when possible. Details: {data_error}")

if summary_df.empty:
    st.error("No aircraft data is available yet.")
    if st.button("Try loading the live feed again", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if st.session_state.auto_refresh and time.monotonic() - st.session_state.last_refresh_ts >= st.session_state.refresh_seconds:
    try:
        ingest_adsb_lol_data()
    except Exception as error:
        st.warning(f"Automatic refresh failed: {error}")
        st.session_state.last_refresh_ts = time.monotonic()
    else:
        st.cache_data.clear()
        st.session_state.last_refresh_ts = time.monotonic()
        st.rerun()

available_dates = sorted(summary_df["flight_date"].unique())
# flight_date (and every value in available_dates) is derived from UTC
# timestamps everywhere else in this app.
today = datetime.now(timezone.utc).date()

# This is a *live* tracker, not a historical archive browser: with only a
# day or so of continuous collection behind it (scripts/ingest_flights.py
# runs via cron every 5 minutes), a full date picker mostly just exposed
# empty days and confused more than it helped. Always show the latest
# collected day — simplest thing that's actually true to "what's flying
# right now" — and revisit a proper history browser once there are weeks
# of real days to make it worthwhile.
selected_date = max(available_dates)
aircraft_snapshot = get_aircraft_snapshot()
st.query_params["map"] = map_mode

date_summary = summary_df[summary_df["flight_date"] == selected_date][
    ["destination_city", "center_lat", "center_lon", "total_aircraft"]
].copy()

city_catalog = (
    pd.DataFrame.from_dict(AIRPORTS, orient="index")
    .groupby("city", as_index=False)
    .agg(center_lat=("lat", "mean"), center_lon=("lon", "mean"))
    .rename(columns={"city": "destination_city"})
)
filtered_df = city_catalog.merge(date_summary, on="destination_city", how="left", suffixes=("_catalog", ""))
filtered_df["center_lat"] = filtered_df["center_lat"].fillna(filtered_df["center_lat_catalog"])
filtered_df["center_lon"] = filtered_df["center_lon"].fillna(filtered_df["center_lon_catalog"])
filtered_df = filtered_df.drop(columns=["center_lat_catalog", "center_lon_catalog"])
filtered_df["flight_date"] = selected_date
filtered_df["total_aircraft"] = filtered_df["total_aircraft"].fillna(0).astype(int)

city_options = sorted(city_catalog["destination_city"].tolist())
selected_cities = st.multiselect("Filter cities", options=city_options, default=city_options)
filtered_df = filtered_df[filtered_df["destination_city"].isin(selected_cities)].copy()

if filtered_df.empty:
    st.warning("No cities match your current selection.")
    st.stop()

cities = filtered_df[["destination_city", "total_aircraft"]].sort_values("total_aircraft", ascending=False)

selected_aircraft = aircraft_snapshot[
    pd.to_datetime(aircraft_snapshot["timestamp"], utc=True).dt.date == selected_date
].copy()
selected_aircraft = selected_aircraft[selected_aircraft["destination_city"].isin(selected_cities)]
known_airlines = selected_aircraft[~selected_aircraft["airline"].isin(NON_AIRLINE_LABELS)]["airline"].nunique()
airborne_count = int((~selected_aircraft["on_ground"]).sum()) if "on_ground" in selected_aircraft else 0
emergency_squawks = {"7500": "Unlawful interference", "7600": "Radio failure", "7700": "General emergency"}
emergency_aircraft = selected_aircraft[selected_aircraft["squawk"].isin(emergency_squawks)].copy()

if not emergency_aircraft.empty:
    st.error(
        f"Emergency squawk detected: {len(emergency_aircraft)} aircraft broadcasting "
        f"{', '.join(sorted(emergency_aircraft['squawk'].unique()))}. Verify through an authoritative source."
    )

city_aircraft = selected_aircraft.groupby("destination_city").agg(
    city_aircraft=("flight_id", "nunique"),
    airline_diversity=("airline", lambda values: values[~values.isin(NON_AIRLINE_LABELS)].nunique()),
    airborne=("on_ground", lambda values: int((~values).sum())),
).reset_index()
cities = cities.merge(city_aircraft, on="destination_city", how="left").fillna(0)
max_activity = max(float(cities["city_aircraft"].max()), 1)
max_diversity = max(float(cities["airline_diversity"].max()), 1)
cities["pulse_score"] = (
    (cities["city_aircraft"] / max_activity * 60)
    + (cities["airline_diversity"] / max_diversity * 25)
    + (cities["airborne"] / cities["city_aircraft"].replace(0, 1) * 15)
).round().astype(int)
altitude_values = pd.to_numeric(selected_aircraft["altitude_ft"], errors="coerce")
altitude_weight = (1 - altitude_values.clip(lower=0, upper=40000) / 40000).fillna(0.5)
overflight_index = round(
    float((altitude_weight * (~selected_aircraft["on_ground"])).sum())
    / max(len(city_catalog), 1)
    * 100,
    1,
)

# Live "mood" theming: a parametric color-grade coupled to the real anomaly
# signal (see scripts/mood_theme.py for why this replaces neural style
# transfer here — the coupling is what matters, not the generative technique).
mood_score = compute_mood_score(
    emergency_count=len(emergency_aircraft),
    total_aircraft=int(cities["total_aircraft"].sum()) if not cities.empty else 0,
    overflight_index=overflight_index,
)
st.markdown(mood_css(mood_score), unsafe_allow_html=True)

flight_options = sorted(selected_aircraft["flight_id"].dropna().unique())
follow_flight = st.selectbox("Follow aircraft", ["None"] + flight_options)
followed_aircraft = selected_aircraft[selected_aircraft["flight_id"] == follow_flight]

with st.expander("Watchlist and geofence"):
    watch_options = ["None"] + sorted(
        set(selected_aircraft["flight_id"].dropna())
        | set(selected_aircraft.loc[~selected_aircraft["airline"].isin(NON_AIRLINE_LABELS), "airline"].dropna())
    )
    watched_target = st.selectbox("Watch flight or airline", watch_options)
    if watched_target != "None":
        watched_rows = selected_aircraft[
            (selected_aircraft["flight_id"] == watched_target)
            | (selected_aircraft["airline"] == watched_target)
        ]
        if watched_rows.empty:
            st.info("No matching observation in this snapshot.")
        else:
            st.success(f"Watchlist match: {len(watched_rows)} aircraft observation(s).")

    geo_col1, geo_col2, geo_col3 = st.columns(3)
    geo_lat = geo_col1.number_input("Geofence latitude", value=33.0, format="%.4f")
    geo_lon = geo_col2.number_input("Geofence longitude", value=-96.0, format="%.4f")
    geo_radius = geo_col3.number_input("Radius (km)", min_value=1.0, max_value=500.0, value=50.0)
    geofence_rows = selected_aircraft.dropna(subset=["lat", "lon"]).copy()
    if not geofence_rows.empty:
        geofence_rows["distance_km"] = geofence_rows.apply(
            lambda row: distance_km(geo_lat, geo_lon, row["lat"], row["lon"]), axis=1
        )
        inside_geofence = geofence_rows[geofence_rows["distance_km"] <= geo_radius]
        st.metric("Aircraft inside geofence", int(inside_geofence["flight_id"].nunique()))

snapshot_date_label = selected_date.strftime("%Y-%m-%d")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Snapshot date", snapshot_date_label)
col2.metric("Aircraft observed", int(cities["total_aircraft"].sum()))
col3.metric("Airborne now", airborne_count)
col4.metric("Monitored cities", int(cities["destination_city"].nunique()))
col5.metric("Overflight index", overflight_index)
col6.metric("Airspace mood", mood_label(mood_score))

with st.expander("Live radar scope"):
    radar_points = selected_aircraft.dropna(subset=["lat", "lon"]).copy()
    if radar_points.empty:
        st.info("No position points are available for the radar scope.")
    else:
        center_lat = radar_points["lat"].mean()
        center_lon = radar_points["lon"].mean()
        lat_span = max(radar_points["lat"].max() - radar_points["lat"].min(), 0.1)
        lon_span = max(radar_points["lon"].max() - radar_points["lon"].min(), 0.1)
        radar_blips = "".join(
            f'<span style="position:absolute;left:{50 + (row.lon - center_lon) / lon_span * 70:.1f}%;'
            f'top:{50 - (row.lat - center_lat) / lat_span * 70:.1f}%;width:7px;height:7px;'
            f'border-radius:50%;background:{"#ef5b67" if row.squawk in emergency_squawks else "#6ef0a7"};'
            'box-shadow:0 0 10px currentColor;"></span>'
            for row in radar_points.itertuples()
        )
        st.markdown(
            '<div style="position:relative;width:100%;max-width:520px;aspect-ratio:1;margin:auto;'
            'border:1px solid rgba(110,240,167,.45);border-radius:50%;'
            'background:radial-gradient(circle,rgba(42,130,104,.16) 0 2%,transparent 2.5% 28%,'
            'rgba(42,130,104,.10) 28.5% 29%,transparent 29.5% 49%,rgba(42,130,104,.10) 49.5% 50%,transparent 50.5%);'
            'overflow:hidden;">'
            '<div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:rgba(110,240,167,.25);"></div>'
            '<div style="position:absolute;top:50%;left:0;width:100%;height:1px;background:rgba(110,240,167,.25);"></div>'
            f'{radar_blips}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Relative position scope for the selected snapshot. Red blips indicate emergency squawks.")

if not cities.empty and cities["total_aircraft"].sum() > 0:
    busiest_city = cities.sort_values("total_aircraft", ascending=False).iloc[0]
    prior_dates = [available_date for available_date in available_dates if available_date < selected_date]
    if prior_dates:
        prior_date = max(prior_dates)
        prior_total = int(summary_df.loc[summary_df["flight_date"] == prior_date, "total_aircraft"].sum())
        change = int(busiest_city["total_aircraft"]) - prior_total
        comparison = f"{abs(change)} more" if change >= 0 else f"{abs(change)} fewer"
        st.info(
            f"Daily insight: {busiest_city['destination_city']} has the most observed activity "
            f"({int(busiest_city['total_aircraft'])} aircraft), {comparison} than the previous collected day."
        )
    else:
        st.info(
            f"Daily insight: {busiest_city['destination_city']} is the busiest monitored city "
            f"with {int(busiest_city['total_aircraft'])} aircraft observed."
        )

st.link_button(
    "Share this view",
    f"?map={map_mode.replace(' ', '%20')}",
)

last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
remaining_seconds = max(st.session_state.refresh_seconds - (time.monotonic() - st.session_state.last_refresh_ts), 0)
st.markdown(f"**Last updated:** {last_updated}   •   **Next refresh in:** {remaining_seconds:.0f}s")

trend_df = summary_df.groupby("flight_date", as_index=False)["total_aircraft"].sum().rename(columns={"flight_date": "date", "total_aircraft": "aircraft"})
trend_df["date"] = pd.to_datetime(trend_df["date"], utc=True)

# Highest-activity monitored cities
city_rank = cities.reset_index(drop=True)
with st.container():
    top_cols = st.columns(min(3, len(city_rank)))
    for idx, col in enumerate(top_cols):
        if idx >= len(city_rank):
            break
        city = city_rank.iloc[idx]
        card_class = "top-card best" if idx == 0 else "top-card"
        badge_text = "Highest activity" if idx == 0 else f"Activity rank {idx + 1}"
        with col:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <h4>{badge_text}</h4>
                    <div class="value">{city['destination_city']}</div>
                    <div>{int(city['total_aircraft'])} aircraft</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

map_col, chart_col = st.columns([2, 1])
with map_col:
    map_layers = []
    skip_pydeck_render = False
    if map_mode == "Airport activity":
        map_layers.append(
            pdk.Layer(
                "HexagonLayer",
                data=filtered_df,
                get_position=["center_lon", "center_lat"],
                radius=35000,
                elevation_scale=150,
                extruded=True,
                auto_highlight=True,
            )
        )
        map_tooltip = {"html": "<b>{destination_city}</b><br>Aircraft observed: {total_aircraft}", "style": {"color": "white"}}
    elif map_mode in {"Flight trails", "3D flight trails"}:
        trail_source = aircraft_snapshot[
            aircraft_snapshot["destination_city"].isin(selected_cities)
        ].dropna(subset=["lat", "lon"]).copy()
        trail_source["timestamp"] = pd.to_datetime(trail_source["timestamp"], utc=True)
        trail_source["altitude_m"] = pd.to_numeric(trail_source["altitude_ft"], errors="coerce").fillna(0) * 0.3048
        trail_source = trail_source.sort_values("timestamp").groupby("flight_id").tail(10)
        path_columns = ["lon", "lat", "altitude_m"] if map_mode == "3D flight trails" else ["lon", "lat"]
        trail_data = (
            trail_source.groupby("flight_id")
            .apply(lambda rows: rows[path_columns].values.tolist())
            .rename("path")
            .reset_index()
        )
        map_layers.append(
            pdk.Layer(
                "PathLayer",
                data=trail_data,
                get_path="path",
                get_color=[72, 201, 176, 170] if map_mode == "Flight trails" else [90, 170, 240, 190],
                get_width=5,
                width_min_pixels=2,
                pickable=True,
            )
        )
        map_tooltip = {"html": "<b>{flight_id}</b><br>Last 10 recorded positions", "style": {"color": "white"}}
    elif map_mode == "Network arcs":
        network_edges = get_inferred_network(aircraft_snapshot).data
        arc_rows = []
        if network_edges is not None and not network_edges.empty:
            for edge in network_edges.itertuples():
                origin = AIRPORTS.get(edge.origin)
                destination = AIRPORTS.get(edge.destination)
                if origin and destination:
                    arc_rows.append(
                        {
                            "source_lon": origin["lon"],
                            "source_lat": origin["lat"],
                            "target_lon": destination["lon"],
                            "target_lat": destination["lat"],
                            "observations": edge.observations,
                            "origin": edge.origin,
                            "destination": edge.destination,
                        }
                    )
        arc_data = pd.DataFrame(arc_rows)
        map_layers.append(
            pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position=["source_lon", "source_lat"],
                get_target_position=["target_lon", "target_lat"],
                get_source_color=[72, 201, 176, 220],
                get_target_color=[245, 180, 70, 220],
                get_width="observations",
                width_min_pixels=2,
                pickable=True,
            )
        )
        map_tooltip = {"html": "<b>{origin} → {destination}</b><br>Inferred observations: {observations}", "style": {"color": "white"}}
    elif map_mode == "3D altitude":
        plot_df = selected_aircraft.dropna(subset=["lat", "lon"]).copy()
        plot_df["altitude_m"] = pd.to_numeric(plot_df["altitude_ft"], errors="coerce").fillna(0) * 0.3048
        plot_df["map_color"] = plot_df.apply(
            lambda row: [230, 64, 75, 230]
            if row["squawk"] in emergency_squawks
            else [70, 180, 240, 220],
            axis=1,
        )
        map_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=plot_df,
                get_position=["lon", "lat", "altitude_m"],
                get_fill_color="map_color",
                get_radius=7000,
                pickable=True,
            )
        )
        map_tooltip = {"html": "<b>{flight_id}</b><br>{airline}<br>Altitude: {altitude_ft} ft", "style": {"color": "white"}}
    elif map_mode == "Star glyphs":
        glyphs = compute_star_glyphs(selected_aircraft)
        polygon_rows = [
            {
                "airport_code": glyph.airport_code,
                "destination_city": glyph.destination_city,
                "polygon": glyph.polygon,
                "activity": glyph.metrics["activity"],
                "anomaly": glyph.metrics["anomaly"],
            }
            for glyph in glyphs
        ]
        spoke_rows = [
            {"source": spoke[0], "target": spoke[1], "airport_code": glyph.airport_code}
            for glyph in glyphs
            for spoke in glyph.spokes
        ]
        map_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=polygon_rows,
                get_polygon="polygon",
                get_fill_color="[80 + anomaly * 150, 197 - anomaly * 100, 151, 140]",
                get_line_color=[230, 240, 250, 200],
                line_width_min_pixels=1,
                pickable=True,
            )
        )
        map_layers.append(
            pdk.Layer(
                "LineLayer",
                data=spoke_rows,
                get_source_position="source",
                get_target_position="target",
                get_color=[230, 240, 250, 120],
                get_width=1,
            )
        )
        map_tooltip = {
            "html": "<b>{destination_city}</b><br>Activity: {activity}<br>Anomaly: {anomaly}",
            "style": {"color": "white"},
        }
        st.caption(
            "Bertin/Tufte star glyphs: one spoke per normalised metric "
            "(activity, airline diversity, airborne share, altitude mix, anomaly) per airport."
        )
    elif map_mode == "Controlled airspace":
        visible_codes = sorted(
            code for code, meta in AIRPORTS.items() if meta["city"] in selected_cities
        )
        volumes = airspace_volumes(visible_codes) if visible_codes else airspace_volumes()
        volume_rows = [
            {
                "polygon": volume_ring(volume),
                "elevation": volume.ceiling_m,
                "color": volume.color + [70],
                "label": volume.label,
            }
            for volume in volumes
        ]
        map_layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=volume_rows,
                get_polygon="polygon",
                get_elevation="elevation",
                extruded=True,
                get_fill_color="color",
                get_line_color=[255, 255, 255, 60],
                pickable=True,
            )
        )
        map_tooltip = {"html": "{label}", "style": {"color": "white"}}
        st.caption(
            "Approximate FAA-style Class B/C controlled-airspace volumes. Situational awareness only — "
            "not for navigation or flight planning."
        )
    elif map_mode == "Wake vortex trails":
        trails = compute_wake_trails(selected_aircraft)
        trail_frame = trails_to_frame(trails)
        if trail_frame.empty:
            st.info("No aircraft are moving fast enough in this snapshot to render wake trails.")
        else:
            trail_frame["altitude_m"] = trail_frame["altitude_m"] + 0.0
            map_layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=trail_frame,
                    get_position=["lon", "lat", "altitude_m"],
                    get_fill_color="fill_color",
                    get_radius="radius_m",
                    pickable=True,
                )
            )
        map_tooltip = {
            "html": "<b>{flight_id}</b><br>{wake_class} wake<br>Age: {age_seconds}s",
            "style": {"color": "white"},
        }
        st.caption(
            "Simplified wake-vortex decay Γ(t) = Γ₀·e^(−t/τ) by ICAO emitter category — "
            "illustrative, not a CFD-grade wake model (see scripts/wake_vortex.py)."
        )
    elif map_mode == "Density field (KDE)":
        band_option = st.selectbox(
            "Altitude band",
            ["Any altitude", "Ground–5,000 ft", "5,000–15,000 ft", "15,000–30,000 ft", "30,000 ft+"],
            key="density_band",
        )
        bands = {
            "Any altitude": None,
            "Ground–5,000 ft": (0, 5000),
            "5,000–15,000 ft": (5000, 15000),
            "15,000–30,000 ft": (15000, 30000),
            "30,000 ft+": (30000, 60000),
        }
        field = build_density_field(selected_aircraft, altitude_band_ft=bands[band_option])
        st.caption(field.message)
        map_tooltip = None
        skip_pydeck_render = True
        if field.ready:
            grid = field_to_grid_frame(field)
            density_fig = go.Figure(
                go.Densitymap(
                    lat=grid["lat"],
                    lon=grid["lon"],
                    z=grid["density"],
                    radius=28,
                    colorscale="Inferno",
                    opacity=0.75,
                    showscale=False,
                )
            )
            density_fig.update_layout(
                map=dict(
                    style="carto-darkmatter",
                    center=dict(lat=float(grid["lat"].mean()), lon=float(grid["lon"].mean())),
                    zoom=3.2,
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
            )
            st.plotly_chart(density_fig, width="stretch")
    elif map_mode == "Flow field":
        flow = compute_flow_field(selected_aircraft)
        arrow_rows = []
        lon_step = (flow.east - flow.west) / max(flow.columns - 1, 1)
        lat_step = (flow.north - flow.south) / max(flow.rows - 1, 1)
        scale = 0.02
        for row_index in range(flow.rows):
            for column_index in range(flow.columns):
                if flow.sample_counts[row_index][column_index] == 0:
                    continue
                origin_lon = flow.west + column_index * lon_step
                origin_lat = flow.south + row_index * lat_step
                east = flow.east_component[row_index][column_index]
                north = flow.north_component[row_index][column_index]
                arrow_rows.append(
                    {
                        "source": [origin_lon, origin_lat],
                        "target": [origin_lon + east * scale, origin_lat + north * scale],
                        "speed": flow.speed[row_index][column_index],
                    }
                )
        if arrow_rows:
            map_layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=arrow_rows,
                    get_source_position="source",
                    get_target_position="target",
                    get_color=[90, 200, 240, 200],
                    get_width=2,
                )
            )
        else:
            st.info("Not enough heading/speed telemetry in this snapshot to build a flow field.")
        map_tooltip = None
        st.caption("Aggregate aircraft heading/speed splatted onto a CONUS grid — each segment is the local mean flow vector.")
    else:
        plot_df = selected_aircraft.dropna(subset=["lat", "lon"]).copy()
        plot_df["map_color"] = plot_df.apply(
            lambda row: [230, 64, 75, 220]
            if row["squawk"] in emergency_squawks
            else ([47, 197, 151, 210] if map_mode == "Aircraft positions" else [60, 150, 220, 210]),
            axis=1,
        )
        map_layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=plot_df,
                get_position=["lon", "lat"],
                get_fill_color="map_color",
                get_radius=9000,
                pickable=True,
            )
        )
        map_tooltip = {
            "html": "<b>{flight_id}</b><br>{airline}<br>Altitude: {altitude_ft} ft<br>Speed: {speed_knots} kt<br>Squawk: {squawk}",
            "style": {"color": "white"},
        }
    if not skip_pydeck_render:
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=float(followed_aircraft.iloc[-1]["lat"]) if not followed_aircraft.empty else 33.0,
                    longitude=float(followed_aircraft.iloc[-1]["lon"]) if not followed_aircraft.empty else -96.0,
                    zoom=7.0 if not followed_aircraft.empty else 3.8,
                    pitch=45,
                ),
                layers=map_layers,
                tooltip=map_tooltip,
            )
        )

with chart_col:
    if len(trend_df) > 1:
        st.subheader("Daily activity")
        st.line_chart(trend_df.set_index("date")["aircraft"])
    else:
        st.info("Daily history will appear here as snapshots accumulate.")

    calendar = summary_df.pivot_table(
        index="flight_date",
        columns="destination_city",
        values="total_aircraft",
        aggfunc="sum",
        fill_value=0,
    )
    if not calendar.empty:
        st.subheader("Traffic calendar")
        st.caption("Rows are collected dates; columns are monitored cities. Higher values indicate more observed aircraft.")
        st.dataframe(calendar, width="stretch")

    st.subheader(f"Activity by monitored city · {snapshot_date_label}")

sort_metric = st.radio("Sort by", ["Total aircraft", "City name"], horizontal=True)
if sort_metric == "City name":
    cities = cities.sort_values("destination_city").reset_index(drop=True)
else:
    cities = cities.sort_values("total_aircraft", ascending=False).reset_index(drop=True)

city_table = cities.rename(
    columns={
        "destination_city": "Monitored city",
        "total_aircraft": "Aircraft observed",
        "pulse_score": "Pulse score",
        "airline_diversity": "Airlines observed",
    }
)
st.dataframe(
    city_table[["Monitored city", "Aircraft observed", "Pulse score", "Airlines observed"]],
    width="stretch",
    hide_index=True,
)

st.subheader("Airline mix")
if selected_aircraft.empty or known_airlines == 0:
    st.info("Airline identification is not available in this snapshot.")
else:
    airline_counts = (
        selected_aircraft[~selected_aircraft["airline"].isin(NON_AIRLINE_LABELS)]
        .groupby("airline", as_index=True)["flight_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(airline_counts, horizontal=True)

with st.expander("How the overflight index works"):
    st.write(
        "The overflight index weights airborne observations more heavily at lower reported "
        "altitudes, then normalizes by the number of monitored cities. It is a relative "
        "activity and potential exposure proxy, not a decibel reading, noise map, or "
        "environmental impact assessment."
    )

with st.expander("Populate older dates"):
    st.write(
        "The live feed creates history from the moment the dashboard starts collecting. "
        "To add older ADS-B archive data, extract the archive and run the importer from "
        "the project root:"
    )
    st.code(
        "python scripts/ingest_adsb_history.py /path/to/extracted/archive",
        language="bash",
    )
    st.caption("Imported parquet chunks are picked up automatically after the next refresh.")

with st.expander("Metadata sources"):
    st.write(
        "Airline names resolve from a local OpenFlights airlines.dat file. Registration, manufacturer, "
        "model, and wake-turbulence class resolve from a local OpenSky Network aircraft database — a "
        "full ~520k-tail-number snapshot joined locally by ICAO24 hex, no per-aircraft network call. "
        "HexDB is now only a last-resort fallback for hex codes neither file has."
    )
    st.markdown(
        "[OpenFlights airlines.dat](https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat)  "
        "| [OpenSky aircraft database](https://opensky-network.org/datasets/metadata/)  "
        "| [HexDB aircraft API](https://hexdb.io/api/v1/aircraft/)"
    )
    st.code(
        "curl -L https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat -o data/airlines.dat\n"
        "curl -L https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv "
        "-o data/opensky_aircraft_database.csv\n"
        "python scripts/backfill_metadata.py  # re-enrich rows already on disk",
        language="bash",
    )
    st.caption("HexDB lookups are limited to unknown identifiers and cached locally. Set AIRSPACE_ENABLE_HEXDB=0 to disable them.")
    st.markdown(
        "Reference datasets: [OpenSky metadata](https://opensky-network.org/datasets/metadata/), "
        "[BTS T-100](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMF), "
        "[FAA airport data](https://www.faa.gov/airports/airport_safety/airportdata_5010/), "
        "[FAA registry](https://registry.faa.gov/aircraftinquiry/)"
    )

with st.expander("Aircraft passport"):
    passport_options = sorted(selected_aircraft["flight_id"].dropna().unique())
    if not passport_options:
        st.info("No aircraft profiles are available for this snapshot.")
    else:
        passport_id = st.selectbox("Select aircraft", passport_options)
        passport_rows = aircraft_snapshot[aircraft_snapshot["flight_id"] == passport_id].sort_values("timestamp")
        latest = passport_rows.iloc[-1]
        passport_col1, passport_col2, passport_col3 = st.columns(3)
        passport_col1.metric("Airline", latest["airline"])
        passport_col2.metric("Observations", len(passport_rows))
        passport_col3.metric("Category", latest["aircraft_category"])
        observed_cities = passport_rows["destination_city"].dropna().unique().tolist()
        if len(observed_cities) > 1:
            st.caption(
                "Observed near: "
                + ", ".join(observed_cities)
                + ". This is an observation pattern, not a confirmed route."
            )
        st.dataframe(
            passport_rows[["timestamp", "destination_city", "altitude_ft", "speed_knots", "heading_deg", "squawk"]].tail(20),
            width="stretch",
            hide_index=True,
        )

with st.expander("Inspect aircraft snapshot"):
    aircraft_df = get_aircraft_snapshot()
    if aircraft_df.empty:
        st.info("Aircraft-level details are not available in the current snapshot.")
    else:
        include_ground = st.checkbox("Include aircraft on ground", value=True)
        aircraft_cities = st.multiselect(
            "Aircraft cities",
            options=sorted(aircraft_df["destination_city"].dropna().unique()),
            default=sorted(set(selected_cities) & set(aircraft_df["destination_city"].dropna().unique())),
        )
        visible_aircraft = aircraft_df[aircraft_df["destination_city"].isin(aircraft_cities)].copy()
        if not include_ground and "on_ground" in visible_aircraft:
            visible_aircraft = visible_aircraft[~visible_aircraft["on_ground"]]
        if visible_aircraft.empty:
            st.info("No aircraft match these filters.")
        else:
            if "airline" not in visible_aircraft:
                visible_aircraft["airline"] = "Unknown airline"
            columns = [
                "flight_id",
                "airline",
                "registration",
                "aircraft_type",
                "destination_city",
                "airport_code",
                "timestamp",
                "on_ground",
                "lat",
                "lon",
            ]
            st.dataframe(visible_aircraft[columns].sort_values("destination_city"), width="stretch")

with st.expander("Forecasting, anomalies, and network analytics"):
    analytics_airport = st.selectbox(
        "Airport for time-series analysis",
        sorted(aircraft_snapshot["airport_code"].dropna().unique()),
    )
    forecast_tab, anomaly_tab, network_tab, benchmark_tab, fleet_tab = st.tabs(
        ["Forecast", "Anomalies", "Network", "Benchmarks", "Fleet"]
    )
    analytics = get_analytics_bundle(aircraft_snapshot, analytics_airport)
    with forecast_tab:
        forecast = analytics["forecast"]
        decomposition = analytics["decomposition"]
        residuals = analytics["residuals"]
        st.caption(forecast.message)
        if forecast.ready:
            st.line_chart(forecast.data.set_index("timestamp")["predicted_aircraft"])
        else:
            st.info("Collect at least 48 hourly observations to activate forecasting.")
        st.caption(decomposition.message)
        if decomposition.ready:
            st.line_chart(decomposition.data.set_index("timestamp")[["observed", "trend", "seasonal"]])
        st.caption(residuals.message)
        if residuals.ready:
            st.dataframe(residuals.data[residuals.data["anomaly"]].tail(10), width="stretch", hide_index=True)
    with anomaly_tab:
        multivariate = analytics["multivariate"]
        change = analytics["change"]
        st.caption(multivariate.message)
        if multivariate.ready:
            st.dataframe(multivariate.data[multivariate.data["is_anomaly"]], width="stretch", hide_index=True)
        else:
            st.info("Collect at least 20 airport-days to activate multivariate anomaly detection.")
        st.caption(change.message)
        if change.ready:
            st.dataframe(change.data, width="stretch", hide_index=True)
        else:
            st.info("Collect at least 72 hourly observations to activate change-point detection.")
    with network_tab:
        network = analytics["network"]
        signatures = analytics["signatures"]
        st.caption(network.message)
        if network.ready:
            st.dataframe(network.data, width="stretch", hide_index=True)
        else:
            st.info("More repeated observations near multiple airports are needed to infer network edges.")
        st.caption(signatures.message)
        if signatures.ready:
            st.dataframe(signatures.data, width="stretch", hide_index=True)
        else:
            st.info("Collect telemetry for at least 10 aircraft to cluster movement signatures.")
    with benchmark_tab:
        benchmark = analytics["benchmark"]
        st.caption(benchmark.message)
        if benchmark.ready:
            st.dataframe(benchmark.data, width="stretch", hide_index=True)
        else:
            st.info("Collect at least 3 days before comparing airports with bootstrap intervals.")
    with fleet_tab:
        fleet = analytics["fleet"]
        st.caption(fleet.message)
        if fleet.ready:
            fleet_col1, fleet_col2 = st.columns(2)
            with fleet_col1:
                st.markdown("**By manufacturer**")
                st.bar_chart(fleet.by_manufacturer.set_index("manufacturer")["aircraft_count"].head(12), horizontal=True)
            with fleet_col2:
                st.markdown("**By wake-turbulence class**")
                st.bar_chart(fleet.by_wake_class.set_index("wake_class")["aircraft_count"])
            st.markdown("**Most common aircraft types**")
            st.dataframe(fleet.by_type, width="stretch", hide_index=True)
        else:
            st.info(
                "Run `python scripts/backfill_metadata.py` after downloading "
                "data/opensky_aircraft_database.csv to populate fleet data."
            )

with st.expander("Advanced signal processing lab"):
    advanced = get_advanced_bundle(aircraft_snapshot)
    signal_tab, information_tab, pooling_tab, alerts_tab = st.tabs(
        ["Trajectory filter", "Information flow", "Partial pooling", "CEP alerts"]
    )
    with signal_tab:
        trajectory_flights = sorted(aircraft_snapshot["flight_id"].dropna().unique())
        if trajectory_flights:
            trajectory_flight = st.selectbox("Aircraft trajectory", trajectory_flights)
            smoothed = get_smoothed_trajectory(aircraft_snapshot, trajectory_flight)
            predicted = predict_trajectory(smoothed)
            if len(smoothed) < 2:
                st.info("At least two time-separated pings are needed for velocity estimation.")
            else:
                trajectory_table = smoothed[["timestamp", "raw_lat", "raw_lon", "filtered_lat", "filtered_lon"]]
                st.line_chart(trajectory_table.set_index("timestamp")[["raw_lat", "filtered_lat", "raw_lon", "filtered_lon"]])
                st.caption("Constant-velocity Kalman estimate. Predicted positions are short-horizon estimates, not flight plans.")
                st.dataframe(predicted, width="stretch", hide_index=True)

                uncertainty = trajectory_uncertainty(smoothed)
                if not uncertainty.empty:
                    st.caption(
                        "Confidence ellipses (95%) from the Kalman covariance — honest uncertainty geometry "
                        "instead of a dimensionless dot per position."
                    )
                    ellipse_rows = [
                        {"ring": row.ring, "confidence_radius_m": row.confidence_radius_m}
                        for row in uncertainty.itertuples()
                    ]
                    path_row = [[lon, lat] for lat, lon in zip(smoothed["filtered_lat"], smoothed["filtered_lon"])]
                    st.pydeck_chart(
                        pdk.Deck(
                            initial_view_state=pdk.ViewState(
                                latitude=float(smoothed["filtered_lat"].iloc[-1]),
                                longitude=float(smoothed["filtered_lon"].iloc[-1]),
                                zoom=9,
                                pitch=0,
                            ),
                            layers=[
                                pdk.Layer(
                                    "PolygonLayer",
                                    data=ellipse_rows,
                                    get_polygon="ring",
                                    get_fill_color=[90, 170, 240, 60],
                                    get_line_color=[90, 170, 240, 160],
                                    stroked=True,
                                ),
                                pdk.Layer(
                                    "PathLayer",
                                    data=[{"path": path_row}],
                                    get_path="path",
                                    get_color=[240, 240, 250, 220],
                                    get_width=3,
                                    width_min_pixels=2,
                                ),
                            ],
                        )
                    )
    with information_tab:
        st.subheader("Airline diversity entropy")
        st.dataframe(advanced["entropy"], width="stretch", hide_index=True)
        airport_options = sorted(aircraft_snapshot["airport_code"].dropna().unique())
        if len(airport_options) >= 2:
            source_airport = st.selectbox("Source airport", airport_options)
            target_options = [airport for airport in airport_options if airport != source_airport]
            target_airport = st.selectbox("Target airport", target_options)
            transfer_value = transfer_entropy(aircraft_snapshot, source_airport, target_airport)
            st.metric("Transfer entropy (bits)", "Insufficient history" if transfer_value is None else f"{transfer_value:.3f}")
            st.caption("Transfer entropy measures predictive information at the selected one-hour lag; it is not proof of causality.")
    with pooling_tab:
        st.caption("Gamma-Poisson partial pooling shrinks small-sample airport rates toward the network mean.")
        st.dataframe(advanced["pooled_rates"], width="stretch", hide_index=True)
    with alerts_tab:
        if advanced["cep"].empty:
            st.info("No CEP rules have fired in the current snapshot.")
        else:
            st.dataframe(advanced["cep"].sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
        st.caption("CEP rules currently cover emergency squawks and local three-aircraft bursts in a two-minute window.")

        st.markdown("**Wake-turbulence separation risk**")
        if advanced["wake_separation"].empty:
            st.info("No same-direction pairs closer than their ICAO-style wake minimum in this snapshot.")
        else:
            st.dataframe(advanced["wake_separation"], width="stretch", hide_index=True)
        st.caption(
            "Simplified Heavy/Large/Small ICAO-style radar separation minima, checked against real "
            "registry-backed wake classes (scripts/aircraft_registry.py). Snapshot-based proximity, not a "
            "verified along-track encounter — situational awareness only, not an ATC tool."
        )

st.header("🔬 Research Lab")
st.caption(
    "Experimental, research-grade views on the same accumulated data. Each one is captioned with what it "
    "actually is and, where a technique doesn't honestly fit this data (a scene-reconstruction model, a "
    "GPU style-transfer pass), what grounded substitute was used instead — see the module docstrings."
)

DARK_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
)

with st.expander("Shape of the traffic record — topology"):
    topo_tab, persistence_tab = st.tabs(["Day-fingerprint embedding & Mapper", "Persistent homology"])
    topology_bundle = get_topology_bundle(aircraft_snapshot)

    with topo_tab:
        embedding = topology_bundle["embedding"]
        st.caption(embedding.message)
        if embedding.ready:
            fig = go.Figure(
                go.Scatter(
                    x=embedding.embeddings["x"],
                    y=embedding.embeddings["y"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=embedding.embeddings["hour_entropy"],
                        colorscale="Viridis",
                        colorbar=dict(title="hour entropy"),
                    ),
                    text=embedding.embeddings["date"].astype(str),
                    hovertemplate="%{text}<br>x=%{x:.2f} y=%{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(**DARK_PLOTLY_LAYOUT, height=380, title="Each point is one collected day")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Collect more distinct days to activate the day-fingerprint embedding.")

        mapper = topology_bundle["mapper"]
        st.caption(mapper.message)
        if mapper.ready and not mapper.nodes.empty:
            fig = go.Figure()
            for edge in mapper.edges.itertuples():
                source = mapper.nodes.loc[mapper.nodes["node_id"] == edge.source].iloc[0]
                target = mapper.nodes.loc[mapper.nodes["node_id"] == edge.target].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=[source["x"], target["x"]],
                        y=[source["y"], target["y"]],
                        mode="lines",
                        line=dict(color="rgba(110,240,167,0.4)", width=1 + edge.overlap),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=mapper.nodes["x"],
                    y=mapper.nodes["y"],
                    mode="markers",
                    marker=dict(size=8 + mapper.nodes["size"] * 3, color=mapper.nodes["mean_aircraft"], colorscale="Turbo"),
                    text=mapper.nodes["dates"].apply(lambda dates: ", ".join(dates[:5])),
                    hovertemplate="Traffic state<br>%{text}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.update_layout(**DARK_PLOTLY_LAYOUT, height=380, title="Mapper graph — nodes are similar traffic states")
            st.plotly_chart(fig, width="stretch")

    with persistence_tab:
        persistence = topology_bundle["persistence"]
        st.caption(persistence.message)
        if persistence.ready and not persistence.barcode.empty:
            barcode = persistence.barcode.reset_index(drop=True)
            colors = {"H0": "#6ef0a7", "H1": "#5aaaf0"}
            fig = go.Figure()
            for row in barcode.itertuples():
                fig.add_trace(
                    go.Scatter(
                        x=[row.birth, row.death],
                        y=[row.Index, row.Index],
                        mode="lines",
                        line=dict(color=colors.get(row.dimension, "#f0c85a"), width=6),
                        showlegend=False,
                        hovertemplate=f"{row.dimension} lifetime={row.lifetime:.2f}<extra></extra>",
                    )
                )
            fig.update_layout(
                **DARK_PLOTLY_LAYOUT,
                height=max(320, len(barcode) * 14),
                title="Persistence barcode — longer bars are stable topological features",
                yaxis=dict(showticklabels=False, title=""),
                xaxis=dict(title="feature-space scale"),
            )
            st.plotly_chart(fig, width="stretch")

with st.expander("Hyperbolic airport network"):
    hyperbolic = get_hyperbolic_layout(aircraft_snapshot)
    st.caption(hyperbolic.message)
    if hyperbolic.ready:
        boundary_angle = np.linspace(0, 2 * np.pi, 100)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.cos(boundary_angle),
                y=np.sin(boundary_angle),
                mode="lines",
                line=dict(color="rgba(255,255,255,0.15)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        for edge in hyperbolic.edges.itertuples():
            fig.add_trace(
                go.Scatter(
                    x=[edge.x0, edge.x1],
                    y=[edge.y0, edge.y1],
                    mode="lines",
                    line=dict(color="rgba(90,170,240,0.35)", width=1 + min(edge.observations, 6)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=hyperbolic.nodes["x"],
                y=hyperbolic.nodes["y"],
                mode="markers+text",
                text=hyperbolic.nodes["airport_code"],
                textposition="top center",
                marker=dict(
                    size=10 + hyperbolic.nodes["observations"].clip(upper=40),
                    color=hyperbolic.nodes["depth"],
                    colorscale="Plasma",
                    colorbar=dict(title="hub depth"),
                ),
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        fig.update_layout(
            **DARK_PLOTLY_LAYOUT,
            height=520,
            title="Poincaré disk — hubs near center, spokes fan outward by generation",
            xaxis=dict(visible=False, range=[-1.05, 1.05]),
            yaxis=dict(visible=False, range=[-1.05, 1.05], scaleanchor="x"),
        )
        st.plotly_chart(fig, width="stretch")

with st.expander("Learned trajectory manifold"):
    manifold_result = get_trajectory_manifold(aircraft_snapshot)
    st.caption(manifold_result.message)
    if manifold_result.ready:
        fig = go.Figure(
            go.Scatter(
                x=manifold_result.manifold["manifold_x"],
                y=manifold_result.manifold["manifold_y"],
                mode="markers",
                marker=dict(size=9, color="#6ef0a7", opacity=0.8),
                text=manifold_result.manifold["flight_id"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            **DARK_PLOTLY_LAYOUT,
            height=420,
            title="Click a point to decode a synthetic trajectory for that region",
            xaxis=dict(title="latent dim 1"),
            yaxis=dict(title="latent dim 2"),
        )
        selection = st.plotly_chart(
            fig, width="stretch", on_select="rerun", selection_mode="points", key="manifold_chart"
        )
        points = (selection.get("selection", {}) or {}).get("points", []) if selection else []
        if points:
            click_x, click_y = points[0]["x"], points[0]["y"]
            st.caption(f"Decoding the neighbourhood around ({click_x:.2f}, {click_y:.2f})...")
            reconstructed = decode_trajectory_at(manifold_result, click_x, click_y)
            if not reconstructed.empty:
                path = list(zip(reconstructed["lon"], reconstructed["lat"]))
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=pdk.ViewState(
                            latitude=float(reconstructed["lat"].mean()),
                            longitude=float(reconstructed["lon"].mean()),
                            zoom=6,
                            pitch=0,
                        ),
                        layers=[
                            pdk.Layer(
                                "PathLayer",
                                data=[{"path": path}],
                                get_path="path",
                                get_color=[110, 240, 167, 220],
                                get_width=4,
                                width_min_pixels=3,
                            )
                        ],
                    )
                )
                st.dataframe(reconstructed, width="stretch", hide_index=True)
        else:
            st.caption("Click any point above to generate its synthetic trajectory.")

with st.expander("Digital twin: capacity what-if"):
    twin_airport = st.selectbox(
        "Airport to simulate", sorted(aircraft_snapshot["airport_code"].dropna().unique()), key="twin_airport"
    )
    capacity_multiplier = st.slider(
        "Capacity multiplier (1.0 = normal, <1.0 = simulated capacity cut e.g. a runway closure)",
        min_value=0.1,
        max_value=1.5,
        value=1.0,
        step=0.05,
    )
    twin = simulate_digital_twin(aircraft_snapshot, twin_airport, capacity_multiplier=capacity_multiplier)
    st.caption(twin.message)
    if twin.ready:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=twin.simulation["step"],
                y=twin.simulation["baseline_lambda"],
                mode="lines",
                name="Observed hourly average (baseline)",
                line=dict(color="#5aaaf0"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=twin.simulation["step"],
                y=twin.simulation["simulated_aircraft"],
                mode="lines+markers",
                name=f"Simulated at {capacity_multiplier:.2f}x capacity",
                line=dict(color="#6ef0a7"),
            )
        )
        fig.update_layout(
            **DARK_PLOTLY_LAYOUT,
            height=360,
            title=f"{twin_airport} — 24h Poisson-arrival twin vs. observed baseline",
            xaxis=dict(title="hour"),
            yaxis=dict(title="aircraft"),
        )
        st.plotly_chart(fig, width="stretch")

with st.expander("Generative art: live airspace as data sculpture"):
    st.caption(
        "Every dot is a real aircraft right now, anchored to real US geography — hover one for its callsign, "
        "altitude, and speed. Color is altitude (see the on-canvas key); the background wash shifts with the "
        "same live mood score as the rest of the dashboard."
    )
    art_html = render_generative_art(selected_aircraft, mood=mood_score)
    components.html(art_html, height=580, scrolling=False)

st.caption(f"Auto-refresh: {'on' if st.session_state.auto_refresh else 'off'}")
