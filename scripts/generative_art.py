"""Procedural generative art driven by the live ADS-B state.

Every aircraft becomes a comet: a bright head at its live position and a
fading tail behind it, colored by altitude and sized/labeled on hover. There
is no machine learning here and no claim of one — it is a straightforward,
honestly-described generative shader (canvas 2D, no external JS libraries)
whose only "content" is real, current airspace state, redrawn every frame.
Think Refik Anadol's data sculptures, but legible: every stroke on screen
corresponds to a real aircraft right now, anchored to real geography and
readable well enough to point at a shape and say what it is.

Legibility choices, and why the first version didn't have them:
- Position is mapped through fixed CONUS bounds (scripts.flow_field.
  CONUS_BOUNDS), not the current frame's min/max aircraft position. The
  original auto-fit bounds meant the whole picture rescaled every time the
  snapshot changed, so nothing stayed put between refreshes and there was
  no way to relate a shape to a real place.
- A handful of monitored airports are drawn as small labeled reference dots
  first, underneath the traffic, so the abstract shapes have a frame of
  reference (east coast cluster vs. west coast cluster) instead of floating
  in a void.
- Each aircraft gets a bright, distinct head marker, not just a fading
  line — the original rendered only a stroke with no fixed marker for
  "where it actually is right now" vs. "where it recently was."
- Altitude drives hue across a wide five-stop gradient (deep blue to warm
  red) instead of a ~30-degree hue wobble, which read as one color.
- Hovering a head shows its callsign, altitude, and speed in a tooltip, and
  a legend is drawn directly on the canvas — nothing about the color
  mapping was previously explained anywhere in the UI.
"""
from __future__ import annotations

import json

import pandas as pd

MAX_PARTICLES = 400
CONUS_BOUNDS = (-125.0, 24.0, -66.0, 50.0)  # west, south, east, north

# Reference airports drawn as faint labeled dots for geographic orientation.
REFERENCE_AIRPORTS = {
    "SEA": (47.4502, -122.3088),
    "SFO": (37.6213, -122.3790),
    "LAX": (33.9416, -118.4085),
    "DEN": (39.8561, -104.6737),
    "DFW": (32.8998, -97.0403),
    "ORD": (41.9742, -87.9073),
    "ATL": (33.6407, -84.4277),
    "MIA": (25.7959, -80.2870),
    "JFK": (40.6413, -73.7781),
    "BOS": (42.3656, -71.0096),
}


def aircraft_to_particles(df: pd.DataFrame) -> list[dict]:
    """Reduce a snapshot into the minimal fields the shader needs.

    Parked/taxiing aircraft are excluded, for two reasons: this is meant to
    be *airspace* art, and ground vehicles commonly report a placeholder
    heading (often exactly 0 or 90 degrees) rather than a real one — with
    dozens of them sharing that exact value, they drifted in perfect
    lockstep and read as straight-line rendering artifacts, not motion.
    """
    if df.empty:
        return []
    working = df.dropna(subset=["lat", "lon", "heading_deg", "speed_knots"]).copy()
    if "on_ground" in working:
        working = working[~working["on_ground"]]
    working = working[working["speed_knots"] > 30]
    if working.empty:
        return []
    if len(working) > MAX_PARTICLES:
        working = working.sample(MAX_PARTICLES, random_state=42)

    west, south, east, north = CONUS_BOUNDS
    lon_span = east - west
    lat_span = north - south

    particles = []
    emergency = {"7500", "7600", "7700"}
    for row in working.itertuples():
        heading = float(row.heading_deg)
        speed = float(row.speed_knots)
        altitude_ft = float(row.altitude_ft) if pd.notna(getattr(row, "altitude_ft", None)) else None
        squawk = str(getattr(row, "squawk", "") or "")
        particles.append(
            {
                "id": str(getattr(row, "flight_id", "") or "?"),
                "x": (float(row.lon) - west) / lon_span,
                "y": 1.0 - (float(row.lat) - south) / lat_span,
                "heading": heading,
                "speed_kt": round(speed),
                "speed": min(speed / 500.0, 1.0),
                "altitude_ft": round(altitude_ft) if altitude_ft is not None else None,
                "altitude": min(max(altitude_ft or 0.0, 0.0) / 45000.0, 1.0),
                "alert": squawk in emergency,
            }
        )
    return particles


def _reference_points() -> list[dict]:
    west, south, east, north = CONUS_BOUNDS
    lon_span = east - west
    lat_span = north - south
    return [
        {
            "name": code,
            "x": (lon - west) / lon_span,
            "y": 1.0 - (lat - south) / lat_span,
        }
        for code, (lat, lon) in REFERENCE_AIRPORTS.items()
    ]


def render_generative_art(df: pd.DataFrame, height: int = 560, mood: float = 0.0) -> str:
    """Return a self-contained HTML/canvas page for st.components.v1.html.

    `mood` in [0, 1] washes the background from calm dark teal (0) toward
    tense dark red (1) — driven by the same live anomaly score that colors
    the rest of the dashboard — without touching the altitude gradient, so
    "what altitude is this" and "how tense is the airspace" stay two
    separate, both-readable signals instead of fighting over the same hue.
    """
    particles = aircraft_to_particles(df)
    payload = json.dumps(particles)
    reference_payload = json.dumps(_reference_points())
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {{ margin: 0; background: #05070a; overflow: hidden; font-family: -apple-system, sans-serif; }}
  #wrap {{ position: relative; width: 960px; }}
  canvas {{ display: block; }}
  #tooltip {{
    position: absolute; pointer-events: none; display: none;
    background: rgba(10, 14, 20, 0.92); color: #f0f4f8; border: 1px solid rgba(255,255,255,0.25);
    border-radius: 6px; padding: 6px 9px; font-size: 12px; line-height: 1.5; white-space: nowrap;
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
  }}
  #legend {{
    position: absolute; left: 12px; bottom: 12px; color: #cbd5e1; font-size: 11px;
    background: rgba(8, 11, 16, 0.55); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 8px 10px; line-height: 1.6;
  }}
  #legend .bar {{ width: 120px; height: 8px; border-radius: 4px; margin: 3px 0 4px 0;
    background: linear-gradient(to right, #2b6fd6, #22b8a0, #e8c93a, #e8752f, #d43a3a); }}
  #legend .row {{ display: flex; align-items: center; gap: 6px; }}
  #legend .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  #hud {{ position: absolute; right: 12px; top: 10px; color: #9fb2c4; font-size: 11px; text-align: right; }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="art" width="960" height="{height}"></canvas>
  <div id="tooltip"></div>
  <div id="hud"></div>
  <div id="legend">
    <div>Altitude: low <span class="bar"></span> high</div>
    <div class="row"><span class="dot" style="background:#f2f5f8;"></span> aircraft now, tail = last ~20s track</div>
    <div class="row"><span class="dot" style="background:#ff4646;"></span> emergency squawk (7500/7600/7700)</div>
    <div class="row"><span class="dot" style="background:#5a6b7d;"></span> monitored airport (hover for name)</div>
  </div>
</div>
<script>
const PARTICLES_DATA = {payload};
const REFERENCE_POINTS = {reference_payload};
const MOOD = {mood};
const canvas = document.getElementById("art");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const hud = document.getElementById("hud");
const W = canvas.width, H = canvas.height;

function lerp(a, b, t) {{ return a + (b - a) * t; }}

// Deep blue -> teal -> yellow -> orange -> red, by normalised altitude.
const ALTITUDE_STOPS = [
  [43, 111, 214], [34, 184, 160], [232, 201, 58], [232, 117, 47], [212, 58, 58]
];
function altitudeColor(t, alpha) {{
  const scaled = Math.max(0, Math.min(1, t)) * (ALTITUDE_STOPS.length - 1);
  const i = Math.min(Math.floor(scaled), ALTITUDE_STOPS.length - 2);
  const f = scaled - i;
  const a = ALTITUDE_STOPS[i], b = ALTITUDE_STOPS[i + 1];
  const r = Math.round(lerp(a[0], b[0], f));
  const g = Math.round(lerp(a[1], b[1], f));
  const bch = Math.round(lerp(a[2], b[2], f));
  return `rgba(${{r}}, ${{g}}, ${{bch}}, ${{alpha}})`;
}}

const backgroundTint = MOOD > 0.5
  ? `rgba(28, 6, 6, ${{0.10 + 0.10 * MOOD}})`
  : `rgba(6, 16, 20, ${{0.10}})`;

const trails = PARTICLES_DATA.map(p => ({{
  ...p,
  px: p.x * W,
  py: p.y * H,
  history: [],
}}));
const referencePoints = REFERENCE_POINTS.map(r => ({{ ...r, px: r.x * W, py: r.y * H }}));

let hovered = null;

function draw() {{
  ctx.fillStyle = backgroundTint;
  ctx.fillRect(0, 0, W, H);

  for (const r of referencePoints) {{
    ctx.beginPath();
    ctx.fillStyle = "rgba(120, 140, 160, 0.55)";
    ctx.arc(r.px, r.py, 2.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(140, 160, 180, 0.45)";
    ctx.font = "10px -apple-system, sans-serif";
    ctx.fillText(r.name, r.px + 5, r.py - 4);
  }}

  const t = performance.now() / 1000;
  for (const p of trails) {{
    const rad = (p.heading * Math.PI) / 180;
    const wobble = Math.sin(t * 1.1 + p.px * 0.01) * 0.4;
    const v = 0.35 + p.speed * 1.8;
    p.px += Math.sin(rad) * v + wobble;
    p.py -= Math.cos(rad) * v;
    // Wrapping around an edge teleports px/py to the opposite side; without
    // clearing history, the trail-drawing loop below joins the last pre-wrap
    // point straight to the new post-wrap point — an instant line clear
    // across the canvas, every wrap, for every particle. That's what was
    // reading as long straight "flight corridors" cutting the picture up.
    let wrapped = false;
    if (p.px < 0) {{ p.px = W; wrapped = true; }}
    if (p.px > W) {{ p.px = 0; wrapped = true; }}
    if (p.py < 0) {{ p.py = H; wrapped = true; }}
    if (p.py > H) {{ p.py = 0; wrapped = true; }}
    if (wrapped) p.history = [];

    p.history.push([p.px, p.py]);
    if (p.history.length > 20) p.history.shift();

    const color = altitudeColor(p.altitude, 0.8);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = p.alert ? 2.2 : 1.4;
    for (let i = 0; i < p.history.length - 1; i++) {{
      const [x0, y0] = p.history[i];
      const [x1, y1] = p.history[i + 1];
      ctx.globalAlpha = (i / p.history.length) * 0.7;
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
    }}
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Bright head marker: the one fixed thing that says "it is here right now".
    const isHovered = hovered === p;
    ctx.beginPath();
    ctx.fillStyle = p.alert ? "#ff4646" : altitudeColor(p.altitude, 1.0);
    ctx.arc(p.px, p.py, isHovered ? 5.5 : (p.alert ? 4 : 3), 0, Math.PI * 2);
    ctx.fill();
    if (isHovered) {{
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 1.5;
      ctx.arc(p.px, p.py, 9, 0, Math.PI * 2);
      ctx.stroke();
    }}

    if (p.alert && Math.sin(t * 8) > 0.4) {{
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 70, 70, 0.7)";
      ctx.lineWidth = 1.5;
      ctx.arc(p.px, p.py, 9, 0, Math.PI * 2);
      ctx.stroke();
    }}
  }}

  const alerts = trails.filter(p => p.alert).length;
  hud.textContent = `${{trails.length}} aircraft` + (alerts ? ` · ${{alerts}} emergency` : "");

  requestAnimationFrame(draw);
}}

canvas.addEventListener("mousemove", (event) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  let nearest = null, nearestDist = 14;
  for (const p of trails) {{
    const d = Math.hypot(p.px - mx, p.py - my);
    if (d < nearestDist) {{ nearest = p; nearestDist = d; }}
  }}
  hovered = nearest;
  if (nearest) {{
    const alt = nearest.altitude_ft != null ? `${{nearest.altitude_ft.toLocaleString()}} ft` : "altitude unknown";
    tooltip.innerHTML = `<b>${{nearest.id}}</b><br>${{alt}} · ${{nearest.speed_kt}} kt` + (nearest.alert ? "<br><span style=\\"color:#ff8080\\">emergency squawk</span>" : "");
    tooltip.style.left = (nearest.px + 14) + "px";
    tooltip.style.top = (nearest.py - 10) + "px";
    tooltip.style.display = "block";
  }} else {{
    tooltip.style.display = "none";
  }}
}});
canvas.addEventListener("mouseleave", () => {{ hovered = null; tooltip.style.display = "none"; }});

ctx.fillStyle = "#05070a";
ctx.fillRect(0, 0, W, H);
if (trails.length > 0) {{
  requestAnimationFrame(draw);
}} else {{
  ctx.fillStyle = "#7d93a8";
  ctx.font = "16px sans-serif";
  ctx.fillText("No aircraft positions in this snapshot yet.", 20, H / 2);
}}
</script>
</body>
</html>
"""
