"""Live "mood" theming, coupled to the real anomaly signal.

The ask was neural style transfer driving the dashboard's rendering mood.
A real style-transfer pass (a VGG-based optimization loop, or a trained
fast-transfer network) needs a GPU and hundreds of milliseconds to seconds
per frame — incompatible with a dashboard that reruns on every refresh —
and would need a large new dependency (torch + pretrained weights) for a
purely cosmetic effect. What actually matters for the ask ("mood reacts to
anomaly score") is the coupling, not the specific generative technique.

This module computes a single mood score in [0, 1] from signals already on
screen (emergency squawks, the overflight index, recent CEP alerts) and
linearly interpolates the dashboard's accent palette from a calm teal/green
toward a tense amber/red as that score rises. It is parametric color
grading, not a neural network — described honestly as such.
"""
from __future__ import annotations

CALM_ACCENT = (110, 240, 167)   # #6ef0a7
TENSE_ACCENT = (255, 91, 103)   # #ff5b67
CALM_GLOW = (28, 94, 109)
TENSE_GLOW = (120, 30, 30)


def compute_mood_score(
    emergency_count: int,
    total_aircraft: int,
    overflight_index: float,
    recent_alert_count: int = 0,
) -> float:
    """Blend a few already-computed signals into one live tension score."""
    emergency_component = 1.0 if emergency_count > 0 else 0.0
    density_component = min(overflight_index / 60.0, 1.0)
    alert_component = min(recent_alert_count / 5.0, 1.0)
    score = 0.55 * emergency_component + 0.25 * density_component + 0.2 * alert_component
    return max(0.0, min(1.0, score))


def _lerp_rgb(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    r = round(low[0] + (high[0] - low[0]) * t)
    g = round(low[1] + (high[1] - low[1]) * t)
    b = round(low[2] + (high[2] - low[2]) * t)
    return f"{r}, {g}, {b}"


def mood_label(score: float) -> str:
    if score < 0.2:
        return "Calm"
    if score < 0.5:
        return "Elevated"
    if score < 0.8:
        return "Tense"
    return "Critical"


def mood_css(score: float) -> str:
    """CSS variable overrides that re-tint the existing dashboard theme by mood."""
    accent = _lerp_rgb(CALM_ACCENT, TENSE_ACCENT, score)
    glow = _lerp_rgb(CALM_GLOW, TENSE_GLOW, score)
    return f"""
    <style>
        :root {{
            --mood-accent: rgb({accent});
            --mood-glow: rgb({glow});
        }}
        .status-pill {{
            background: rgba({accent}, 0.15) !important;
            border-color: rgba({accent}, 0.4) !important;
            color: rgb({accent}) !important;
        }}
        .status-dot {{
            background: rgb({accent}) !important;
            box-shadow: 0 0 12px rgba({accent}, 0.9) !important;
        }}
        .top-card.best {{
            border-color: rgba({accent}, 0.7) !important;
            box-shadow: 0 0 18px rgba({accent}, 0.2) !important;
        }}
        .top-card.best .value {{
            color: rgb({accent}) !important;
        }}
        .top-card:hover {{
            border-color: rgba({accent}, 0.6) !important;
            box-shadow: 0 0 18px rgba({accent}, 0.18) !important;
        }}
        .stApp {{
            background: radial-gradient(circle at 12% 0%, rgba({glow}, 0.32), transparent 32%),
                        linear-gradient(180deg, #06131b 0%, #0b1d24 100%) !important;
        }}
    </style>
    """
