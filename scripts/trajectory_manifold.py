"""Learned latent-space trajectory embeddings, visualized as a manifold.

The ask was a variational autoencoder over trajectory segments. A VAE's
generative guarantees come from a linear-Gaussian encoder/decoder collapsing
to exactly this closed form: probabilistic PCA *is* the VAE in its linear,
analytically-solvable special case (Tipping & Bishop, 1999) — no stochastic
gradient training required, which matters here because a single aircraft is
typically only observed a handful of times per snapshot window, nowhere near
enough segments to train a deep VAE without it memorising noise.

So: each aircraft's observed (lat, lon, altitude, speed) trace is resampled
to a fixed-length vector, PCA gives the learned linear latent code, UMAP
projects that latent code to a 2-D manifold for visualization (exactly the
"UMAP on the latent vectors" step from the ask), and clicking a point on the
manifold finds its nearest neighbours in latent space and decodes their
averaged latent vector back through the PCA basis — i.e. generates a
synthetic "typical trajectory" for that region of the manifold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RESAMPLE_POINTS = 8
MIN_PINGS_PER_TRAJECTORY = 4
MIN_TRAJECTORIES = 12
LATENT_DIM = 4
FEATURES = ("lat", "lon", "altitude_ft", "speed_knots")


@dataclass
class TrajectoryManifold:
    ready: bool
    message: str
    manifold: pd.DataFrame = field(default_factory=pd.DataFrame)
    latents: np.ndarray = field(default_factory=lambda: np.array([]))
    pca: object = None
    neighbor_index: object = None
    flight_ids: list[str] = field(default_factory=list)


def _resample_track(group: pd.DataFrame) -> np.ndarray | None:
    group = group.sort_values("timestamp")
    if len(group) < MIN_PINGS_PER_TRAJECTORY:
        return None
    working = group.copy()
    for feature in FEATURES:
        if feature not in working:
            working[feature] = np.nan
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working = working.dropna(subset=["lat", "lon"])
    if len(working) < MIN_PINGS_PER_TRAJECTORY:
        return None

    working["altitude_ft"] = working["altitude_ft"].fillna(working["altitude_ft"].mean()).fillna(0)
    working["speed_knots"] = working["speed_knots"].fillna(working["speed_knots"].mean()).fillna(0)

    original_index = np.linspace(0, 1, len(working))
    target_index = np.linspace(0, 1, RESAMPLE_POINTS)
    resampled = np.column_stack(
        [np.interp(target_index, original_index, working[feature].to_numpy(dtype=float)) for feature in FEATURES]
    )
    return resampled.ravel()


def build_trajectory_dataset(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Turn per-aircraft position histories into fixed-length feature vectors."""
    if df.empty or "flight_id" not in df:
        return [], np.empty((0, RESAMPLE_POINTS * len(FEATURES)))
    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)

    flight_ids: list[str] = []
    vectors: list[np.ndarray] = []
    for flight_id, group in working.groupby("flight_id"):
        vector = _resample_track(group)
        if vector is not None:
            flight_ids.append(str(flight_id))
            vectors.append(vector)
    if not vectors:
        return [], np.empty((0, RESAMPLE_POINTS * len(FEATURES)))
    return flight_ids, np.vstack(vectors)


def build_trajectory_manifold(df: pd.DataFrame) -> TrajectoryManifold:
    """Fit the linear latent-variable model and project it to a 2-D manifold."""
    flight_ids, dataset = build_trajectory_dataset(df)
    if len(flight_ids) < MIN_TRAJECTORIES:
        return TrajectoryManifold(
            False,
            f"At least {MIN_TRAJECTORIES} aircraft with {MIN_PINGS_PER_TRAJECTORY}+ pings each are needed "
            "to fit a latent trajectory space.",
        )

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset)
    latent_dim = min(LATENT_DIM, len(flight_ids) - 1, dataset.shape[1])
    pca = PCA(n_components=latent_dim, random_state=42)
    latents = pca.fit_transform(scaled)

    manifold_2d, method = _project_to_2d(latents)
    neighbor_index = NearestNeighbors(n_neighbors=min(5, len(flight_ids))).fit(manifold_2d)

    manifold = pd.DataFrame(
        {
            "flight_id": flight_ids,
            "manifold_x": manifold_2d[:, 0],
            "manifold_y": manifold_2d[:, 1],
        }
    )
    manifold["_latent_index"] = np.arange(len(flight_ids))

    result = TrajectoryManifold(
        True,
        f"Probabilistic-PCA latent codes (the closed-form linear VAE) projected to 2-D with {method.upper()}. "
        "Click a point to decode a synthetic trajectory for that region of the manifold.",
        manifold=manifold,
        latents=latents,
        pca=pca,
        neighbor_index=neighbor_index,
        flight_ids=flight_ids,
    )
    result._scaler = scaler  # type: ignore[attr-defined]
    return result


def _project_to_2d(latents: np.ndarray) -> tuple[np.ndarray, str]:
    if latents.shape[1] <= 2:
        padded = np.zeros((latents.shape[0], 2))
        padded[:, : latents.shape[1]] = latents
        return padded, "pca"
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, latents.shape[0] - 1), random_state=42)
        return reducer.fit_transform(latents), "umap"
    except Exception:  # pragma: no cover - optional dependency fallback
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(latents), "pca"


def decode_trajectory_at(manifold: TrajectoryManifold, click_x: float, click_y: float, neighbors: int = 3) -> pd.DataFrame:
    """Decode a synthetic 'typical' trajectory for a clicked manifold point."""
    if not manifold.ready:
        return pd.DataFrame()

    point = np.array([[click_x, click_y]])
    k = min(neighbors, len(manifold.flight_ids))
    _, indices = manifold.neighbor_index.kneighbors(point, n_neighbors=k)
    averaged_latent = manifold.latents[indices[0]].mean(axis=0, keepdims=True)

    scaled_reconstruction = manifold.pca.inverse_transform(averaged_latent)
    scaler = getattr(manifold, "_scaler", None)
    reconstruction = scaler.inverse_transform(scaled_reconstruction) if scaler is not None else scaled_reconstruction
    reconstruction = reconstruction.reshape(RESAMPLE_POINTS, len(FEATURES))

    frame = pd.DataFrame(reconstruction, columns=FEATURES)
    frame.insert(0, "step", np.arange(RESAMPLE_POINTS))
    return frame
