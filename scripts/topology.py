"""Topology and high-dimensional structure of the traffic record.

Each day's full traffic pattern is encoded as a feature vector, then exposed as
two complementary structures:

* a 2-D embedding (UMAP when available, else t-SNE, else PCA) where every point
  is a day, so days that behaved alike cluster together;
* a Mapper graph over the same feature space, whose nodes are clusters of
  similar traffic states and whose edges mark transitions between them, revealing
  the shape of the state space rather than a flat table.

UMAP and t-SNE both need a handful of days before they are meaningful, so the
functions degrade to PCA and then to "not ready" as the record thins out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_DAYS_FOR_EMBEDDING = 3
MIN_DAYS_FOR_MAPPER = 4
MIN_DAYS_FOR_PERSISTENCE = 6
HOURLY_PROFILE_LENGTH = 24
EMBEDDING_METHODS = ("umap", "tsne", "pca")


@dataclass
class TopologyResult:
    ready: bool
    message: str
    embeddings: pd.DataFrame = field(default_factory=pd.DataFrame)
    nodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    edges: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = "none"


@dataclass
class PersistenceResult:
    ready: bool
    message: str
    barcode: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_day_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """Encode every day's traffic pattern as a flat feature vector."""
    from scripts.analytics import normalize_columns

    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    working = normalize_columns(df)
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    working["date"] = working["timestamp"].dt.date
    working["hour"] = working["timestamp"].dt.hour

    rows = []
    for day, group in working.groupby("date"):
        hourly = group.groupby("hour")["flight_id"].nunique().reindex(
            range(HOURLY_PROFILE_LENGTH), fill_value=0
        )
        total = int(hourly.sum())
        shares = hourly.to_numpy(dtype=float)
        profile_sum = shares.sum() or 1.0
        shares = shares / profile_sum
        entropy = float(-(shares[shares > 0] * np.log2(shares[shares > 0])).sum())
        record = {
            "date": day,
            "aircraft_count": group["flight_id"].nunique(),
            "airline_diversity": group["airline"].nunique(),
            "category_diversity": group["aircraft_category"].nunique(),
            "airborne_ratio": float(1 - group["on_ground"].mean()),
            "mean_altitude": float(pd.to_numeric(group["altitude_ft"], errors="coerce").mean()),
            "hour_entropy": entropy,
            "peak_hour": int(hourly.idxmax()) if total else -1,
        }
        for hour in range(HOURLY_PROFILE_LENGTH):
            record[f"hour_{hour:02d}"] = float(shares[hour])
        rows.append(record)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def feature_matrix(fingerprints: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return the scaled numeric feature matrix and the column names."""
    from sklearn.preprocessing import StandardScaler

    excluded = {"date", "peak_hour"}
    feature_columns = [
        column for column in fingerprints.columns if column not in excluded
    ]
    values = fingerprints[feature_columns].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    if len(fingerprints) < 2:
        return values, feature_columns
    scaled = StandardScaler().fit_transform(values)
    return scaled, feature_columns


def _embed(matrix: np.ndarray, preferred: str) -> tuple[np.ndarray, str]:
    """Project the feature matrix to 2-D, degrading gracefully."""
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], 2)), "none"

    order = [preferred] + [method for method in EMBEDDING_METHODS if method != preferred]
    for method in order:
        try:
            if method == "umap":
                import umap  # type: ignore

                reducer = umap.UMAP(n_components=2, n_neighbors=min(15, matrix.shape[0] - 1), random_state=42)
                return reducer.fit_transform(matrix), "umap"
            if method == "tsne" and matrix.shape[0] >= 5:
                from sklearn.manifold import TSNE

                perplexity = max(2.0, min(30.0, (matrix.shape[0] - 1) / 3))
                embedding = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
                return embedding.fit_transform(matrix), "tsne"
            if method == "pca":
                from sklearn.decomposition import PCA

                return PCA(n_components=2, random_state=42).fit_transform(matrix), "pca"
        except Exception:  # pragma: no cover - optional dependency failures
            continue
    return np.zeros((matrix.shape[0], 2)), "none"


def project_days(fingerprints: pd.DataFrame, method: str = "umap") -> TopologyResult:
    """Project each day into 2-D so similar days land near each other."""
    if fingerprints.empty or len(fingerprints) < MIN_DAYS_FOR_EMBEDDING:
        return TopologyResult(
            False,
            f"At least {MIN_DAYS_FOR_EMBEDDING} collected days are needed for a day-fingerprint projection.",
        )

    matrix, _ = feature_matrix(fingerprints)
    coordinates, used = _embed(matrix, method)
    embeddings = pd.DataFrame(
        {
            "date": fingerprints["date"],
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "aircraft_count": fingerprints["aircraft_count"].to_numpy(),
            "airline_diversity": fingerprints["airline_diversity"].to_numpy(),
            "hour_entropy": fingerprints["hour_entropy"].to_numpy(),
        }
    )
    return TopologyResult(
        True,
        f"Day fingerprints projected to 2-D with {used.upper()}.",
        embeddings=embeddings,
        method=used,
    )


def _cluster(interval_matrix: np.ndarray, max_clusters: int) -> np.ndarray:
    if len(interval_matrix) < 2:
        return np.zeros(len(interval_matrix), dtype=int)
    cluster_count = max(1, min(max_clusters, len(interval_matrix)))
    if cluster_count < 2:
        return np.zeros(len(interval_matrix), dtype=int)
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(
        interval_matrix
    )


def mapper_graph(
    fingerprints: pd.DataFrame,
    n_intervals: int = 5,
    overlap: float = 0.35,
    max_clusters: int = 3,
) -> TopologyResult:
    """Construct a Mapper graph from the day-fingerprint feature space."""
    if fingerprints.empty or len(fingerprints) < MIN_DAYS_FOR_MAPPER:
        return TopologyResult(
            False,
            f"At least {MIN_DAYS_FOR_MAPPER} collected days are needed for a Mapper graph.",
        )

    matrix, _ = feature_matrix(fingerprints)
    coordinates, used = _embed(matrix, "pca")
    filter_values = fingerprints["aircraft_count"].to_numpy(dtype=float)
    low, high = float(filter_values.min()), float(filter_values.max())
    if np.isclose(low, high):
        high = low + 1.0

    span = high - low
    interval_length = span / max(1.0, n_intervals - overlap * (n_intervals - 1))
    step = interval_length * (1 - overlap)

    node_records: list[dict] = []
    node_members: list[set[int]] = []
    interval_nodes: list[list[int]] = []

    for interval_index in range(n_intervals):
        start = low + interval_index * step
        stop = start + interval_length
        selected = np.where((filter_values >= start) & (filter_values <= stop))[0]
        current_nodes: list[int] = []
        if selected.size:
            labels = _cluster(matrix[selected], max_clusters)
            for label in range(int(labels.max()) + 1):
                member_indices = selected[labels == label]
                if member_indices.size == 0:
                    continue
                node_id = len(node_records)
                centroid = coordinates[member_indices].mean(axis=0)
                node_records.append(
                    {
                        "node_id": node_id,
                        "interval": interval_index,
                        "size": int(member_indices.size),
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                        "mean_aircraft": float(filter_values[member_indices].mean()),
                        "dates": [str(fingerprints["date"].iloc[index]) for index in member_indices],
                    }
                )
                node_members.append(set(int(index) for index in member_indices))
                current_nodes.append(node_id)
        interval_nodes.append(current_nodes)

    edge_records: list[dict] = []
    for previous, current in zip(interval_nodes[:-1], interval_nodes[1:]):
        for source in previous:
            for target in current:
                shared = node_members[source] & node_members[target]
                if shared:
                    edge_records.append(
                        {
                            "source": source,
                            "target": target,
                            "overlap": len(shared),
                        }
                    )

    return TopologyResult(
        True,
        "Mapper graph built over the day-fingerprint space (nodes are traffic states).",
        embeddings=pd.DataFrame(
            {
                "date": fingerprints["date"],
                "x": coordinates[:, 0],
                "y": coordinates[:, 1],
            }
        ),
        nodes=pd.DataFrame(node_records),
        edges=pd.DataFrame(edge_records),
        method=used,
    )


def persistence_diagram(fingerprints: pd.DataFrame, max_dimension: int = 1) -> PersistenceResult:
    """Compute persistent homology (H0/H1) on the day-fingerprint point cloud.

    Each day is a point in feature space; Vietoris-Rips persistent homology
    tracks, as the connection radius grows, which points merge into single
    connected components (H0) and which loops of "similar-but-distinct
    traffic states" open and close (H1). A long-lived bar in the barcode is a
    genuine structural feature of the traffic-pattern space (e.g. a stable
    cluster of "normal weekday" days); a short bar is noise. This is applied
    algebraic topology on the same feature space scripts.topology already
    builds for the UMAP/Mapper views, via the ripser library.
    """
    if fingerprints.empty or len(fingerprints) < MIN_DAYS_FOR_PERSISTENCE:
        return PersistenceResult(
            False,
            f"At least {MIN_DAYS_FOR_PERSISTENCE} collected days are needed for a persistence diagram.",
        )

    from ripser import ripser

    matrix, _ = feature_matrix(fingerprints)
    result = ripser(matrix, maxdim=max_dimension)
    rows = []
    for dimension, diagram in enumerate(result["dgms"]):
        for birth, death in diagram:
            if not np.isfinite(death):
                death = float(matrix.std() * 4) if matrix.size else birth + 1.0
            rows.append(
                {
                    "dimension": f"H{dimension}",
                    "birth": float(birth),
                    "death": float(death),
                    "lifetime": float(death - birth),
                }
            )
    barcode = pd.DataFrame(rows).sort_values(["dimension", "lifetime"], ascending=[True, False])
    return PersistenceResult(
        True,
        "Vietoris-Rips persistent homology over the day-fingerprint cloud: long bars are "
        "stable topological features (components/loops) of the traffic-pattern space, "
        "short bars are noise.",
        barcode=barcode.reset_index(drop=True),
    )
