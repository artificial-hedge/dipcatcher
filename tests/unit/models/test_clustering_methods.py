"""Clustering canon: k-means++, PAM, DBSCAN, OPTICS."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.clustering_methods import (
    bench_cluster,
    dbscan,
    kmeans_pp,
    optics_ordering,
    pam,
)


@pytest.fixture()
def blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = np.vstack(
        [
            rng.normal([0, 0], 0.2, (30, 2)),
            rng.normal([4, 4], 0.2, (30, 2)),
        ]
    )
    return x, np.repeat([0, 1], 30)


def _agree(labels: np.ndarray, truth: np.ndarray) -> float:
    best = 0.0
    for perm in ([0, 1], [1, 0]):
        mapped = np.array(perm)[labels]
        best = max(best, float((mapped == truth).mean()))
    return best


def test_kmeans_pp_recovers(blobs):
    x, truth = blobs
    r = kmeans_pp(x, 2, seed=1)
    assert _agree(np.asarray(r["labels"]), truth) > 0.95


def test_pam_recovers(blobs):
    x, truth = blobs
    r = pam(x, 2, seed=1)
    assert _agree(np.asarray(r["labels"]), truth) > 0.95
    assert len(r["medoids"]) == 2


def test_dbscan_two_clusters(blobs):
    x, _ = blobs
    r = dbscan(x, eps=0.5, min_pts=3)
    assert r["n_clusters"] == 2


def test_dbscan_noise_flagged():
    rng = np.random.default_rng(3)
    core = rng.normal([0, 0], 0.15, (25, 2))
    x = np.vstack([core, np.array([[10.0, 10.0], [-10.0, 10.0]])])
    r = dbscan(x, eps=0.5, min_pts=3)
    lab = np.asarray(r["labels"])
    assert lab[-2] == -1 and lab[-1] == -1


def test_optics_ordering_covers_all(blobs):
    x, _ = blobs
    r = optics_ordering(x, eps=1.0, min_pts=3)
    assert len(r["order"]) == len(x)


def test_bench_cluster():
    out = bench_cluster(seed=535)
    assert out["synthetic_blobs_purity_km"] > 0.95
    assert out["synthetic_dbscan_clusters"] == 2.0
