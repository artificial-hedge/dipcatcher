"""Classical clustering canon: k-means++, PAM
(k-medoids), DBSCAN, and an OPTICS reachability
ordering — the density/medoid complement to the
spectral and centroid methods already in canon.

All functions are deterministic given a seed and
return plain dicts; `bench_cluster` runs a synthetic
self-check on well-separated blobs plus a two-moons
density case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quant_fund._typing import FloatArray

__all__ = [
    "kmeans_pp",
    "pam",
    "dbscan",
    "optics_ordering",
    "bench_cluster",
]


def kmeans_pp(x: FloatArray, k: int, seed: int = 0, it: int = 100) -> dict[str, object]:
    """k-means++ (Arthur-Vassilvitskii 2007): D²
    seeding then Lloyd iterations."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    cent = np.empty((k, x.shape[1]))
    cent[0] = x[rng.integers(n)]
    d2 = np.sum((x - cent[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = d2 / d2.sum()
        cent[i] = x[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((x - cent[i]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(it):
        dist = np.linalg.norm(x[:, None, :] - cent[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                cent[c] = x[m].mean(axis=0)
    return {"labels": labels, "centers": cent}


def _pam_cost(x: FloatArray, med_idx: list[int]) -> float:
    dist = np.linalg.norm(x[:, None, :] - x[med_idx][None], axis=2)
    return float(dist.min(axis=1).sum())


def pam(x: FloatArray, k: int, seed: int = 0, it: int = 50) -> dict[str, object]:
    """PAM (Kaufman-Rousseeuw): BUILD greedy medoid
    init then SWAP improvement passes."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    # BUILD: first medoid minimizes total distance
    first = int(np.argmin(d.sum(axis=1)))
    med: list[int] = [first]
    while len(med) < k:
        best_j, best_gain = -1, np.inf
        for j in range(n):
            if j in med:
                continue
            cand = med + [j]
            cost = float(np.minimum.reduce(d[:, cand]).sum())
            if cost < best_gain:
                best_gain, best_j = cost, j
        med.append(best_j)
    cost = _pam_cost(x, med)
    for _ in range(it):
        improved = False
        nonmed = [j for j in range(n) if j not in med]
        for mi in range(len(med)):
            for j in nonmed:
                cand = med.copy()
                cand[mi] = j
                c = _pam_cost(x, cand)
                if c < cost - 1e-12:
                    med, cost = cand, c
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    labels = np.argmin(d[:, med], axis=1)
    return {"labels": labels, "medoids": np.array(med)}


def dbscan(x: FloatArray, eps: float, min_pts: int) -> dict[str, object]:
    """DBSCAN (Ester et al. 1996): core/border/noise
    via eps-neighborhood expansion."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    nbr = [np.flatnonzero(d[i] <= eps) for i in range(n)]
    labels = np.full(n, -1)  # -1 = noise
    cluster = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        if len(nbr[i]) < min_pts:
            continue
        labels[i] = cluster
        queue: list[int] = [int(j) for j in nbr[i]]
        k = 0
        while k < len(queue):
            j = queue[k]
            k += 1
            if labels[j] == -1:
                labels[j] = cluster  # border joins
            if labels[j] != -1 and labels[j] != cluster:
                continue
            labels[j] = cluster
            if len(nbr[j]) >= min_pts:
                for q in nbr[j]:
                    if labels[q] != cluster:
                        queue.append(int(q))
        cluster += 1
    return {"labels": labels, "n_clusters": cluster}


def optics_ordering(x: FloatArray, eps: float, min_pts: int) -> dict[str, object]:
    """OPTICS (Ankerst et al. 1999): reachability
    ordering of the dataset via core-distance walk."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    processed = np.zeros(n, dtype=bool)
    order: list[int] = []
    reach = np.full(n, np.inf)
    for i in range(n):
        if processed[i]:
            continue
        order.append(i)
        processed[i] = True
        nbr = np.flatnonzero(d[i] <= eps)
        if len(nbr) < min_pts:
            continue
        core = float(np.partition(d[i], min_pts - 1)[min_pts - 1])
        seeds = np.argsort(np.maximum(core, d[i, nbr]) + 1e-12 * nbr).tolist()
        seed_list = [nbr[s] for s in seeds]
        while seed_list:
            j = seed_list.pop(0)
            if processed[j]:
                continue
            r_j = max(core, float(d[i, j]))
            if reach[j] > r_j:
                reach[j] = r_j
            processed[j] = True
            order.append(j)
            nbr_j = np.flatnonzero(d[j] <= eps)
            if len(nbr_j) < min_pts:
                continue
            core_j = float(np.partition(d[j], min_pts - 1)[min_pts - 1])
            cand = [q for q in nbr_j if not processed[q]]
            keys = [max(core_j, float(d[j, q])) for q in cand]
            new_seeds = [cand[s] for s in np.argsort(keys)]
            seed_list.extend(new_seeds)
    return {"order": np.array(order), "reachability": reach}


def bench_cluster(seed: int = 535) -> dict[str, float]:
    """SYNTHETIC: well-separated blobs (kmeans/PAM must
    recover near-perfectly) + moons for DBSCAN noise
    handling."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}

    def purity(lab: FloatArray, truth: FloatArray) -> float:
        lab = np.asarray(lab)
        truth = np.asarray(truth)
        best = 0.0
        for c in np.unique(lab):
            m = lab == c
            if not m.any():
                continue
            frac = np.bincount(truth[m]) / m.sum()
            best += m.sum() * frac.max()
        return float(best / len(lab))

    blobs = np.vstack(
        [
            rng.normal([0, 0], 0.3, (60, 2)),
            rng.normal([5, 5], 0.3, (60, 2)),
            rng.normal([0, 6], 0.3, (60, 2)),
        ]
    )
    truth = np.repeat([0, 1, 2], 60)
    km = kmeans_pp(blobs, 3, seed)
    pm = pam(blobs, 3, seed)
    p_km = purity(km["labels"], truth)
    p_pm = purity(pm["labels"], truth)
    out["synthetic_blobs_purity_km"] = p_km
    out["synthetic_blobs_purity_pm"] = p_pm
    if p_km < 0.98 or p_pm < 0.98:
        raise ValueError(f"blob purity off: {p_km}/{p_pm}")
    # two moons + noise for DBSCAN
    t1 = np.linspace(0, np.pi, 40)
    moon1 = np.c_[np.cos(t1), np.sin(t1)]
    t2 = np.linspace(0, np.pi, 40)
    moon2 = np.c_[np.cos(t2) + 1.0, -np.sin(t2) - 0.4]
    moons = np.vstack([moon1, moon2]) + rng.normal(scale=0.04, size=(80, 2))
    moons = np.vstack([moons, rng.uniform(-1.5, 2.5, (8, 2))])
    db = dbscan(moons, eps=0.22, min_pts=4)
    lab = np.asarray(db["labels"], dtype=np.int64)
    n_noise = int((lab == -1).sum())
    out["synthetic_dbscan_noise"] = float(n_noise)
    out["synthetic_dbscan_clusters"] = float(len(np.unique(lab[lab >= 0])))
    if n_noise < 2 or int(np.asarray(db["n_clusters"])) != 2:
        raise ValueError(f"dbscan off: noise={n_noise} k={db['n_clusters']}")
    return out
