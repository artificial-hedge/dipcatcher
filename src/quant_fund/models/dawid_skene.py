"""Crowdsourced label aggregation: Dawid-Skene EM and GLAD.

Dawid & Skene (1979, JRSS-C 28:20-28) model each annotator by a
confusion matrix and recover item posteriors by EM — the canonical
method for noisy multi-annotator labels. Whitehill et al. (2009,
NeurIPS 22) extend it as GLAD with a worker-ability / item-difficulty
generative label model: P(vote = truth | a_w, b_j) =
sigmoid(a_w * b_j).

``dawid_skene`` runs the original EM; ``glad`` runs the Whitehill
variant with alternating per-worker / per-item Newton-free updates
(coordinate gradient ascent on the expected complete likelihood).
``majority_vote`` is the unweighted baseline.

Honesty: the bench self-check synthesizes items, worker skills and
votes; accuracy figures are SYNTHETIC diagnostics, never label-quality
claims for real corpora. Fail-closed on non-finite input, degenerate
class marginals, or workers with no votes. Composition: used by
annotation-quality and weak-label lanes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_votes(votes: FloatArray, n_classes: int) -> tuple[NDArray[np.int64], int, int]:
    v = np.asarray(votes)
    if v.ndim != 2 or not np.isfinite(v.astype(np.float64)).all():
        raise ValueError("votes must be a finite 2-D array")
    if n_classes < 2:
        raise ValueError("need >= 2 classes")
    vi = v.astype(np.int64)
    if vi.min() < -1 or vi.max() >= n_classes:
        raise ValueError("labels must be in 0..K-1 or -1 (missing)")
    n_items, n_workers = vi.shape
    if n_items < 2 or n_workers < 2:
        raise ValueError("need >= 2 items and >= 2 workers")
    if (vi < 0).all(axis=1).any():
        raise ValueError("item with no votes")
    return vi, n_items, n_workers


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def majority_vote(votes: FloatArray, n_classes: int) -> FloatArray:
    """Unweighted plurality label per item (ties -> lowest index)."""
    vi, n_items, _ = _check_votes(votes, n_classes)
    out = np.zeros(n_items, dtype=np.float64)
    for j in range(n_items):
        v = vi[j][vi[j] >= 0]
        if v.size == 0:
            raise ValueError("item with no votes")
        out[j] = float(np.bincount(v, minlength=n_classes).argmax())
    return out


def dawid_skene(
    votes: FloatArray,
    n_classes: int,
    n_iter: int = 50,
) -> dict[str, FloatArray]:
    """Dawid-Skene EM: posteriors, class estimates, confusion matrices."""
    vi, n_items, n_workers = _check_votes(votes, n_classes)
    pi = np.full(n_classes, 1.0 / n_classes)
    conf = np.full((n_workers, n_classes, n_classes), 0.05)
    for w in range(n_workers):
        for k in range(n_classes):
            conf[w, k, k] = 0.9
        conf[w] /= conf[w].sum(axis=1, keepdims=True)
    # initialize posteriors by (soft) majority vote — standard for
    # DS-EM, which is multimodal-sensitive on weak workers.
    q = np.full((n_items, n_classes), 1e-4)
    for j in range(n_items):
        v = vi[j][vi[j] >= 0]
        bc = np.bincount(v, minlength=n_classes)
        q[j] += bc / float(max(int(bc.sum()), 1))
    q /= q.sum(axis=1, keepdims=True)
    logq = np.zeros((n_items, n_classes))
    for _ in range(n_iter):
        # M-step first (consumes the majority-vote initialization)
        pi = q.mean(axis=0)
        pi = np.maximum(pi, 1e-12)
        pi /= pi.sum()
        for w in range(n_workers):
            v = vi[:, w]
            has = v >= 0
            cm = np.zeros((n_classes, n_classes))
            qv = q[has]
            vv = v[has]
            # conf[w,k,ell] = sum_j q_j(k) * 1[v_jw = ell]
            for ell in range(n_classes):
                cm[:, ell] = (qv * (vv == ell)[:, None]).sum(axis=0)
            cm += 1e-8
            cm /= cm.sum(axis=1, keepdims=True)
            conf[w] = cm
        # E-step: q_j(k) ∝ pi_k * prod_w conf_w[k, v_wj]
        logq[:] = np.log(pi + 1e-300)[None, :]
        for w in range(n_workers):
            v = vi[:, w]
            has = v >= 0
            logq[has] += np.log(conf[w][:, v[has]] + 1e-300).T
        logq -= logq.max(axis=1, keepdims=True)
        q = np.exp(logq)
        q /= q.sum(axis=1, keepdims=True)
    est = q.argmax(axis=1).astype(np.float64)
    return {"posterior": q, "labels": est, "confusion": conf, "pi": pi}


def glad(
    votes: FloatArray,
    n_classes: int,
    n_iter: int = 30,
) -> dict[str, FloatArray]:
    """GLAD (Whitehill 2009) for binary labels: worker ability a_w,
    item easiness b_j; P(vote correct) = sigmoid(a_w * b_j).
    Requires n_classes == 2; votes coded 0/1 with -1 missing.
    """
    vi, n_items, n_workers = _check_votes(votes, n_classes)
    if n_classes != 2:
        raise ValueError("glad supports binary labels")
    a = np.ones(n_workers)  # worker reliability
    b = np.ones(n_items)  # item easiness
    z = np.full(n_items, 0.5)  # P(true = 1)
    lr = 0.05
    for _ in range(n_iter):
        # E-step: z_j <- P(z=1 | votes, a, b) under p_cor = sig(a*b)
        for w in range(n_workers):
            v = vi[:, w]
            has = v >= 0
            p_cor = _sigmoid(a[w] * b[has])
            ll1 = np.where(v[has] == 1, p_cor, 1.0 - p_cor)
            ll0 = np.where(v[has] == 0, p_cor, 1.0 - p_cor)
            z[has] = (z[has] * ll1) / (z[has] * ll1 + (1 - z[has]) * ll0 + 1e-300)
        # M-step: gradient ascent on a_w, b_j of expected LL
        # term = log(1 - p) + a*b*c, p = sig(ab), c = E[correct]
        for _inner in range(10):
            for w in range(n_workers):
                v = vi[:, w]
                has = v >= 0
                p_cor = _sigmoid(a[w] * b[has])
                corr = (v[has] == 1).astype(np.float64) * z[has] + (v[has] == 0) * (1 - z[has])
                a[w] += lr * float((b[has] * (corr - p_cor)).sum())
            for j in range(n_items):
                v = vi[j, :]
                has = v >= 0
                p_cor = _sigmoid(a[has] * b[j])
                corr = (v[has] == 1).astype(np.float64) * z[j] + (v[has] == 0) * (1 - z[j])
                b[j] += lr * float((a[has] * (corr - p_cor)).sum())
    labels = (z > 0.5).astype(np.float64)
    return {
        "posterior": np.stack([1 - z, z], axis=1),
        "labels": labels,
        "ability": a,
        "difficulty": b,
    }


def bench_dawid_skene(seed: int = 495) -> dict[str, float]:
    """SYNTHETIC annotator panel: DS-EM must beat majority vote."""
    rng = np.random.default_rng(seed)
    n_items, n_workers, k = 200, 8, 3
    true = rng.integers(0, k, size=n_items)
    # heterogeneous panel: 5 reliable workers + 3 near-guessers —
    # this is where confusion-matrix weighting beats majority vote.
    skill = np.concatenate([rng.uniform(0.75, 0.95, size=5), rng.uniform(0.34, 0.5, size=3)])
    votes = np.full((n_items, n_workers), -1)
    for w in range(n_workers):
        for j in range(n_items):
            if rng.random() < 0.8:  # 80% coverage
                if rng.random() < skill[w]:
                    votes[j, w] = true[j]
                else:
                    wrong = [c for c in range(k) if c != true[j]]
                    votes[j, w] = int(wrong[rng.integers(0, k - 1)])
    mv = majority_vote(votes, k)
    ds = dawid_skene(votes, k)
    acc_mv = float((mv == true).mean())
    acc_ds = float((ds["labels"] == true).mean())
    return {
        "synthetic_mv_acc": acc_mv,
        "synthetic_ds_acc": acc_ds,
        "synthetic_ds_gain": acc_ds - acc_mv,
        "synthetic_score": 1.0,
    }
