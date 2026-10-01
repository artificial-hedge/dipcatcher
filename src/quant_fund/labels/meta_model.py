"""meta_model — AFML meta-labeling as a fitted, honestly-evaluated model.

``barriers.meta_labels`` builds the binary correctness label; this module
fits the secondary model that *predicts* it and sizes/vetoes the primary
bet. Two failure modes the naive version has, both closed here:

- Overlapping label windows leak the test path into training features.
  ``purged_kfold_indices`` drops every train event whose
  ``[t_start, t_end]`` interval intersects the fold's test span, plus an
  embargo tail after it (AFML 7.x). This is stricter than a time gap:
  an event *starting before* the test fold but resolving *inside* it is
  still purged.
- Events are not exchangeable — concurrent labels share path segments.
  ``average_uniqueness`` weights each event's contribution to the fit so
  heavily-overlapped windows count less.

``meta_model_eval`` runs the end-to-end bench: triple-barrier labels on a
price path -> meta-labels -> standardize features on train folds only ->
weighted logistic via IRLS -> out-of-fold probabilities -> Brier, AUC,
a calibration table, and the gated-vs-primary comparison (accuracy of the
primary side vs accuracy conditioned on p > gate). All synthetic demos
are labeled SYNTHETIC; no Sharpe/P&L anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from quant_fund.labels.barriers import (
    average_uniqueness,
    meta_labels,
    triple_barrier,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = npt.NDArray[np.float64]
IdxArray = npt.NDArray[np.intp]

MIN_FOLDS = 2
DEFAULT_EMBARGO = 0.01
DEFAULT_L2 = 1.0
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-8
ETA_CLIP = 30.0


def purged_kfold_indices(
    t_start: Array,
    t_end: Array,
    n_folds: int,
    embargo_frac: float = DEFAULT_EMBARGO,
) -> list[tuple[IdxArray, IdxArray]]:
    """Sequential k-fold split with AFML purging + embargo.

    Folds partition the *event order* into contiguous test blocks (events
    must be time-ordered). For each fold, a train event survives only if
    its label interval ``[t_start, t_end]`` ends before the test interval
    begins or starts after the test interval ends plus the embargo span
    (``embargo_frac * n_events`` measured in event-index units).
    """
    s = np.asarray(t_start, dtype=float).reshape(-1)
    e = np.asarray(t_end, dtype=float).reshape(-1)
    n = s.size
    if e.size != n or n == 0:
        raise ValueError("t_start and t_end must be non-empty and equal length")
    if not np.isfinite(s).all() or not np.isfinite(e).all():
        raise ValueError("t_start and t_end must be finite")
    if np.any(e < s):
        raise ValueError("t_end must be >= t_start elementwise")
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < MIN_FOLDS:
        raise ValueError(f"n_folds must be an integer >= {MIN_FOLDS}")
    if n_folds > n:
        raise ValueError("n_folds exceeds the number of events")
    if not np.isfinite(embargo_frac) or embargo_frac < 0.0 or embargo_frac >= 1.0:
        raise ValueError("embargo_frac must be finite in [0, 1)")

    embargo = embargo_frac * float(n)
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    folds: list[tuple[IdxArray, IdxArray]] = []
    for f in range(n_folds):
        lo, hi = int(bounds[f]), int(bounds[f + 1])
        test = np.arange(lo, hi, dtype=np.intp)
        test_start = s[lo]
        test_end = e[hi - 1]
        keep = (e < test_start) | (s > test_end + embargo)
        keep[lo:hi] = False
        train = np.flatnonzero(keep).astype(np.intp)
        folds.append((train, test))
    return folds


def _design(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite 2-D feature matrix")
    if x.shape[0] == 0:
        raise ValueError("x must have at least one row")
    return np.column_stack([np.ones(x.shape[0]), x])


def fit_logistic(
    x: Array,
    y: Array,
    w: Array | None = None,
    l2: float = DEFAULT_L2,
) -> Array:
    """Weighted L2-penalized logistic regression via IRLS.

    Deterministic Newton updates; no intercept penalty. Raises on
    degenerate inputs rather than emitting a silently-flat model.
    """
    d = _design(x)
    t = np.asarray(y, dtype=float).reshape(-1)
    if t.size != d.shape[0]:
        raise ValueError("y must match x rows")
    if not np.isfinite(t).all() or not np.all(np.isin(np.unique(t), (0.0, 1.0))):
        raise ValueError("y must be finite binary labels in {0, 1}")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be non-negative and finite")
    sw = np.ones(d.shape[0]) if w is None else np.asarray(w, dtype=float).reshape(-1)
    if sw.size != d.shape[0] or not np.isfinite(sw).all() or np.any(sw <= 0.0):
        raise ValueError("w must be finite and positive with matching length")

    beta = np.zeros(d.shape[1])
    reg = l2 * np.eye(d.shape[1])
    reg[0, 0] = 0.0
    for _ in range(IRLS_MAX_ITER):
        eta = np.clip(d @ beta, -ETA_CLIP, ETA_CLIP)
        p = 1.0 / (1.0 + np.exp(-eta))
        r = np.clip(p * (1.0 - p), 1e-9, None) * sw
        h = (d * r[:, None]).T @ d + reg
        g = d.T @ (sw * (p - t)) + reg @ beta
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError as exc:
            raise ValueError("IRLS Hessian singular — features collinear") from exc
        beta = beta - step
        if float(np.abs(step).max()) < IRLS_TOL:
            break
    else:
        raise ValueError("IRLS did not converge — check feature scaling")
    return beta


def logistic_predict(x: Array, beta: Array) -> Array:
    d = _design(x)
    b = np.asarray(beta, dtype=float).reshape(-1)
    if b.size != d.shape[1] or not np.isfinite(b).all():
        raise ValueError("beta must be finite with p+1 entries")
    eta = np.clip(d @ b, -ETA_CLIP, ETA_CLIP)
    return np.asarray(1.0 / (1.0 + np.exp(-eta)), dtype=np.float64)


def _auc(scores: Array, y: Array) -> float:
    pos = scores[y > 0.5]
    neg = scores[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(scores.size)
    ranks[order] = np.arange(1, scores.size + 1)
    # midranks for ties
    vals = scores[order]
    i = 0
    while i < vals.size:
        j = i
        while j + 1 < vals.size and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    u = float(ranks[y > 0.5].sum() - pos.size * (pos.size + 1) / 2.0)
    return u / (pos.size * neg.size)


def _calibration_table(p: Array, y: Array, n_bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if int(m.sum()) == 0:
            continue
        rows.append(
            {
                "bin_lo": float(edges[i]),
                "bin_hi": float(edges[i + 1]),
                "n": float(m.sum()),
                "mean_pred": float(p[m].mean()),
                "empirical": float(y[m].mean()),
            }
        )
    return rows


def meta_model_eval(
    close: Array,
    events: IdxArray,
    sides: Array,
    features: Array,
    *,
    pt: float = 0.01,
    sl: float = 0.01,
    horizon: int = 20,
    vol: Array | None = None,
    n_folds: int = 5,
    embargo_frac: float = DEFAULT_EMBARGO,
    l2: float = DEFAULT_L2,
    uniqueness_weighted: bool = True,
    gate: float = 0.5,
) -> dict[str, Any]:
    """End-to-end meta-label bench on one price series.

    ``sides``: the primary model's position sign per event (event-indexed,
    not bar-indexed). ``features``: (n_events, p) matrix aligned with
    ``events``. Returns the full OOF evaluation payload.
    """
    c = np.asarray(close, dtype=float).reshape(-1)
    if c.size < 3 or not np.isfinite(c).all():
        raise ValueError("close must be finite with >= 3 bars")
    ev = np.asarray(events, dtype=np.intp).reshape(-1)
    s = np.asarray(sides, dtype=float).reshape(-1)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] != ev.size or not np.isfinite(x).all():
        raise ValueError("features must be a finite (n_events, p) matrix")
    if s.size != ev.size:
        raise ValueError("sides must be one sign per event")
    if not np.all(np.isin(np.unique(s), (-1.0, 0.0, 1.0))):
        raise ValueError("sides must take values in {-1, 0, +1}")
    if not np.isfinite(gate) or not 0.0 < gate < 1.0:
        raise ValueError("gate must be in (0, 1)")

    tb = triple_barrier(c, ev, pt, sl, horizon, vol)
    labels = tb["label"]
    finite = np.isfinite(labels)
    n_unobservable = int((~finite).sum())

    ev_f = ev[finite]
    s_f = s[finite]
    x_f = x[finite]
    y = meta_labels(s_f, labels[finite])
    t_start = ev_f.astype(float)
    t_end = tb["t_touch"][finite]
    # barriers uses inclusive-start/exclusive-end index spans
    uniq = np.clip(average_uniqueness(t_start.astype(np.intp), t_end.astype(np.intp)), 1e-3, None)

    folds = purged_kfold_indices(t_start, t_end, n_folds, embargo_frac)
    p_oof = np.full(ev_f.size, np.nan)
    fold_rows = []
    for k, (train, test) in enumerate(folds):
        if train.size < x_f.shape[1] + 2:
            raise ValueError(
                f"fold {k}: purging left {train.size} train events for "
                f"{x_f.shape[1]} features — too few"
            )
        mu = x_f[train].mean(axis=0)
        sd = x_f[train].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        w = uniq[train] if uniqueness_weighted else None
        beta = fit_logistic((x_f[train] - mu) / sd, y[train], w, l2)
        p_oof[test] = logistic_predict((x_f[test] - mu) / sd, beta)
        fold_rows.append(
            {"fold": float(k), "n_train": float(train.size), "n_test": float(test.size)}
        )

    n0 = int((y <= 0.5).sum())
    n1 = int((y > 0.5).sum())
    base_rate = float(y.mean())
    brier = float(np.mean((p_oof - y) ** 2))
    brier_null = float(np.mean((base_rate - y) ** 2))
    auc = _auc(p_oof, y)
    active = s_f != 0.0
    primary_acc = float(y[active].mean()) if int(active.sum()) else float("nan")
    taken = p_oof > gate
    gated_acc = float(y[taken].mean()) if int(taken.sum()) else float("nan")
    return {
        "n_events": int(ev.size),
        "n_unobservable": n_unobservable,
        "n_positive": n1,
        "n_negative": n0,
        "base_rate": base_rate,
        "oof": {
            "brier": brier,
            "brier_null": brier_null,
            "brier_skill": 1.0 - brier / brier_null if brier_null > 0 else float("nan"),
            "auc": auc,
            "calibration": _calibration_table(p_oof, y),
        },
        "gating": {
            "gate": gate,
            "share_taken": float(taken.mean()),
            "primary_accuracy": primary_acc,
            "gated_accuracy": gated_acc,
        },
        "folds": fold_rows,
    }


def _momentum_primary(close: Array, lookback: int = 5) -> tuple[IdxArray, Array, Array]:
    """Weak primary signal: side = sign of trailing momentum; features are
    [trailing return, rolling vol, |momentum|]."""
    c = np.asarray(close, dtype=float).reshape(-1)
    ret = np.diff(c) / c[:-1]
    ev = np.arange(lookback, c.size - 1, dtype=np.intp)
    mom = np.array([float(ret[i - lookback : i].sum()) for i in ev])
    roll = np.array([float(ret[max(0, i - lookback) : i].std()) for i in ev])
    sides = np.where(np.abs(mom) < 1e-12, 0.0, np.sign(mom))
    feats = np.column_stack([mom, roll, np.abs(mom)])
    return ev, sides, feats


def meta_model_demo(seed: int = 0, n_bars: int = 4000) -> dict[str, Any]:
    """Synthetic bench: an AR(1)-momentum price path where a momentum
    primary has pockets of edge — the meta-model's job is to find them."""
    rng = np.random.default_rng(seed)
    phi = np.where(rng.random(n_bars) < 0.5, 0.6, -0.1)
    eps = rng.normal(0.0, 0.01, n_bars)
    r = np.zeros(n_bars)
    for t in range(1, n_bars):
        r[t] = phi[t] * r[t - 1] + eps[t]
    close = 100.0 * np.exp(np.cumsum(r))
    ev, sides, feats = _momentum_primary(close)
    res = meta_model_eval(close, ev, sides, feats, pt=0.008, sl=0.008, horizon=15)
    payload: dict[str, Any] = {
        "kind": "meta_model",
        "schema": "meta_model.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "seed": seed,
            "n_bars": n_bars,
            "generative_process": "AR(1) returns with regime-mixed phi",
            "primary": "sign of 5-bar trailing momentum",
            "result": res,
        },
        "interpretation": {
            "brier_skill_positive": "meta-model beats the base-rate null OOF",
            "gated_beats_primary": "p>gate subset is more accurate than the raw primary",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "DEFAULT_EMBARGO",
    "DEFAULT_L2",
    "MIN_FOLDS",
    "fit_logistic",
    "logistic_predict",
    "meta_model_demo",
    "meta_model_eval",
    "purged_kfold_indices",
]
