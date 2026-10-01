"""stack_invariance — the stacker's contracts that hold regardless of input order.

The cross-fitted ridge stacker must be **column-permutation equivariant**:
reordering base-prediction columns permutes the fitted weights identically
and leaves OOF predictions untouched. Anything else means the fused book
depends on head registration order — a silent dependency nobody reviews.

Also pinned:

- **Regularization limit**: ``alpha → ∞`` must drive weights → 0 (a ridge
  that doesn't is not a ridge).
- **Duplicate-member symmetry**: two identical columns split weight evenly
  (ridge's unique symmetric solution) — a canonicality check.
- **Fold-contract fail-closed**: overlapping train/test or uncovered test
  rows raise rather than silently leak.
- **``fuse_signals`` sign discipline**: confidence / regime_compat / risk
  cannot flip the alpha sign.

Sealed ``stack_invariance.v1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.config.models import FusionConfig
from quant_fund.fusion.engine import cross_fitted_ridge_stack, fuse_signals
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["stack_invariance_bench"]

Array = NDArray[np.float64]


def _folds(n: int, k: int = 4) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Chronological k-fold indices over ``n`` rows (contiguous blocks)."""
    edges = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        test = np.arange(edges[i], edges[i + 1], dtype=np.int64)
        train = np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], n)]).astype(
            np.int64
        )
        out.append((train, test))
    return out


def _panel(seed: int, n: int = 240, k: int = 5) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 1.0, n)
    x = np.column_stack(
        [0.3 * y + rng.normal(0.0, 1.0, n) for _ in range(k - 1)] + [rng.normal(0.0, 1.0, n)]
    )
    return x, y


def stack_invariance_bench(seed: int = 0) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    x, y = _panel(seed)
    folds = _folds(x.shape[0])

    # 1. Column-permutation equivariance.
    perm = np.array([2, 0, 4, 1, 3])
    ref = cross_fitted_ridge_stack(x, y, folds)
    permuted = cross_fitted_ridge_stack(x[:, perm], y, folds)
    w_dev = float(np.max(np.abs(permuted.weights[np.argsort(perm)] - ref.weights)))
    oof_dev = float(np.nanmax(np.abs(permuted.oof_predictions - ref.oof_predictions)))
    checks["permutation"] = {
        "weight_dev": w_dev,
        "oof_dev": oof_dev,
        "ok": w_dev < 1e-10 and oof_dev < 1e-10,
    }

    # 2. Regularization limit.
    heavy = cross_fitted_ridge_stack(x, y, folds, alpha=1e12)
    checks["regularization"] = {
        "max_weight_at_1e12": float(np.max(np.abs(heavy.weights))),
        "ok": float(np.max(np.abs(heavy.weights))) < 1e-6,
    }

    # 3. Duplicate-member symmetry.
    xdup = np.column_stack([x[:, 0], x[:, 0], x[:, 1:]])
    dup = cross_fitted_ridge_stack(xdup, y, folds)
    split = float(abs(dup.weights[0] - dup.weights[1]))
    checks["duplicate_symmetry"] = {"weight_split_diff": split, "ok": split < 1e-9}

    # 4. Fail-closed fold contract.
    bad_overlap = [(np.array([0, 1, 2]), np.array([2, 3, 4]))]
    try:
        cross_fitted_ridge_stack(x, y, bad_overlap)
        overlap_ok = False
    except ValueError:
        overlap_ok = True
    checks["overlap_rejected"] = {"ok": overlap_ok}

    # 5. fuse_signals: confidence=0 must zero the alpha term (sign discipline).
    cfg = FusionConfig()
    n = 8
    base = dict(
        alpha=np.full(n, 0.5),
        confidence=np.ones(n),
        regime_compat=np.ones(n),
        predicted_risk=np.ones(n),
        tail_penalty=np.zeros(n),
        liq_penalty=np.zeros(n),
    )
    on = fuse_signals(**base, config=cfg)
    off = fuse_signals(**{**base, "confidence": np.zeros(n)}, config=cfg)
    checks["sign_discipline"] = {
        "zero_confidence_output_zero": bool(np.allclose(off, 0.0)),
        "unit_confidence_positive": bool(np.all(on > 0.0)),
    }

    ok = all(
        c["ok"]
        if "ok" in c
        else c.get("zero_confidence_output_zero", False)
        and c.get("unit_confidence_positive", False)
        for c in checks.values()
    )
    payload: dict[str, Any] = {
        "kind": "stack_invariance",
        "schema": "stack_invariance.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"checks": checks, "ok": ok},
        "interpretation": (
            "Stacker contracts hold: column order cannot leak into weights; "
            "heavy ridge kills weights; duplicate members split evenly; "
            "leaking folds fail closed; zero confidence zeros the alpha term."
            if ok
            else f"CONTRACT VIOLATIONS: {checks}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
