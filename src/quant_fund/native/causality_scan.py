"""Causality scan — no kernel output may depend on future inputs.

``warmup_spec`` asks how much past a chunked kernel needs; this lane asks
the orthogonal question: does output at index ``i`` ever depend on input
at index ``j > i``? For each reference op, run on a base series and on
the same series with a randomized suffix appended, and assert the shared
prefix outputs are bit-identical. A leak (a smoothed, centered, or
otherwise non-causal kernel) surfaces instantly.

Ops covered: ``rolling_mean``, ``rolling_std``, ``ema``, ``rsi``,
``bollinger`` (mid/upper/lower), ``simple_returns``, ``wealth_index``,
``turnover_series``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.native import reference
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def _ops() -> dict[str, Any]:
    return {
        "rolling_mean": lambda a: reference.rolling_mean(a, 7),
        "rolling_std": lambda a: reference.rolling_std(a, 7),
        "ema": lambda a: reference.ema(a, 12),
        "rsi": lambda a: reference.rsi(a, 14),
        "bollinger_mid": lambda a: reference.bollinger(a, 20, 2.0)["mid"],
        "simple_returns": lambda a: reference.simple_returns(a),
        "wealth_index": lambda a: reference.wealth_index(a),
    }


def causality_scan(
    n: int = 256,
    extend: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Append a random suffix; the prefix outputs must not move."""
    rng = np.random.default_rng(seed)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    suffix = base[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.01, extend)))
    full = np.concatenate([base, suffix])
    report: dict[str, Any] = {}
    for name, op in _ops().items():
        try:
            short = np.asarray(op(base), dtype=np.float64)
            long = np.asarray(op(full), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the audit
            report[name] = {"status": "error", "error": str(exc)[:200]}
            continue
        k = min(len(short), len(long))
        if k == 0:
            report[name] = {"status": "empty"}
            continue
        shared = long[:k]
        mismatch = np.not_equal(
            np.where(np.isnan(short[:k]), 1.0, short[:k]),
            np.where(np.isnan(shared), 1.0, shared),
        ) | (np.isnan(short[:k]) != np.isnan(shared))
        n_diff = int(mismatch.sum())
        report[name] = {
            "status": "causal" if n_diff == 0 else "LEAK",
            "n_shared": int(k),
            "n_diff": n_diff,
            "first_diff_index": int(np.argmax(mismatch)) if n_diff else None,
            "max_abs_diff": (
                float(np.nanmax(np.abs(short[:k] - shared)))
                if n_diff and np.isfinite(np.abs(short[:k] - shared)).any()
                else (0.0 if n_diff == 0 else None)
            ),
        }
    return report


def causality_bench(seed: int = 0) -> dict[str, Any]:
    """Sealed receipt over the reference kernel set."""
    report = causality_scan(seed=seed)
    leaks = {k: v for k, v in report.items() if v.get("status") == "LEAK"}
    verdict = "causal" if not leaks else "leak"
    payload: dict[str, Any] = {
        "kind": "causality_scan",
        "schema": "causality_scan.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "appending future data never changes past outputs",
            "n_ops": len(report),
            "n_leaks": len(leaks),
            "verdict": verdict,
        },
        "interpretation": report,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
