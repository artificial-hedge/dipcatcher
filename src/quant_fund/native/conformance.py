"""conformance — does the native backend honor the float contract?

``native/__init__.py`` dispatches each kernel to the optional ``quant_core``
Rust extension; ``reference.py`` is the NumPy oracle. The module docstring
declares two contract classes:

- **bit-exact** (NaNs included): ``rolling_mean``, ``rolling_std``,
  ``simple_returns``, ``wealth_index``, ``hash_bytes``, ``hash_many``
- **within RTOL/ATOL**: ``ema``, ``rsi``, ``bollinger``, ``turnover``,
  ``turnover_series``, ``book_features``

``run_conformance`` generates seeded edge+random inputs per kernel, runs
both paths, and reports per-kernel verdicts:

- ``bit_exact_ok`` / ``within_tol`` — contract satisfied on the live backend
- ``mismatch`` — a real divergence (with the worst case recorded)
- ``reference_only`` — the Python backend is active, so conformance is
  self-vs-self plus *structural invariants* (nonnegative std, band
  ordering, NaN placement) — the harness still verifies those so the
  reference path can't silently drift either.

The report seals as a ``native_conformance.v1`` SYNTHETIC receipt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from quant_fund.native import BACKEND
from quant_fund.native import reference as ref
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = npt.NDArray[np.float64]

BIT_EXACT = frozenset(
    {"rolling_mean", "rolling_std", "simple_returns", "wealth_index", "hash_bytes", "hash_many"}
)
TOLERANT = frozenset({"ema", "rsi", "bollinger", "turnover", "turnover_series", "book_features"})


def _cases(seed: int) -> dict[str, list[tuple[Any, ...]]]:
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 5.0, 300)
    # reference kernels fail closed on non-finite *inputs*; NaN coverage
    # lives in output positions (warmup) and in _compare's NaN-placement
    # contract instead
    noisy = np.concatenate([rng.normal(100.0, 5.0, 150), rng.normal(99.0, 5.0, 150)])
    flat = np.full(200, 42.5)
    ramp = np.linspace(50.0, 150.0, 250)
    series_sets = [base, noisy, flat, ramp, rng.normal(0.0, 1.0, 64)]
    windows = [1, 5, 20]
    cases: dict[str, list[tuple[Any, ...]]] = {
        "rolling_mean": [(s, w) for s in series_sets for w in windows],
        "rolling_std": [(s, w) for s in series_sets for w in windows],
        "ema": [(s, w) for s in series_sets for w in windows],
        "rsi": [(s, w) for s in series_sets for w in (5, 14)],
        "bollinger": [(s, 20, 2.0) for s in series_sets],
        "simple_returns": [(s,) for s in series_sets],
        "wealth_index": [(rng.normal(0.001, 0.02, 200),) for _ in range(4)],
        "turnover": [(rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))) for _ in range(4)],
        "turnover_series": [(np.abs(rng.normal(0.0, 1.0, (5, 40))),) for _ in range(3)],
        "hash_bytes": [(f"chunk-{i}".encode(),) for i in range(4)],
        "hash_many": [([f"c{i}".encode() for i in range(k)],) for k in (0, 1, 5)],
        "book_features": [
            (
                100.0 - np.cumsum(np.abs(rng.normal(0.0, 0.5, (8, 5))), axis=1),
                rng.integers(1, 50, (8, 5)).astype(float),
                100.0 + np.cumsum(np.abs(rng.normal(0.0, 0.5, (8, 5))), axis=1),
                rng.integers(1, 50, (8, 5)).astype(float),
            )
            for _ in range(2)
        ],
    }
    return cases


def _call(name: str, fn: Any, args: tuple[Any, ...]) -> Any:
    return fn(*args)


def _native_fn(name: str) -> Any:
    import quant_fund.native as n

    return getattr(n, name)


def _ref_fn(name: str) -> Any:
    return getattr(ref, name)


def _compare(name: str, a: Any, b: Any) -> tuple[bool, float]:
    """True if outputs honor the kernel's contract; returns (ok, max_abs_err)."""
    exact = name in BIT_EXACT
    if isinstance(a, str) and isinstance(b, str):
        return (a == b, 0.0)
    if isinstance(a, list) and all(isinstance(x, str) for x in a):
        return (a == b, 0.0)
    if isinstance(a, dict):
        ok, worst = True, 0.0
        for k in a:
            sub_ok, sub_err = _compare(name, a[k], b[k])
            ok = ok and sub_ok
            worst = max(worst, sub_err)
        return ok, worst
    if isinstance(a, (int, float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True, 0.0
        return (
            bool(a == b) if exact else bool(np.isclose(a, b, rtol=ref.RTOL, atol=ref.ATOL)),
            float(abs(a - b)),
        )
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        return False, float("inf")
    both_nan = np.isnan(arr_a) & np.isnan(arr_b)
    if not np.array_equal(np.isnan(arr_a), np.isnan(arr_b)):
        return False, float("inf")  # NaN placement drift is a contract breach even in tol mode
    fa, fb = arr_a[~both_nan], arr_b[~both_nan]
    if fa.size == 0:
        return True, 0.0
    err = float(np.max(np.abs(fa - fb)))
    if exact:
        return bool(np.array_equal(fa, fb)), err
    return bool(np.allclose(fa, fb, rtol=ref.RTOL, atol=ref.ATOL)), err


def _invariants(name: str, out: Any) -> list[str]:
    """Structural properties the *reference* itself must keep."""
    errs: list[str] = []
    if name == "rolling_std":
        a = np.asarray(out, dtype=float)
        if np.any(a[np.isfinite(a)] < 0):
            errs.append("rolling_std negative")
    if name == "bollinger":
        lo, mid, hi = (np.asarray(out[k], dtype=float) for k in ("lower", "mid", "upper"))
        fin = np.isfinite(lo) & np.isfinite(hi)
        if np.any(lo[fin] > hi[fin]):
            errs.append("bollinger lower > upper")
        if np.any((mid[fin] < lo[fin] - ref.ATOL) | (mid[fin] > hi[fin] + ref.ATOL)):
            errs.append("bollinger mid outside band")
    if name == "turnover" or name == "turnover_series":
        a = np.asarray(out, dtype=float)
        if np.any(a[np.isfinite(a)] < -ref.ATOL):
            errs.append("negative turnover")
    if name == "hash_bytes" and (not isinstance(out, str) or len(out) != 64):
        errs.append("hash_bytes not 64-hex")
    return errs


def run_conformance(seed: int = 0) -> dict[str, Any]:
    """Full sweep; returns a sealable report."""
    cases = _cases(seed)
    per_kernel: dict[str, Any] = {}
    for name in sorted(BIT_EXACT | TOLERANT):
        ref_fn, nat_fn = _ref_fn(name), _native_fn(name)
        n_run = 0
        worst = {"err": 0.0, "case": -1}
        invariant_errs: list[str] = []
        mismatch = False
        for i, args in enumerate(cases.get(name, [])):
            ref_out = _call(name, ref_fn, args)
            nat_out = _call(name, nat_fn, args)
            invariant_errs.extend(_invariants(name, ref_out))
            ok, err = _compare(name, ref_out, nat_out)
            n_run += 1
            if err > float(worst["err"]):
                worst = {"err": err, "case": i}
            if not ok:
                mismatch = True
        if BACKEND == "python":
            verdict = "reference_only"
        elif mismatch:
            verdict = "mismatch"
        else:
            verdict = "bit_exact_ok" if name in BIT_EXACT else "within_tol"
        per_kernel[name] = {
            "contract": "bit_exact" if name in BIT_EXACT else "tolerant",
            "n_cases": n_run,
            "verdict": verdict,
            "max_abs_err": float(worst["err"]),
            "worst_case": int(worst["case"]),
            "invariant_errors": sorted(set(invariant_errs)),
        }
    n_bad = sum(
        1 for k in per_kernel.values() if k["verdict"] == "mismatch" or k["invariant_errors"]
    )
    payload: dict[str, Any] = {
        "kind": "native_conformance",
        "schema": "native_conformance.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "seed": seed,
            "backend": BACKEND,
            "n_kernels": len(per_kernel),
            "per_kernel": per_kernel,
            "n_failing_kernels": n_bad,
            "ok": n_bad == 0,
        },
        "interpretation": {
            "reference_only": "python backend active — contract verified structurally, native parity unexercised",
            "mismatch": "native kernel diverged from the NumPy oracle beyond its declared contract",
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["BIT_EXACT", "TOLERANT", "run_conformance"]
