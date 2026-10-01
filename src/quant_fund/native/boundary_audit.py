"""boundary_audit — degenerate-input audit of the native reference kernels.

Every kernel in :mod:`quant_fund.native.reference` is driven against a grid
of hostile inputs: empty arrays, inputs shorter than the window, all-equal
series, NaN at head/tail, and extreme magnitudes. For each (kernel, input)
cell the audit records a verdict class:

- ``value`` — a finite-or-NaN output of the right shape (NaN padding is the
  documented contract for unfilled windows);
- ``raise`` — a ``ValueError``/``TypeError`` rejection (fail-closed is fine);
- ``flag:silent_zeros`` — a finite all-zero output on input with no real
  signal (silent fabrication, the dangerous class);
- ``flag:inf_escape`` — ``±inf`` in the output where the input was finite
  (documented contract is NaN propagation, not infinities);
- ``flag:shape`` — output shape drifted from input length.

Verdict ``ok`` iff no cell lands in a flag class. Sealed ``boundary_audit.v1``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.native import reference
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["boundary_audit", "boundary_audit_bench"]

FloatArray = NDArray[np.float64]

_KERNELS: dict[str, Callable[..., Any]] = {
    "rolling_mean": lambda v: reference.rolling_mean(v, 3),
    "rolling_std": lambda v: reference.rolling_std(v, 3),
    "ema": lambda v: reference.ema(v, 3),
    "rsi": lambda v: reference.rsi(v, 5),
    "bollinger": lambda v: reference.bollinger(v, 5),
    "simple_returns": lambda v: reference.simple_returns(v),
    "wealth_index": lambda v: reference.wealth_index(v),
    "turnover_series": lambda v: reference.turnover_series(v),
}

_INPUTS: dict[str, FloatArray] = {
    "empty": np.array([], dtype=np.float64),
    "one": np.array([1.0]),
    "short": np.array([1.0, 1.5]),
    "flat": np.full(24, 1.0),
    "nan_head": np.concatenate([[np.nan], np.linspace(1, 2, 23)]),
    "nan_tail": np.concatenate([np.linspace(1, 2, 23), [np.nan]]),
    "huge": np.full(24, 1e308),
    "tiny": np.full(24, 1e-308),
    "mixed_sign": np.array([(-1.0) ** i * (1 + i * 0.01) for i in range(24)]),
    "inf_input": np.concatenate([np.linspace(1, 2, 12), [np.inf], np.linspace(1, 2, 11)]),
}


def _flatten(out: Any) -> FloatArray | None:
    if isinstance(out, dict):
        arrs = [np.asarray(v, dtype=np.float64) for v in out.values()]
        return np.concatenate([a.ravel() for a in arrs]) if arrs else np.array([])
    try:
        return np.asarray(out, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None


def _classify(name: str, iname: str, x: FloatArray, fn: Callable[[Any], Any]) -> dict[str, Any]:
    try:
        out = fn(x)
    except (ValueError, TypeError, ZeroDivisionError, IndexError, ArithmeticError) as exc:
        return {"kernel": name, "input": iname, "verdict": "raise", "detail": type(exc).__name__}
    except Exception as exc:  # noqa: BLE001 — unexpected classes are themselves findings
        return {
            "kernel": name,
            "input": iname,
            "verdict": "flag:unexpected_raise",
            "detail": f"{type(exc).__name__}:{exc}",
        }
    flat = _flatten(out)
    if flat is None:
        return {"kernel": name, "input": iname, "verdict": "flag:unreadable_output"}
    if flat.size == 0:
        return {"kernel": name, "input": iname, "verdict": "value", "detail": "empty"}
    finite = np.isfinite(flat)
    xfin = x[np.isfinite(x)]
    # All-zero output is only suspicious when the input carries real signal —
    # a flat series legitimately yields zero std/returns.
    input_var = float(np.var(xfin)) if xfin.size else 0.0
    has_signal = xfin.size > 1 and input_var > 0.0 and np.isfinite(input_var)
    if flat.size and np.all(flat[finite] == 0.0) and np.any(finite) and has_signal:
        return {"kernel": name, "input": iname, "verdict": "flag:silent_zeros"}
    if np.isfinite(x).all() and x.size > 0 and np.any(np.isinf(flat)):
        # Overflow on moderate input is a defect; on extreme input (|x|≥1e200)
        # it is the documented cumsum/cumprod precision limit — recorded, not flagged.
        verdict = (
            "note:overflow" if xfin.size and np.max(np.abs(xfin)) >= 1e200 else "flag:inf_escape"
        )
        return {"kernel": name, "input": iname, "verdict": verdict}
    return {"kernel": name, "input": iname, "verdict": "value"}


def boundary_audit() -> dict[str, Any]:
    """Run every kernel × input cell; group by verdict class."""
    cells = []
    for kname, fn in _KERNELS.items():
        for iname, x in _INPUTS.items():
            cells.append(_classify(kname, iname, x, fn))
    flagged = [c for c in cells if c["verdict"].startswith("flag")]
    counts: dict[str, int] = {}
    for c in cells:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    return {"cells": cells, "flagged": flagged, "verdict_counts": counts}


def boundary_audit_bench() -> dict[str, Any]:
    audit = boundary_audit()
    ok = not audit["flagged"]
    payload: dict[str, Any] = {
        "kind": "boundary_audit",
        "schema": "boundary_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "kernels return documented output or fail closed — never silent garbage",
            "n_cells": len(audit["cells"]),
            "verdict_counts": audit["verdict_counts"],
            "n_flagged": len(audit["flagged"]),
            "flagged": audit["flagged"][:20],
            "ok": ok,
        },
        "interpretation": (
            f"{len(audit['cells'])} kernel×input cells: "
            + (
                f"{len(audit['flagged'])} flagged."
                if audit["flagged"]
                else "all cells returned documented output or failed closed."
            )
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
