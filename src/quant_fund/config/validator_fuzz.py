"""Validator fuzz — hostile values through every config field.

The config models rely on ``model_validator``/``field_validator`` code
rather than numeric bounds; this lane enumerates every ``float``/``int``
field of every ``StrictConfigModel``, and for each one constructs a
config with the field set to a hostile value:

- ``nan`` / ``inf`` — a numeric field that accepts a non-finite value is
  a defect: the config then feeds that NaN into costs, thresholds, or
  gate bounds silently.
- ``-1`` on a ``*_bps``/``*_rate``/``*_limit``/``*_window`` field —
  negative physical quantities should reject.
- ``0`` on ``*_window``/``*_period``/``*_lookback`` fields — degenerate
  windows should reject.
- ``10**12`` — unbounded magnitudes are reported as notes (many are
  legitimately unbounded; no verdict impact).

Anything flagged is a real hardening gap — every accepted hostile value
flows silently into a downstream computation. Sealed
``validator_fuzz.v1``.
"""

from __future__ import annotations

import inspect
from typing import Any

import pydantic

from quant_fund.config import models as m
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_MODELS = [
    c
    for _, c in inspect.getmembers(m, inspect.isclass)
    if issubclass(c, m.StrictConfigModel) and c is not m.StrictConfigModel
]

_NEGATIVE_REJECT_HINTS = (
    "bps",
    "rate",
    "limit",
    "window",
    "period",
    "lookback",
    "width",
    "sigma",
    "tol",
    "eps",
    "horizon",
    "lag",
    "alpha",
    "kappa",
    "theta",
    "lambda",
    "floor",
    "threshold",
    "spread",
)
_ZERO_REJECT_HINTS = ("window", "period", "lookback", "horizon", "lag")


def _try(model: type, field: str, value: Any) -> bool:
    """True if the model accepts ``field=value`` (rest at defaults)."""
    try:
        model(**{field: value})
        return True
    except (pydantic.ValidationError, ValueError, TypeError, OverflowError):
        return False


def validator_fuzz() -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    n_fields = 0
    for model in _MODELS:
        for fname, finfo in model.model_fields.items():
            ann = finfo.annotation
            is_num = ann in (float, int)
            if not is_num:
                # Optional[float]/Optional[int]
                args = getattr(ann, "__args__", ())
                is_num = any(a in (float, int) for a in args)
            if not is_num:
                continue
            n_fields += 1
            tag = f"{model.__name__}.{fname}"
            for bad_name, bad in (("nan", float("nan")), ("inf", float("inf"))):
                if _try(model, fname, bad):
                    violations.append({"field": tag, "hostile": bad_name})
            low = fname.lower()
            is_float = ann is float
            if any(h in low for h in _NEGATIVE_REJECT_HINTS) and _try(
                model, fname, -1.0 if is_float else -1
            ):
                violations.append({"field": tag, "hostile": "negative"})
            if any(h in low for h in _ZERO_REJECT_HINTS) and _try(
                model, fname, 0.0 if is_float else 0
            ):
                violations.append({"field": tag, "hostile": "zero"})
            if _try(model, fname, 1e12):
                notes.append({"field": tag, "hostile": "1e12_accepted"})
    return {
        "n_models": len(_MODELS),
        "n_numeric_fields": n_fields,
        "n_violations": len(violations),
        "n_notes": len(notes),
        "verdict": "ok" if not violations else "violations",
        "violations": violations,
        "notes": notes[:30],
    }


def validator_fuzz_bench() -> dict[str, Any]:
    report = validator_fuzz()
    payload: dict[str, Any] = {
        "kind": "validator_fuzz",
        "schema": "validator_fuzz.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "no numeric config field accepts NaN/inf; physical fields reject negative/zero degenerate inputs",
            "verdict": report["verdict"],
            "n_numeric_fields": report["n_numeric_fields"],
            "n_violations": report["n_violations"],
        },
        "interpretation": report,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
