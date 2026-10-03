"""honesty_audit — adversarial probes on the fx-1 honesty gate itself.

Pinned contract:

- Headline claims — a forbidden token (sharpe/sortino/calmar/pnl/nav)
  followed by a numeric value through any connector — refuse.
- Live-performance and synthetic-as-live phrases refuse.
- Synthetic presentation without the uppercase SYNTHETIC label refuses;
  labeled, and non-presenting discussion, pass.
- Documented escape surfaces (flagged, not fixed): spelled-out numerals
  carry no ``\\d`` so "Sharpe of two" passes; a mid-word token glued to a
  prefix ("mySharpe 2.1") has no leading boundary and passes.

Sealed ``honesty_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["honesty_audit", "honesty_audit_bench"]

_HEADLINE_CASES = (
    "Sharpe 2.1",
    "pnl: +$4,200",
    "nav (1.9)",
    "sharpe…2.1",
    "Sortino ratio of 3.4",
    "calmar was 0.8",
    "the Sharpe — 2.1",
    "NAV at 1.05",
)
_CLAIM_CASES = (
    "live p&l of $4k",
    "live trading returns of 12%",
    "real-money gains",
    "guaranteed returns",
    "synthetic results prove live performance",
)
_CLEAN_CASES = (
    "Sharpe is a forbidden headline token in this lab.",
    "The desk tracks risk with proper scores only.",
    "No numeric claim here.",
)


def honesty_audit() -> dict[str, Any]:
    from fx1.honesty import Fx1HonestyError, validate_fx1_output

    out: dict[str, Any] = {}
    out["headlines_refuse"] = all(
        _raises(lambda t=t: validate_fx1_output(t)) == "Fx1HonestyError" for t in _HEADLINE_CASES
    )
    out["live_claims_refuse"] = all(
        _raises(lambda t=t: validate_fx1_output(t)) == "Fx1HonestyError" for t in _CLAIM_CASES
    )
    out["clean_passes"] = all(validate_fx1_output(t) == t for t in _CLEAN_CASES)
    # synthetic presentation needs the uppercase label
    out["unlabeled_synthetic_refuses"] = (
        _raises(lambda: validate_fx1_output("the synthetic backtest shows 12%"))
        == "Fx1HonestyError"
    )
    out["labeled_synthetic_passes"] = (
        validate_fx1_output("the SYNTHETIC backtest shows 12%")
        == "the SYNTHETIC backtest shows 12%"
    )
    out["lowercase_label_insufficient"] = (
        _raises(lambda: validate_fx1_output("the Synthetic backtest shows 12%"))
        == "Fx1HonestyError"
    )
    # commentary: synthetic mention with no numeric claim passes
    out["synthetic_commentary_passes"] = (
        validate_fx1_output("synthetic data is not market evidence")
        == "synthetic data is not market evidence"
    )
    # numeric but non-presenting verb passes (no show/prove/score/result…)
    out["numeric_nonpresenting_passes"] = (
        validate_fx1_output("the synthetic set has 12 rows") == "the synthetic set has 12 rows"
    )
    # flagged escape surfaces — pinned as actual behavior, not fixed
    out["flag_spelled_number_evades"] = validate_fx1_output("Sharpe of two") == "Sharpe of two"
    out["flag_glued_prefix_evades"] = validate_fx1_output("mySharpe 2.1") == "mySharpe 2.1"
    out["error_is_value_error"] = issubclass(Fx1HonestyError, ValueError)
    out["identity_return"] = validate_fx1_output("clean") == "clean"
    return out


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def honesty_audit_bench() -> dict[str, Any]:
    r = honesty_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "honesty_audit",
        "schema": "honesty_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Honesty gate holds: headline tokens, live-claim phrases, and "
            "unlabeled synthetic presentation all refuse; clean discussion "
            "passes. Flagged surfaces: spelled-out numerals and glued "
            "prefixes still evade — pinned, not fixed."
            if ok
            else f"HONESTY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
