"""masking_audit — adversarial probes on the masked-twin eval machinery.

The memory-gap ship gate depends on masking actually removing recall
surfaces. Pinned contract:

- Deterministic and consistent: same surface → same ``<ASSET_*>`` /
  ``<DATE_*>`` placeholder across calls and tasks.
- Date formats covered: ISO, ``Month D, YYYY``, ``Qn YYYY``, bare year.
- ``mask_task`` masks ``required_tokens`` consistently so a reasoning
  model can still pass by emitting placeholders; forbidden patterns are
  carried verbatim (violations under masking are still violations).
- ``memory_gap_report`` fails closed on mismatched/empty inputs.

**Flagged escapes** (pinned, not fixed — scope of the ticker regex is a
design choice, but the escape must be visible):

- ``_TICKER_RE`` only matches 2–6 uppercase chars: ``"BITCOIN"`` (7)
  escapes, as does any longer all-caps ticker-like surface.
- Lowercase surface forms (``"aapl"``) are never masked — a recall
  channel when prompts use non-canonical casing.

Sealed ``masking_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["masking_audit", "masking_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
    except ValueError:
        return "raise:ValueError"
    return "no-raise"


def masking_audit() -> dict[str, Any]:
    from fx1.eval.masking import (
        mask_task,
        mask_text,
        masked_twins,
        memory_gap_report,
    )
    from fx1.eval.suite import EvalTask

    out: dict[str, Any] = {}
    m1 = mask_text("bought AAPL on 2024-01-15")
    m2 = mask_text("sold AAPL on 2024-02-01")
    out["deterministic"] = m1 == mask_text("bought AAPL on 2024-01-15")
    out["consistent_surface"] = "AAPL" not in m1 and "AAPL" not in m2
    # same surface maps to the same placeholder across different texts
    ph1 = [tok for tok in m1.split() if tok.startswith("<ASSET_")]
    ph2 = [tok for tok in m2.split() if tok.startswith("<ASSET_")]
    out["same_placeholder"] = ph1 == ph2

    for txt, gone in (
        ("close 2024-01-15", "2024-01-15"),
        ("in March 5, 2024 it fell", "March 5, 2024"),
        ("during Q3 2024", "Q3 2024"),
        ("year 2019 crisis", "2019"),
    ):
        out.setdefault("dates_masked", []).append(gone not in mask_text(txt))

    # allowlist survives (procedural vocabulary)
    out["allowlist_survives"] = all(
        tok in mask_text(f"the {tok} metric") for tok in ("SYNTHETIC", "TWAP")
    )

    # escapes (pinned findings)
    out["long_ticker_escapes"] = "BITCOIN" in mask_text("holding BITCOIN spot")
    out["lowercase_escapes"] = "aapl" in mask_text("bought aapl calls")

    # twin semantics
    t = EvalTask(
        name="probe",
        kind="domain",
        messages=[{"role": "user", "content": "what happened to AAPL in 2024?"}],
        required_tokens=["AAPL"],
        forbidden_patterns=[r"\blive p(?:&|and)l\b"],
    )
    twin = mask_task(t)
    out["twin_name"] = twin.name == "probe__masked"
    out["twin_required_masked"] = twin.required_tokens[0].startswith("<ASSET_")
    out["twin_forbidden_kept"] = twin.forbidden_patterns == t.forbidden_patterns
    twins = masked_twins([t])
    out["interleaved"] = len(twins) == 2 and twins[0].name == "probe"

    # gap report
    out["gap_mismatch_raises"] = _raises(lambda: memory_gap_report([True], [True, False]))
    out["gap_empty_raises"] = _raises(lambda: memory_gap_report([], []))
    rep = memory_gap_report([True] * 4, [True, True, False, True], budget=0.25)
    out["gap_value"] = rep.memory_gap
    out["gap_within"] = rep.within_budget  # 0.25 == budget → within (inclusive)
    rep_bad = memory_gap_report([True] * 4, [False] * 4, budget=0.25)
    out["gap_over_budget"] = rep_bad.within_budget is False
    return out


def masking_audit_bench() -> dict[str, Any]:
    r = masking_audit()
    ok = (
        all(
            r[k] is True
            for k in (
                "deterministic",
                "consistent_surface",
                "same_placeholder",
                "allowlist_survives",
                "twin_name",
                "twin_required_masked",
                "twin_forbidden_kept",
                "interleaved",
                "gap_over_budget",
                "gap_within",
            )
        )
        and all(r["dates_masked"])
        and r["gap_mismatch_raises"] == "raise:ValueError"
        and r["gap_empty_raises"] == "raise:ValueError"
        and abs(r["gap_value"] - 0.25) < 1e-9
    )
    out: dict[str, Any] = {
        "kind": "masking_audit",
        "schema": "masking_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {
                "long_ticker_escapes": r["long_ticker_escapes"],
                "lowercase_escapes": r["lowercase_escapes"],
            },
            "ok": ok,
        },
        "interpretation": (
            "Masking contract holds: deterministic consistent placeholders, "
            "all four date formats masked, allowlist survives, twins "
            "interleave, gap report fail-closed. Pinned escapes: >6-char "
            "tickers and lowercase forms evade the masker (visible in flags)."
            if ok
            else f"MASKING AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
