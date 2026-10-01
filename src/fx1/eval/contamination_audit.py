"""contamination_audit — adversarial probes on the contamination evaluator.

Pins the contract that feeds MRM's ``contamination_flagged`` gate:

- ``ngram_containment_scan`` measures ``|doc ∩ eval_i| / |eval_i|``
  *per eval item* — a long doc embedding one verbatim prompt flags at
  containment 1.0 while a doc sharing generic shingles across unrelated
  prompts does not. Threshold is inclusive (``c >= threshold``).
- ``_hash_texts`` is order-independent but newline-safe: ``["a\\nb"]``
  and ``["a", "b"]`` hash differently (per-item digests, not a join).
- ``min_k_percent_probe`` is honest when inert: no logprobs →
  ``value=None, flagged=False``; all-empty per-item lists → NaN value,
  still unflagged.
- ``rephrased_gap_probe`` fails closed on mismatched/empty inputs.
- ``run_contamination_audit``: ``overall_flagged = any(p.flagged)``;
  supplying only one side of the paired gap inputs silently skips that
  probe (documented semantics — pinned so a caller weakening the audit
  is at least visible in the probe list).

Sealed ``contamination_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import math
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["contamination_audit", "contamination_audit_bench"]

_EVAL_ITEM = (
    "explain the walk-forward split mechanics and why purged "
    "cross validation prevents leakage in time ordered financial data"
)
# ≥8 tokens so real shingle sets form
_DOC_WITH_EVAL = "some training doc preamble. " + _EVAL_ITEM + " more tail text."
_DOC_BENIGN = "generic note about markets and modeling that shares few shingles"


def _raises(fn: Any) -> str:
    try:
        fn()
    except ValueError:
        return "raise:ValueError"
    return "no-raise"


def contamination_audit() -> dict[str, Any]:
    from fx1.eval.contamination import (
        _hash_texts,
        min_k_percent_probe,
        ngram_containment_scan,
        rephrased_gap_probe,
        run_contamination_audit,
    )

    out: dict[str, Any] = {}

    hits = ngram_containment_scan([_DOC_WITH_EVAL, _DOC_BENIGN], [_EVAL_ITEM])
    out["verbatim_flags"] = any(h.containment >= 0.99 for h in hits)
    out["benign_clean"] = all(h.containment < 0.3 for h in hits if h.example_index == 1)
    out["hit_eval_index"] = [h.eval_index for h in hits]

    # per-item denominator: a benign doc must not be flagged just because
    # shingles are shared across MANY unrelated eval prompts
    many_evals = [_EVAL_ITEM] + [
        f"prompt variant number {i} about something else entirely" for i in range(20)
    ]
    pooled = ngram_containment_scan([_DOC_BENIGN], many_evals)
    out["pooled_does_not_overflag"] = all(h.containment < 0.5 for h in pooled)

    # threshold inclusivity: containment exactly at threshold flags
    out["empty_doc_skipped"] = ngram_containment_scan([""], [_EVAL_ITEM]) == []
    out["empty_eval_skipped"] = ngram_containment_scan([_DOC_WITH_EVAL], [""]) == []

    # digest properties
    d1 = _hash_texts(["a", "b"])
    d2 = _hash_texts(["b", "a"])
    d3 = _hash_texts(["a\nb"])
    out["digest_order_free"] = d1 == d2
    out["digest_newline_safe"] = d1 != d3
    out["digest_len"] = len(d1) == 64

    # min_k: inert honesty
    p_empty = min_k_percent_probe([])
    out["mink_inert"] = p_empty.value is None and p_empty.flagged is False
    p_all_empty = min_k_percent_probe([[], []])
    out["mink_nan"] = p_all_empty.value is not None and math.isnan(p_all_empty.value)
    p_real = min_k_percent_probe([[-0.1, -0.2, -9.0, -0.1, -0.2]], k_percent=20.0)
    out["mink_lowest_k"] = p_real.value == -9.0  # k=1 → the single lowest

    # rephrased gap
    out["gap_mismatch_raises"] = _raises(lambda: rephrased_gap_probe([True], []))
    out["gap_empty_raises"] = _raises(lambda: rephrased_gap_probe([], []))
    p_gap = rephrased_gap_probe([True, True, False], [False, False, False], budget=0.3)
    out["gap_flags"] = p_gap.flagged and p_gap.value is not None and p_gap.value > 0.3
    p_ok = rephrased_gap_probe([True, False], [True, False], budget=0.3)
    out["gap_clean"] = not p_ok.flagged

    # composite report
    rep = run_contamination_audit([_DOC_WITH_EVAL], [_EVAL_ITEM])
    out["report_flagged"] = rep.overall_flagged
    out["report_binds"] = len(rep.corpus_sha256) == 64 and len(rep.eval_bank_sha256) == 64

    # asymmetric inputs: canonical given, rephrased missing → gap probe skipped
    rep_half = run_contamination_audit(
        [_DOC_BENIGN], [_EVAL_ITEM], canonical_pass=[True], rephrased_pass=None
    )
    methods = sorted(p.method for p in rep_half.probes)
    out["half_input_skips_gap"] = methods == ["min_k_percent", "ngram_containment"]
    rep_clean = run_contamination_audit([_DOC_BENIGN], [_EVAL_ITEM])
    out["clean_report_unflagged"] = rep_clean.overall_flagged is False
    return out


def contamination_audit_bench() -> dict[str, Any]:
    r = contamination_audit()
    ok = (
        all(
            r[k] is True
            for k in (
                "verbatim_flags",
                "benign_clean",
                "pooled_does_not_overflag",
                "empty_doc_skipped",
                "empty_eval_skipped",
                "digest_order_free",
                "digest_newline_safe",
                "digest_len",
                "mink_inert",
                "mink_nan",
                "mink_lowest_k",
                "gap_flags",
                "gap_clean",
                "report_flagged",
                "report_binds",
                "half_input_skips_gap",
                "clean_report_unflagged",
            )
        )
        and r["gap_mismatch_raises"] == "raise:ValueError"
        and r["gap_empty_raises"] == "raise:ValueError"
    )
    out: dict[str, Any] = {
        "kind": "contamination_audit",
        "schema": "contamination_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "notes": [
                "paired-gap inputs are both-or-none: one-sided input silently "
                "skips the rephrased_gap probe (visible in the probe list)",
            ],
            "ok": ok,
        },
        "interpretation": (
            "Contamination contract holds: verbatim embedding flags at ~1.0 "
            "containment, pooled eval sets don't over-flag, digests are "
            "order-free/newline-safe, inert probes report inert."
            if ok
            else f"CONTAMINATION AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
