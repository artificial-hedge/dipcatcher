"""timepart_audit — adversarial probes on post-cutoff task partitioning.

The ship gate treats post-cutoff domain performance as the
least-contaminated competence estimate, so the partition boundary must be
pinned exactly:

- ``stamp > cutoff`` → post; ``stamp == cutoff`` → **pre** (the item was
  created at/before the cutoff — it could be in the pretraining corpus);
  missing provenance → ``undated``, excluded from both headline slices.
- Comparison is string ordering on ISO stamps: well-formed
  ``YYYY-MM-DD`` compares correctly, but a malformed stamp sorts
  lexicographically like any string — ``"2025-13-99" > "2025-01-01"``
  lands in *post*. Pinned as a visible caveat (the caller owns date
  hygiene; the partition itself does not validate).
- ``post_cutoff_pass_rate`` returns ``None`` on an empty post slice
  (no evidence, not zero) and treats a missing result as a fail.

Sealed ``timepart_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["timepart_audit", "timepart_audit_bench"]

_CUTOFF = "2025-01-01"


def _task(name: str) -> Any:
    from fx1.eval.suite import EvalTask

    return EvalTask(
        name=name,
        kind="domain",
        messages=[{"role": "user", "content": "q"}],
    )


def timepart_audit() -> dict[str, Any]:
    from fx1.eval.timepart import partition_tasks, post_cutoff_pass_rate

    out: dict[str, Any] = {}
    tasks = [_task(n) for n in ("pre", "at", "post", "undated", "garbage")]
    created = {
        "pre": "2024-06-01",
        "at": _CUTOFF,  # boundary → pre
        "post": "2025-06-15",
        "garbage": "2025-99-99",  # malformed ISO, sorts post
    }
    p = partition_tasks(tasks, created, _CUTOFF)
    out["boundary_is_pre"] = "at" in p.pre_cutoff
    out["post_membership"] = sorted(p.post_cutoff)
    out["undated_membership"] = p.undated
    out["malformed_sorts_post"] = "garbage" in p.post_cutoff

    out["empty_post_none"] = (
        post_cutoff_pass_rate(
            partition_tasks([_task("only")], {"only": "2020-01-01"}, _CUTOFF),
            {"only": True},
        )
        is None
    )
    out["missing_result_fails"] = (
        post_cutoff_pass_rate(partition_tasks(tasks, created, _CUTOFF), {}) == 0.0
    )  # post tasks absent from results → counted as fail
    out["pass_rate_value"] = post_cutoff_pass_rate(
        partition_tasks(tasks, created, _CUTOFF),
        {"post": True, "garbage": False},
    )
    return out


def timepart_audit_bench() -> dict[str, Any]:
    r = timepart_audit()
    ok = (
        r["boundary_is_pre"] is True
        and r["post_membership"] == ["garbage", "post"]
        and r["undated_membership"] == ["undated"]
        and r["malformed_sorts_post"] is True
        and r["empty_post_none"] is True
        and r["missing_result_fails"] is True
        and abs(r["pass_rate_value"] - 0.5) < 1e-9
    )
    out: dict[str, Any] = {
        "kind": "timepart_audit",
        "schema": "timepart_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {"malformed_dates_sort_lexically": r["malformed_sorts_post"]},
            "ok": ok,
        },
        "interpretation": (
            "Partition contract holds: boundary stamp is pre, undated is "
            "excluded from both slices, empty post slice reports None, "
            "missing results count as fails. Caveat pinned: malformed "
            "provenance dates sort lexicographically into post."
            if ok
            else f"TIMEPART AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
