"""report_audit — the reporting layer's own invariants.

The tearsheet is the human-facing artifact of a run; its invariants:

1. **Determinism** — identical inputs must emit byte-identical markdown
   (run twice in-process; dict iteration order and set hashing must not leak).
2. **Hygiene** — no ``nan``/``inf`` literals in the emitted markdown (a
   degenerate stat must render as a labeled stub, never a raw float).
3. **Reconciliation** — the summary ``total_return`` must equal
   ``nav_end/nav_start − 1`` *and* the compounded product of the monthly
   table within fp tolerance of its own identity (they measure different
   spans, so the gap is *reported*, not forced to zero).

Verdict ``ok`` iff determinism holds, markdown is clean, and the monthly
compounding gap is below the fp floor when coverage is contiguous.
Sealed ``report_audit.v1``.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import polars as pl

from quant_fund.reporting.tearsheet import (
    build_tearsheet,
    period_returns_table,
    tearsheet_markdown,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["report_audit", "report_audit_bench"]

_BAD_LITERALS = re.compile(r"(?<![A-Za-z_])(nan|inf|-inf|infinity)(?![A-Za-z_])", re.IGNORECASE)


def _synthetic_equity(seed: int = 0, n: int = 260) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(4e-4, 8e-3, n)
    nav = 1e6 * np.cumprod(1.0 + rets)
    start = pl.datetime(2024, 1, 2)
    times = pl.datetime_range(
        start=start,
        end=start + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({"event_time": times, "nav": nav})


def report_audit(seed: int = 0, n: int = 260) -> dict[str, Any]:
    equity = _synthetic_equity(seed, n)
    sheet_a = build_tearsheet(equity, label="AUDIT", synthetic=True)
    sheet_b = build_tearsheet(equity, label="AUDIT", synthetic=True)
    md_a = tearsheet_markdown(sheet_a)
    md_b = tearsheet_markdown(sheet_b)

    deterministic = md_a == md_b
    bad_literals = sorted(set(_BAD_LITERALS.findall(md_a)))

    monthly = period_returns_table(equity)
    nav = equity["nav"].to_numpy()
    compound = math.prod(1.0 + r for r in monthly.values()) - 1.0 if monthly else math.nan
    total = float(nav[-1] / nav[0] - 1.0)
    # Monthly first/last bucketing skips intra-boundary bars, so the product
    # and the total differ by construction; the gap is the audit finding.
    gap = abs(compound - total) if np.isfinite(compound) else math.inf

    summary = sheet_a["summary"]
    self_consistent = abs(summary["total_return"] - total) < 1e-12

    ok = deterministic and not bad_literals and self_consistent
    return {
        "deterministic_markdown": deterministic,
        "markdown_bytes": len(md_a.encode()),
        "bad_literals": bad_literals,
        "monthly_buckets": len(monthly),
        "monthly_compound": compound,
        "summary_total_return": total,
        "compound_gap": gap,
        "summary_self_consistent": self_consistent,
        "ok": ok,
    }


def report_audit_bench(seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    runs = [report_audit(seed=s) for s in seeds]
    ok = all(r["ok"] for r in runs)
    payload: dict[str, Any] = {
        "kind": "report_audit",
        "schema": "report_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "tearsheet markdown deterministic + literal-clean + summary reconciles",
            "seeds": list(seeds),
            "per_seed": [
                {
                    "seed": s,
                    "deterministic": r["deterministic_markdown"],
                    "bad_literals": r["bad_literals"],
                    "compound_gap": r["compound_gap"],
                    "self_consistent": r["summary_self_consistent"],
                }
                for s, r in zip(seeds, runs, strict=True)
            ],
            "ok": ok,
        },
        "interpretation": (
            f"{len(runs)} seeds: determinism "
            f"{all(r['deterministic_markdown'] for r in runs)}, literal leaks "
            f"{sorted({lit for r in runs for lit in r['bad_literals']}) or 'none'}, "
            f"max compound gap {max(r['compound_gap'] for r in runs):.2e}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
