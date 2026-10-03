"""Corpus-wide replay sweep — every committed replay carrier, one verdict.

``dipcatcher replay`` proves a single carrier; this module sweeps the
whole carrier directory (``data/manifests/replay/*.json``) and seals a
``replay_coverage.v1`` receipt: per-carrier verdicts, the corpus's
reproducibility counts, and an aggregate ``coverage`` fraction — the
single number answering "what fraction of the declared lane corpus
reproduces byte-for-byte under this checkout".

Skip classes are honest, never folded into failures:

- ``inputs_not_verified`` → ``skipped`` (the lane could not run here —
  e.g. tape absent on CI). Honest, never counted as a failure.
- ``fail`` means the lane *ran* and its produced bytes diverged, or it
  exited non-zero, or its manifest is dangerous (committed-overwrite
  guard, cwd escape) — the class the gate exists for.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.replay_proof import run_replay

REPLAY_COVERAGE_SCHEMA = "replay_coverage.v1"

__all__ = [
    "REPLAY_COVERAGE_SCHEMA",
    "replay_coverage_contract_errors",
    "run_replay_sweep",
]

_SKIP_CLASSES = {"inputs_not_verified"}


def run_replay_sweep(
    carriers_dir: Path | str,
    *,
    root: Path | str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Replay every ``replay_manifest.v1`` carrier under ``carriers_dir``.

    Returns the ``replay_coverage.v1`` body (unsealed — callers seal via
    ``seal_receipt``/``wrap_receipt_v2``). Raises ``ValueError`` when the
    directory is absent or carries no well-formed carriers — an empty
    sweep attests nothing.
    """
    carriers_path = Path(carriers_dir)
    root_path = Path(root).resolve()
    if not carriers_path.is_dir():
        raise ValueError(f"carriers directory does not exist: {carriers_path}")

    rows: list[dict[str, Any]] = []
    n_claims_total = 0
    n_claims_byte_equal = 0
    for carrier in sorted(carriers_path.glob("*.json")):
        try:
            carrier_body = json.loads(carrier.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"carrier unreadable: {carrier.name}: {exc}") from exc
        if carrier_body.get("schema") != "replay_manifest.v1":
            continue

        try:
            rel = str(carrier.resolve().relative_to(root_path))
        except ValueError:
            rel = carrier.name
        try:
            proof = run_replay(carrier, root=root_path, timeout_s=timeout_s)
        except ValueError as exc:
            rows.append(
                {
                    "carrier": rel,
                    "verdict": "fail",
                    "note": f"manifest_malformed:{exc}",
                }
            )
            continue

        skipped = proof.get("execution_skipped")
        row: dict[str, Any] = {
            "carrier": rel,
            "verdict": proof["verdict"] if skipped is None else "skipped",
            "all_match": proof["all_match"],
            "exit_code": proof["exit_code"],
        }
        if skipped is not None:
            row["skipped_reason"] = skipped
        rows.append(row)

        reproduces = carrier_body.get("reproduces")
        if isinstance(reproduces, list):
            n_claims_total += len(reproduces)
            n_claims_byte_equal += sum(
                1
                for item in reproduces
                if isinstance(item, Mapping) and item.get("claims_equal") is True
            )

    if not rows:
        raise ValueError(f"no replay_manifest.v1 carriers under {carriers_path}")

    n_pass = sum(1 for row in rows if row["verdict"] == "pass")
    n_fail = sum(1 for row in rows if row["verdict"] == "fail")
    n_skipped = sum(1 for row in rows if row["verdict"] == "skipped")
    coverage = n_pass / len(rows)
    verdict = "pass" if n_fail == 0 and n_pass > 0 else "fail"

    return {
        "schema": REPLAY_COVERAGE_SCHEMA,
        "kind": "replay_coverage",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "carriers_dir": str(carriers_path),
        "rows": rows,
        "n_carriers": len(rows),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skipped": n_skipped,
        "coverage": coverage,
        "n_claims_total": n_claims_total,
        "n_claims_byte_equal": n_claims_byte_equal,
        "claims_byte_equal_frac": (n_claims_byte_equal / n_claims_total)
        if n_claims_total
        else None,
        "timeout_s": float(timeout_s),
        "verdict": verdict,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def replay_coverage_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Internal consistency for a ``replay_coverage.v1`` body — fail closed.

    Re-derives every count from the carrier rows plus the headline
    ``verdict == ("pass" iff n_fail == 0 and n_pass > 0)`` equivalence, so
    a body lying about its own arithmetic fails under a fresh seal.
    """
    if not isinstance(payload, Mapping):
        return ["payload_not_object"]
    errors: list[str] = []
    if payload.get("schema") != REPLAY_COVERAGE_SCHEMA:
        errors.append("schema")
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if payload.get("data_label") != "CORPUS":
        errors.append("data_label")

    rows = payload.get("rows")
    row_verdicts: list[str] = []
    if not isinstance(rows, list) or not rows:
        errors.append("rows")
    else:
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"rows[{index}]")
                continue
            verdict = row.get("verdict")
            if verdict not in ("pass", "fail", "skipped"):
                errors.append(f"rows[{index}].verdict")
                verdict = "fail"
            row_verdicts.append(str(verdict))
            carrier = row.get("carrier")
            if not isinstance(carrier, str) or not carrier.strip():
                errors.append(f"rows[{index}].carrier")
            if verdict == "skipped" and row.get("skipped_reason") not in _SKIP_CLASSES:
                errors.append(f"rows[{index}].skipped_reason")

    for field in ("n_carriers", "n_pass", "n_fail", "n_skipped"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(field)
    if row_verdicts and not errors:
        n_rows = len(row_verdicts)
        n_pass = row_verdicts.count("pass")
        n_fail = row_verdicts.count("fail")
        n_skipped = row_verdicts.count("skipped")
        if payload.get("n_carriers") != n_rows:
            errors.append("n_carriers")
        if payload.get("n_pass") != n_pass:
            errors.append("n_pass")
        if payload.get("n_fail") != n_fail:
            errors.append("n_fail")
        if payload.get("n_skipped") != n_skipped:
            errors.append("n_skipped")
        coverage = payload.get("coverage")
        if (
            not isinstance(coverage, (int, float))
            or isinstance(coverage, bool)
            or not math.isclose(float(coverage), n_pass / n_rows, rel_tol=0, abs_tol=1e-12)
        ):
            errors.append("coverage")
        expected = "pass" if n_fail == 0 and n_pass > 0 else "fail"
        if payload.get("verdict") != expected:
            errors.append("verdict")

    for field in ("n_claims_total", "n_claims_byte_equal"):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(field)
    frac = payload.get("claims_byte_equal_frac")
    if frac is not None and (
        not isinstance(frac, (int, float)) or isinstance(frac, bool) or not math.isfinite(frac)
    ):
        errors.append("claims_byte_equal_frac")

    return errors
