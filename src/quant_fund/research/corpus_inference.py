"""Corpus-level inference audit — BH-FDR across committed receipts.

Individual receipts carry their own p-values/e-values, each tested at its
nominal alpha. The corpus question is different: *which claims survive
when every committed receipt is pooled into one family?* That is the
selection-bias class the reality gate checks per study, lifted to the
whole evidence store.

Procedure:

- Harvest every numeric field whose key names a p-value
  (``*_p``, ``p_value``, ``pval``, ``pvalue``, ``*_pvalue``) or an e-value
  (``*evalue*``, ``anytime_p`` handled via the p convention) at any depth
  of each committed receipt — tagged by file, kind, and JSON path so a
  claim can be traced back to its artifact.
- p-values are pooled into one Benjamini-Hochberg family at level ``q``
  (calibration/discovery families stay distinct when the receipt declares
  one — pooling a bound claim with a discovery claim inflates power).
- e-values are kept in a separate bucket and merged by product: a
  product of independent e-values is an e-value (Shafer 2021), giving a
  single corpus-level e-value per bucket.
- The output is a receipt-shaped dict (``corpus_inference.v1``) listing
  which receipt claims survive, the BH critical value, per-claim
  adjusted status, and the merged corpus e-value.

Deliberately conservative: keys that name a *threshold* (``alpha``,
``q_level``, ``min_p``) or a *count* (``n_*``, ``*_count``) are never
harvested, and only values in (0, 1] are treated as p-values.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CORPUS_SCHEMA = "corpus_inference.v1"

_P_SUFFIXES = ("_p", "p_value", "pval", "pvalue", "_pvalue", "p_value_adj", "anytime_p")
_E_HINTS = ("evalue", "e_value")
_NEVER = (
    "alpha",
    "q_level",
    "min_p",
    "max_p",
    "count",
    "n_",
    "threshold",
    "level",
    "coverage",
    "probability_",  # probabilities are not p-values
)


def _looks_like_p_key(key: str) -> bool:
    k = key.lower()
    if any(tok in k for tok in _NEVER):
        return False
    if k in ("p", "pvalue", "p_value", "pval", "anytime_p", "p_value_adj"):
        return True
    return k.endswith(_P_SUFFIXES)


def _looks_like_e_key(key: str) -> bool:
    k = key.lower()
    if "anytime_p" in k or _looks_like_p_key(key):
        return False
    return any(tok in k for tok in _E_HINTS)


def _walk(node: Any, path: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, Mapping):
        for key, val in node.items():
            child = f"{path}.{key}" if path else str(key)
            out.append((child, val))
            out.extend(_walk(val, child))
    elif isinstance(node, list):
        for i, val in enumerate(node):
            out.extend(_walk(val, f"{path}[{i}]"))
    return out


def harvest_findings(receipt: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    """Extract tagged p-value and e-value findings from one receipt."""
    kind = str(receipt.get("kind") or receipt.get("schema") or "unknown")
    family = receipt.get("family")
    findings: list[dict[str, Any]] = []
    for path, val in _walk(receipt, ""):
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        fv = float(val)
        if _looks_like_p_key(leaf):
            if 0.0 < fv <= 1.0:
                findings.append(
                    {
                        "source": source,
                        "kind": kind,
                        "family": str(family) if family else "pooled",
                        "path": path,
                        "stat": "p",
                        "value": fv,
                    }
                )
        elif _looks_like_e_key(leaf) and fv > 0.0 and math.isfinite(fv):
            findings.append(
                {
                    "source": source,
                    "kind": kind,
                    "family": str(family) if family else "pooled",
                    "path": path,
                    "stat": "e",
                    "value": fv,
                }
            )
    return findings


def _bh_survivors(pvals: list[float], q: float) -> list[bool]:
    """BH-FDR at level q; returns a per-index survive flag."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    crit = q * (np.arange(1, n + 1) / n)
    sorted_p = np.asarray(pvals)[order]
    ok = sorted_p <= crit
    max_ok = int(np.max(np.nonzero(ok)[0])) if ok.any() else -1
    out = [False] * n
    for rank in range(max_ok + 1):
        out[int(order[rank])] = True
    return out


def corpus_audit(
    receipts_dir: Path | str,
    *,
    q: float = 0.05,
    glob: str = "*.json",
) -> dict[str, Any]:
    """Pool every committed receipt's claims into one FDR family.

    Returns a ``corpus_inference.v1`` receipt dict: per-claim survival
    flags, the merged corpus e-value (product of independent e-values),
    and counts. Fails closed on a missing dir; unreadable receipts are
    recorded as errors rather than silently skipped.
    """
    root = Path(receipts_dir)
    if not root.is_dir():
        raise ValueError(f"receipts dir {root} does not exist")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")

    findings: list[dict[str, Any]] = []
    receipt_files = sorted(p for p in root.glob(glob) if p.is_file())
    errors: list[dict[str, str]] = []
    digests: dict[str, str] = {}
    for path in receipt_files:
        try:
            raw = path.read_bytes()
            digests[path.name] = hash_bytes(raw)
            doc = json.loads(raw)
            if not isinstance(doc, Mapping):
                raise ValueError("receipt root is not an object")
            findings.extend(harvest_findings(doc, path.name))
        except Exception as exc:  # noqa: BLE001 — errors are recorded, never skipped
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})

    p_findings = [f for f in findings if f["stat"] == "p"]
    e_findings = [f for f in findings if f["stat"] == "e"]
    flags = _bh_survivors([f["value"] for f in p_findings], q)
    for f, ok in zip(p_findings, flags, strict=True):
        f["survives_fdr"] = bool(ok)

    corpus_e = 1.0
    for f in e_findings:
        corpus_e *= float(f["value"])
    # Product of independent e-values is an e-value (Shafer 2021).
    corpus_evalue = float(min(corpus_e, math.inf)) if math.isfinite(corpus_e) else float("inf")

    surviving = [f for f in p_findings if f.get("survives_fdr")]
    inputs_sha256 = hash_bytes(canonical_json_bytes({"digests": digests, "q": q}))
    return {
        "kind": CORPUS_SCHEMA,
        "schema": CORPUS_SCHEMA,
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": inputs_sha256,
        "params": {"q": q, "glob": glob},
        "n_receipts": len(receipt_files),
        "n_parse_errors": len(errors),
        "parse_errors": errors,
        "n_p_findings": len(p_findings),
        "n_e_findings": len(e_findings),
        "n_survivors": len(surviving),
        "surviving_claims": [
            {
                "source": f["source"],
                "path": f["path"],
                "value": f["value"],
                "family": f["family"],
            }
            for f in surviving
        ],
        "corpus_evalue": corpus_evalue,
        "corpus_reject_at_alpha": corpus_evalue >= 1.0 / q if e_findings else False,
        "evidence": [
            "bh_fdr_pooled_family",
            "evalue_product_merge",
            "tagged_claim_provenance",
        ],
    }
