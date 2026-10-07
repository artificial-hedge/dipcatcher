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
- p-values are pooled into one Benjamini-Hochberg family at level ``q``.
  A receipt's declared ``family`` tag is recorded on each finding for
  provenance; the BH run itself pools — the conservative direction,
  since a larger family tightens the per-rank critical value.
- e-values are kept in a separate bucket and merged by arithmetic mean:
  the mean of e-values is an e-value under *arbitrary dependence*
  (Vovk–Wang 2021), which receipts require — they are produced by
  correlated experiments on overlapping data, so the independence a
  product merge needs cannot be assumed. The product is still reported
  as a labeled diagnostic (valid only under independence).
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
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.receipt import verified_corpus_files
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
    members: Collection[str] | None = None,
) -> dict[str, Any]:
    """Pool every committed receipt's claims into one FDR family.

    Returns a ``corpus_inference.v1`` receipt dict: per-claim survival
    flags, the merged corpus e-value (arithmetic mean — valid under
    arbitrary dependence), and counts. Fails closed on a missing dir;
    unreadable receipts are recorded as errors rather than silently
    skipped. ``members`` pins the input set to exactly those basenames —
    without it the glob picks up whatever the dir contains, so a pinned
    membership is required for byte-reproducible replays. A member absent
    from the dir is a recorded error, never silently dropped.
    """
    root = Path(receipts_dir)
    if not root.is_dir():
        raise ValueError(f"receipts dir {root} does not exist")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")

    findings: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if members is not None:
        member_set = {str(m) for m in members}
        root_resolved = root.resolve()
        present = {
            p.relative_to(root_resolved).as_posix()
            for p in verified_corpus_files(root, pattern=glob)
        }
        receipt_files = []
        for name in sorted(member_set):
            p = (root / name).resolve()
            if not p.is_relative_to(root_resolved):
                # a pinned member must name a corpus file — a traversal name
                # would attest outside bytes under the member's basename
                errors.append({"file": name, "error": "member_path_uncontained"})
                continue
            if name not in present:
                errors.append({"file": name, "error": "member_missing_from_dir"})
                continue
            receipt_files.append(p)
        # Files in the dir that are not members are ignored — a pinned
        # membership audits the same frozen set even as the corpus grows.
    else:
        receipt_files = verified_corpus_files(root, pattern=glob)
    digests: dict[str, str] = {}
    input_labels: dict[str, str] = {}
    # Retractions exclude their target's findings from the inference pool —
    # a retracted claim must not keep scoring under FDR.
    from quant_fund.research.receipt_tombstone import load_tombstones

    tombs = load_tombstones(root)
    retracted = tombs["active"]
    for bad in tombs["invalid"]:
        errors.append({"file": bad.split(":", 1)[0], "error": bad.split(":", 1)[1]})
    for path in receipt_files:
        rel_name = path.relative_to(root.resolve()).as_posix()
        try:
            raw = path.read_bytes()
            digests[rel_name] = hash_bytes(raw)
            doc = json.loads(raw)
            if not isinstance(doc, Mapping):
                raise ValueError("receipt root is not an object")
            body = doc.get("payload")
            inner = body if isinstance(body, Mapping) else doc
            input_labels[rel_name] = str(inner.get("data_label") or "UNKNOWN")
            if inner.get("kind") == "receipt_tombstone.v1":
                continue
            scope = (retracted.get(rel_name) or {}).get("scope")
            if scope == "all":
                continue
            scoped = set(scope) if isinstance(scope, list) else None
            for f in harvest_findings(doc, rel_name):
                if scoped is not None and f["path"] in scoped:
                    continue
                findings.append(f)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
        ) as exc:
            # Narrowed from `except Exception` (quality ratchet): receipt read/parse
            # faults are IO/JSON plus the explicit shape ValueError and the untrusted-
            # document walk in harvest_findings; exotic errors propagate. Recorded,
            # never skipped.
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})

    p_findings = [f for f in findings if f["stat"] == "p"]
    e_findings = [f for f in findings if f["stat"] == "e"]
    flags = _bh_survivors([f["value"] for f in p_findings], q)
    for f, ok in zip(p_findings, flags, strict=True):
        f["survives_fdr"] = bool(ok)

    corpus_evalue_product = 1.0
    for f in e_findings:
        corpus_evalue_product *= float(f["value"])
    if e_findings:
        # Mean of e-values is an e-value under arbitrary dependence
        # (Vovk & Wang 2021) — receipts share data and models, so the
        # independence a product merge requires cannot be assumed. The
        # product is kept as a labeled diagnostic only.
        corpus_evalue = float(np.mean([float(f["value"]) for f in e_findings]))
    else:
        corpus_evalue = 1.0

    surviving = [f for f in p_findings if f.get("survives_fdr")]
    inputs_sha256 = hash_bytes(canonical_json_bytes({"digests": digests, "q": q}))
    # Corpus-level fingerprint: digest over the audited receipt file
    # contents only — corpus lanes over the same directory agree on it,
    # which is what the cross-receipt lattice edges on.
    dataset_sha256 = hash_bytes(
        canonical_json_bytes({"shards": {name: {"file_sha256": d} for name, d in digests.items()}})
    )
    distinct_labels = set(input_labels.values())
    if len(distinct_labels) == 1:
        data_label = distinct_labels.pop()
    elif distinct_labels:
        data_label = "MIXED"
    else:
        data_label = "UNKNOWN"
    return {
        "kind": CORPUS_SCHEMA,
        "schema": CORPUS_SCHEMA,
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": inputs_sha256,
        "dataset_sha256": dataset_sha256,
        "params": {"q": q, "glob": glob, "input_labels": input_labels},
        "n_receipts": len(receipt_files),
        "n_parse_errors": len(errors),
        "parse_errors": errors,
        "n_p_findings": len(p_findings),
        "n_e_findings": len(e_findings),
        "n_retracted": len(retracted),
        "retracted": {name: t["tombstone"] for name, t in sorted(retracted.items())},
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
        # diagnostic only — a valid e-value solely under independence
        "corpus_evalue_product_dependence_assuming": corpus_evalue_product,
        "corpus_reject_at_alpha": corpus_evalue >= 1.0 / q if e_findings else False,
        "evidence": [
            "bh_fdr_pooled_family",
            "evalue_mean_merge",
            "evalue_product_merge_diagnostic_only",
            "tagged_claim_provenance",
        ],
    }


def write_corpus_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a corpus_inference receipt and write ``corpus_inference_<hash>.json``.

    Filename digest = ``inputs_sha256`` (v1) or the canonical
    ``receipt_sha256`` (v2). Atomic, fail-closed on a malformed receipt.
    ``receipt_version=2`` wraps the same body in the unified ``receipt.v2``
    envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if (
        receipt.get("kind") != CORPUS_SCHEMA
        or receipt.get("schema") != CORPUS_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
    ):
        raise ValueError("corpus_inference receipt violates its contract")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
        name_digest = str(receipt.get("inputs_sha256") or digest)[:16]
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass",
            )
        )
        name_digest = str(payload["receipt_sha256"])[:16]
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"corpus_inference_{name_digest}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
