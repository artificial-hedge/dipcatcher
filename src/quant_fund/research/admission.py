"""Receipt admission gate — decides whether a new receipt may join the corpus.

``verify-receipt`` answers "is this receipt valid"; ``dipcatcher admit``
answers "does this receipt deserve to sit in the evidence corpus next to
everything already there". A valid-but-contradictory receipt must not merge
quietly — it is quarantined until its lattice deltas are pinned or resolved.

Checks, in order:
1. ``seal``       — ``verify_receipt_file`` (signature, contract, integrity).
2. ``honesty``    — ``research_only is True`` and ``live_pnl_claim is False``
   declared somewhere the seal covers, ``data_label`` a non-empty string.
   Asserted dishonesty (``live_pnl_claim is True``) rejects outright.
3. ``lattice``    — run the receipt lattice over the corpus with and without
   the candidate; any *new* ``inconsistent`` claim group involving the
   candidate quarantines it.
4. ``corpus``     — corpus_audit on both corpora; the survivor-set delta is
   recorded as evidence (BH families legitimately shift as members arrive).

Verdicts: ``admit`` | ``quarantine`` | ``reject``. ``--strict`` exits
nonzero on anything but ``admit``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

ADMISSION_SCHEMA = "receipt_admission.v1"


def _candidate_body(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    """The claim-bearing body: v2 inner payload when present, else the doc."""
    inner = doc.get("payload")
    return inner if isinstance(inner, Mapping) else doc


def _honesty_errors(doc: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(reject_errors, quarantine_errors) over the sealed honesty stamps."""
    reject: list[str] = []
    quarantine: list[str] = []
    bodies = [doc]
    inner = doc.get("payload")
    if isinstance(inner, Mapping):
        bodies.append(inner)
    for body in bodies:
        if body.get("live_pnl_claim") is True:
            reject.append("live_pnl_claim_true")
        if body.get("research_only") is False:
            reject.append("research_only_false")
    claim = _candidate_body(doc)
    if claim.get("research_only") is not True:
        quarantine.append("research_only_missing")
    if claim.get("live_pnl_claim") is not False:
        quarantine.append("live_pnl_claim_missing")
    label = claim.get("data_label")
    if not (isinstance(label, str) and label.strip()):
        quarantine.append("data_label_missing")
    return reject, quarantine


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.samefile(src):  # candidate already inside the corpus dir
            return
        raise FileExistsError(f"shadow corpus name collision: {dst.name}")
    try:
        os.link(src, dst)
    except OSError:  # cross-filesystem corpus dirs still gate correctly
        shutil.copy2(src, dst)


def _shadow_corpus(corpus_dir: Path, extra: Path | None = None) -> Path:
    """Link the corpus (+ optional extra receipt) into a private dir."""
    shadow = Path(tempfile.mkdtemp(prefix="admit_corpus_"))
    for path in sorted(corpus_dir.glob("*.json")):
        if path.is_file():
            _link_or_copy(path, shadow / path.name)
    if extra is not None:
        _link_or_copy(extra, shadow / extra.name)
    return shadow


def _inconsistent_groups(lattice: Mapping[str, Any]) -> set[str]:
    """Fingerprints of the inconsistent claim groups in a lattice receipt."""
    out: set[str] = set()
    for group in lattice.get("groups") or []:
        if isinstance(group, Mapping) and group.get("verdict") == "inconsistent":
            out.add(str(group.get("fingerprint", "?")))
    return out


def _survivors(corpus: Mapping[str, Any]) -> set[str]:
    return {
        f"{c.get('source')}|{c.get('path')}"
        for c in corpus.get("surviving_claims") or []
        if isinstance(c, Mapping)
    }


def admission_check(
    candidate: Path | str,
    corpus_dir: Path | str = Path("receipts"),
    *,
    q: float = 0.05,
    known_inconsistent: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Gate a candidate receipt against an existing corpus.

    Fails closed: a missing candidate or unreadable corpus dir raises;
    per-check findings are recorded rather than swallowed.
    """
    candidate = Path(candidate)
    corpus_dir = Path(corpus_dir)
    if not candidate.is_file():
        raise ValueError(f"candidate receipt {candidate} does not exist")
    if not corpus_dir.is_dir():
        raise ValueError(f"corpus dir {corpus_dir} does not exist")

    checks: list[dict[str, Any]] = []
    candidate_sha = hash_bytes(candidate.read_bytes())

    # -- 1. seal ---------------------------------------------------------------
    from quant_fund.research.receipt_v2 import verify_receipt_file

    verdict_result = verify_receipt_file(candidate)
    seal_ok = bool(verdict_result["valid"])
    checks.append(
        {
            "name": "seal",
            "ok": seal_ok,
            "errors": list(verdict_result["errors"]),
        }
    )

    try:
        doc = json.loads(candidate.read_text())
        if not isinstance(doc, Mapping):
            raise ValueError("receipt root is not an object")
    except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
        doc = {}
        seal_ok = False
        checks.append({"name": "parse", "ok": False, "errors": [str(exc)]})

    # -- 2. honesty stamps ------------------------------------------------------
    reject_errors: list[str] = []
    quarantine_errors: list[str] = []
    if doc:
        reject_errors, quarantine_errors = _honesty_errors(doc)
    checks.append(
        {
            "name": "honesty",
            "ok": not reject_errors and not quarantine_errors,
            "reject_errors": reject_errors,
            "quarantine_errors": quarantine_errors,
        }
    )

    # -- 3. lattice delta -------------------------------------------------------
    from quant_fund.research.receipt_lattice import receipt_lattice

    before_shadow = _shadow_corpus(corpus_dir)
    after_shadow = _shadow_corpus(corpus_dir, extra=candidate)
    pins = dict(known_inconsistent or {})
    before = receipt_lattice(before_shadow, known_inconsistent=pins)
    after = receipt_lattice(after_shadow, known_inconsistent=pins)
    new_inconsistent = _inconsistent_groups(after) - _inconsistent_groups(before)
    checks.append(
        {
            "name": "lattice",
            "ok": not new_inconsistent,
            "new_inconsistent_groups": len(new_inconsistent),
            "groups_after": after.get("n_claim_groups"),
        }
    )

    # -- 4. corpus FDR delta (evidence — never gate-driving on its own) ---------
    from quant_fund.research.corpus_inference import corpus_audit

    corpus_before = corpus_audit(before_shadow, q=q)
    corpus_after = corpus_audit(after_shadow, q=q)
    added = sorted(_survivors(corpus_after) - _survivors(corpus_before))
    removed = sorted(_survivors(corpus_before) - _survivors(corpus_after))
    checks.append(
        {
            "name": "corpus",
            "ok": True,
            "survivors_added": len(added),
            "survivors_removed": len(removed),
        }
    )

    corpus_files = sorted(p for p in corpus_dir.glob("*.json") if p.is_file())
    inputs_sha256 = hash_bytes(
        json.dumps(
            {
                "candidate": candidate.name,
                "candidate_sha256": candidate_sha,
                "corpus": {p.name: hash_bytes(p.read_bytes()) for p in corpus_files},
                "q": q,
            },
            sort_keys=True,
        ).encode()
    )

    if not seal_ok or reject_errors:
        verdict = "reject"
    elif quarantine_errors or new_inconsistent:
        verdict = "quarantine"
    else:
        verdict = "admit"

    return {
        "kind": ADMISSION_SCHEMA,
        "schema": ADMISSION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": False,
        "data_label": "CORPUS",
        "inputs_sha256": inputs_sha256,
        "params": {"q": q, "corpus_dir": str(corpus_dir), "known_inconsistent": sorted(pins)},
        "candidate": candidate.name,
        "candidate_sha256": candidate_sha,
        "n_corpus_receipts": len(corpus_files),
        "checks": checks,
        "lattice_new_inconsistent": sorted(new_inconsistent),
        "survivors_added": added,
        "survivors_removed": removed,
        "verdict": verdict,
    }


def admission_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Re-derive the accounting of a ``receipt_admission.v1`` body."""
    errors: list[str] = []
    if payload.get("kind") != ADMISSION_SCHEMA and payload.get("schema") != ADMISSION_SCHEMA:
        return ["schema"]
    if payload.get("verdict") not in ("admit", "quarantine", "reject"):
        errors.append("verdict")
    for key in ("inputs_sha256", "candidate_sha256"):
        value = payload.get(key)
        if not (isinstance(value, str) and len(value) == 64):
            errors.append(key)
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks_missing")
    else:
        names = [c.get("name") for c in checks if isinstance(c, Mapping)]
        for required in ("seal", "honesty", "lattice", "corpus"):
            if required not in names:
                errors.append(f"check_{required}_missing")
    n_corpus = payload.get("n_corpus_receipts")
    if not isinstance(n_corpus, int) or n_corpus < 0:
        errors.append("n_corpus_receipts")
    # verdict↔checks coherence: reject requires a failed seal or honesty reject;
    # quarantine requires some quarantine-class finding; admit requires all ok.
    if isinstance(checks, list) and errors == []:
        seal_check: Mapping[str, Any] = next((c for c in checks if c.get("name") == "seal"), {})
        honesty: Mapping[str, Any] = next((c for c in checks if c.get("name") == "honesty"), {})
        lattice: Mapping[str, Any] = next((c for c in checks if c.get("name") == "lattice"), {})
        verdict = payload["verdict"]
        rejectable = not seal_check.get("ok", True) or bool(honesty.get("reject_errors"))
        quarantinable = bool(honesty.get("quarantine_errors")) or not lattice.get("ok", True)
        if verdict == "reject" and not rejectable:
            errors.append("verdict_reject_without_cause")
        elif verdict == "quarantine" and (rejectable or not quarantinable):
            errors.append("verdict_quarantine_mismatch")
        elif verdict == "admit" and (rejectable or quarantinable):
            errors.append("verdict_admit_with_findings")
    return errors


def write_admission_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a receipt_admission receipt and write ``receipt_admission_<hash>.json``.

    Filename digest = canonical ``receipt_sha256``. Atomic, fail-closed on a
    malformed receipt. ``receipt_version=2`` wraps the body in the unified
    ``receipt.v2`` envelope.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.atomicio import atomic_write_text
    from quant_fund.utils.hashing import canonical_json_bytes

    if (
        receipt.get("kind") != ADMISSION_SCHEMA
        or receipt.get("schema") != ADMISSION_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("inputs_sha256"), str)
        or not isinstance(receipt.get("params"), Mapping)
    ):
        raise ValueError("receipt_admission receipt violates its contract")
    errors = admission_contract_errors(receipt)
    if errors:
        raise ValueError(f"receipt_admission receipt violates its contract: {errors}")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass" if receipt.get("verdict") == "admit" else "fail",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"receipt_admission_{digest[:16]}.json"
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
