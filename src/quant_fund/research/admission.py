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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.legacy_unsealed import (
    EMPTY_QUARANTINE,
    Quarantine,
    honesty_stamp_errors,
)
from quant_fund.utils.hashing import hash_bytes

ADMISSION_SCHEMA = "receipt_admission.v1"


def _honesty_errors(doc: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(reject_errors, quarantine_errors) over the sealed honesty stamps."""
    return honesty_stamp_errors(doc)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return  # already linked — e.g. the candidate lives inside the corpus
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
    tombstone_dir: Path | str | None = None,
    epoch_dir: Path | str | None = None,
    expected_head: Mapping[str, Any] | None = None,
    epoch_acks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Gate a candidate receipt against an existing corpus.

    ``epoch_dir`` is the durable corpus whose epoch chain the admission binds
    to (default ``corpus_dir``); ``admit_batch`` passes the real corpus while
    ``corpus_dir`` is a shadow, because an intermediate shadow is missing the
    batch's stamped members by construction. The chain is checked with the
    same heads pin and acknowledgment ledger as ``corpus-epoch --check``.

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
    except (OSError, ValueError, RecursionError) as exc:
        # Narrowed from `except Exception` (quality ratchet): the parse block
        # reads bytes then json-loads — OSError/ValueError are the only fault
        # modes; exotic errors propagate. Recorded, never swallowed.
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

    # -- 5. epoch-chain integrity ------------------------------------------------
    # Binding to the stamped epoch chain is defense in depth on top of the
    # member-map digest: if the corpus was tampered *after* its last stamp,
    # admitting into it must not claim a clean pass. Feature-detected so this
    # module works whether or not corpus_epoch has merged/stamped yet.
    chain_errors: list[str] = []
    hard_chain_errors: list[str] = []
    epoch_head: str | None = None
    epoch_root_value: str | None = None
    try:
        from quant_fund.research.corpus_epoch import check_epoch_chain
    except ImportError:
        checks.append({"name": "epoch_chain", "ok": True, "skipped": "corpus_epoch_unavailable"})
    else:
        chain_result = check_epoch_chain(
            Path(epoch_dir) if epoch_dir is not None else corpus_dir,
            acks=epoch_acks,
            expected_head=expected_head,
        )
        chain_errors = list(chain_result["errors"])
        # An unstamped corpus ("no_epoch_receipts") is benign — the gate
        # works fine before the first epoch stamp exists — unless a heads pin
        # says a chain exists. Only a *broken stamped chain* (fork,
        # member_removed/mutated, dishonest delta) is a quarantine-class
        # finding.
        hard_chain_errors = [
            e for e in chain_errors if e != "no_epoch_receipts" or expected_head is not None
        ]
        head_name = chain_result.get("head")
        epoch_head = head_name if isinstance(head_name, str) else None
        root_value = chain_result.get("head_epoch_root")
        epoch_root_value = root_value if isinstance(root_value, str) else None
        checks.append(
            {
                "name": "epoch_chain",
                "ok": not hard_chain_errors,
                "errors": chain_errors,
                "acknowledged": list(chain_result.get("acknowledged", [])),
                "unstamped": list(chain_result["unstamped"]),
            }
        )

    # -- 6. retraction -----------------------------------------------------------
    # An append-only corpus can't delete a bad receipt — it retracts it via a
    # sealed tombstone. Admission is the re-entry surface: the exact retracted
    # bytes must never be re-admitted, and the retracted name slot / claim
    # paths must not quietly re-fill. Feature-detected like epoch_chain.
    retract_reject: list[str] = []
    retract_findings: list[str] = []
    try:
        from quant_fund.research.receipt_tombstone import load_tombstones
    except ImportError:
        checks.append({"name": "tombstone", "ok": True, "skipped": "tombstone_unavailable"})
    else:
        # Resolve tombstones against the durable corpus, not the (possibly
        # shadowed) admission view — a retracted target must stay pinned to
        # real bytes even when admit_batch shadows it out mid-diff.
        tombs = load_tombstones(Path(tombstone_dir) if tombstone_dir is not None else corpus_dir)
        entry = tombs["active"].get(candidate.name)
        if entry is not None:
            if entry.get("target_sha256") == candidate_sha:
                retract_reject.append("retracted_bytes")
            elif entry.get("scope") == "all":
                retract_findings.append("retracted_slot")
            else:
                # Partial scope: the retracted claim paths must not re-enter.
                from quant_fund.research.corpus_inference import harvest_findings

                scoped = set(entry["scope"]) if isinstance(entry["scope"], list) else set()
                try:
                    cand_doc = json.loads(candidate.read_text())
                except (OSError, ValueError):  # parse already recorded above
                    cand_doc = {}
                paths = (
                    {f["path"] for f in harvest_findings(cand_doc, candidate.name)}
                    if isinstance(cand_doc, Mapping)
                    else set()
                )
                if paths & scoped:
                    retract_findings.append("retracted_scope_overlap")
        checks.append(
            {
                "name": "tombstone",
                "ok": not retract_reject and not retract_findings,
                "reject_errors": retract_reject,
                "findings": retract_findings,
            }
        )

    if not seal_ok or reject_errors or retract_reject:
        verdict = "reject"
    elif quarantine_errors or new_inconsistent or hard_chain_errors or retract_findings:
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
        "corpus_epoch_receipt": epoch_head,
        "corpus_epoch_root": epoch_root_value,
        "lattice_new_inconsistent": sorted(new_inconsistent),
        "survivors_added": added,
        "survivors_removed": removed,
        "verdict": verdict,
    }


def _corpus_rel(cand: Path, corpus_dir: Path) -> str:
    try:
        return cand.resolve().relative_to(corpus_dir.resolve()).as_posix()
    except ValueError:
        return cand.name


def _partition_base_stamped(
    ordered: list[Path], corpus_dir: Path, base_stamped: Mapping[str, str]
) -> tuple[list[str], list[Path]]:
    """Split off candidates whose bytes the base's pinned head epoch already
    stamped — corpus members, not arrivals — from everything else."""
    stamped: list[str] = []
    fresh: list[Path] = []
    for cand in ordered:
        rel = _corpus_rel(cand, corpus_dir)
        if base_stamped.get(rel) == hash_bytes(cand.read_bytes()):
            stamped.append(rel)
        else:
            fresh.append(cand)
    return stamped, fresh


def _partition_quarantined(
    ordered: list[Path], corpus_dir: Path, quarantine: Quarantine
) -> tuple[list[str], list[Path]]:
    """Split candidates into registry-matched quarantined records (retained,
    never admitted) and everything else (gated as usual)."""
    from quant_fund.research.legacy_unsealed import classify_receipt
    from quant_fund.research.receipt_v2 import verify_receipt_bytes

    retained: list[str] = []
    gated: list[Path] = []
    for cand in ordered:
        rel = _corpus_rel(cand, corpus_dir)
        raw = cand.read_bytes()
        ver = verify_receipt_bytes(raw, cand)
        status, _why = classify_receipt(
            rel,
            hash_bytes(raw),
            valid=bool(ver["valid"]),
            errors=list(ver["errors"]),
            quarantine=quarantine,
            raw=raw,
        )
        if status == "quarantined":
            retained.append(rel)
        else:
            gated.append(cand)
    return retained, gated


def _batch_shadow(corpus_dir: Path, ordered: list[Path], gated: list[Path]) -> Path:
    """Corpus minus the batch, plus the batch's retained quarantined records.

    Mirrors every verifiable receipt (recursively, minus quarantined
    subtrees — the lattice's own membership). Retained quarantined records
    are part of the merged corpus every gated candidate lands in (the
    lattice reads them), so they enter up front."""
    from quant_fund.utils.receipt import is_quarantined, verified_corpus_files

    changed_names = {c.name for c in ordered}
    shadow = Path(tempfile.mkdtemp(prefix="admit_batch_base_"))
    for path in verified_corpus_files(corpus_dir):
        rel_path = path.relative_to(corpus_dir)
        if len(rel_path.parts) == 1 and path.name in changed_names:
            continue
        (shadow / rel_path).parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(path, shadow / rel_path)
    for cand in ordered:
        rel_path = Path(_corpus_rel(cand, corpus_dir))
        # Keep the relative path: a retained legacy-unsealed/ record stays in
        # the quarantine subtree the lattice skips, exactly as in the corpus.
        if cand not in gated and not is_quarantined(corpus_dir / rel_path, corpus_dir):
            (shadow / rel_path).parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(cand, shadow / rel_path)
    return shadow


def admit_batch(
    candidates: Iterable[Path | str],
    corpus_dir: Path | str = Path("receipts"),
    *,
    q: float = 0.05,
    known_inconsistent: Mapping[str, str] | None = None,
    quarantine: Quarantine = EMPTY_QUARANTINE,
    expected_head: Mapping[str, Any] | None = None,
    epoch_acks: list[dict[str, Any]] | None = None,
    base_stamped: Mapping[str, str] | None = None,
    base_epochs: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Gate a set of incoming receipts the way a merge actually lands them.

    ``admission_check`` on an already-committed receipt has a vacuous lattice
    check: the candidate is already inside the corpus, so before == after and
    no delta is observable. For a diff of N changed receipts the honest gate
    is sequential — each candidate is checked against

        corpus − {all changed names} ∪ {changed files processed so far}

    which is exactly the intermediate state each file creates, so
    intra-diff contradictions are attributed to the file that introduces
    them and no commit sneaks a contradiction in inside a batch.

    The shadow mirrors every verifiable corpus receipt (recursively, minus
    quarantined subtrees — the lattice's own membership). The epoch-chain
    check is NOT run on the shadow: it binds to the real ``corpus_dir`` (the
    merged state) with ``expected_head``/``epoch_acks``, because a shadow
    lacks the batch's stamped members by construction.

    ``base_stamped`` is the member map of the head epoch the *base* branch's
    heads pin authenticates (``corpus_epoch.pinned_head_members``). A
    candidate whose bytes equal what the base already stamped (e.g. a
    receipt restored to its pinned bytes) is not new evidence entering the
    corpus: it is listed under ``already_stamped`` and not re-admitted. Only
    the base pin can grant this — the candidate's own branch cannot.

    ``base_epochs`` (``corpus_epoch.chain_ancestry`` over the base pin)
    closes the stamp-and-excuse path: every acknowledgment in ``epoch_acks``
    — removal or mutation — must name an epoch the base chain already
    contained, so a single change can never both record a removal or rewrite
    of an append-only receipt and acknowledge it. A violation rejects the
    batch (``epoch_ack_postdates_base``); the same rule runs repo-wide, on
    every PR, as ``repo_integrity.ack_ledger_gate``.

    When no candidate is gated (only restorations / retained quarantine),
    the real corpus's chain is still checked once at batch level
    (``chain_errors``, quarantine-class like the per-candidate check).

    A candidate that matches its ``quarantine`` registry entry exactly
    (path, bytes, error set) is retained record, not evidence asking to be
    admitted: it is listed under ``retained_quarantined`` and never receives
    an ``admit`` verdict. Any other non-verifying candidate is gated, and
    rejected, as usual.

    Fails closed: a missing candidate, a candidate inside the corpus under a
    *different* name, or an unreadable corpus dir raises.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise ValueError(f"corpus dir {corpus_dir} does not exist")
    ordered = sorted((Path(c) for c in candidates), key=lambda p: p.name)
    for cand in ordered:
        if not cand.is_file():
            raise ValueError(f"candidate receipt {cand} does not exist")

    stamped, fresh = _partition_base_stamped(ordered, corpus_dir, base_stamped or {})
    retained, gated = _partition_quarantined(fresh, corpus_dir, quarantine)
    shadow = _batch_shadow(corpus_dir, ordered, gated)

    results: list[dict[str, Any]] = []
    for cand in gated:
        result = admission_check(
            cand,
            shadow,
            q=q,
            known_inconsistent=known_inconsistent,
            tombstone_dir=corpus_dir,
            epoch_dir=corpus_dir,
            expected_head=expected_head,
            epoch_acks=epoch_acks,
        )
        results.append(result)
        # Post-merge coexistence: a merged diff lands all of its files
        # together, so each later candidate must clear a corpus that already
        # contains the earlier ones — whatever verdict they drew.
        _link_or_copy(cand, shadow / cand.name)

    ack_errors = _batch_ack_errors(epoch_acks or [], base_epochs)
    # Per-candidate checks bind the chain only when something is gated; a
    # batch of restorations alone must still not land on a broken chain.
    chain_errors = [] if gated else _batch_chain_errors(corpus_dir, expected_head, epoch_acks)
    verdicts = (
        [r["verdict"] for r in results]
        + ["reject"] * bool(ack_errors)
        + ["quarantine"] * bool(chain_errors)
    )
    verdict = next((v for v in ("reject", "quarantine") if v in verdicts), "admit")
    return {
        "verdict": verdict,
        "n_candidates": len(ordered),
        "n_gated": len(gated),
        "results": results,
        "retained_quarantined": retained,
        "already_stamped": stamped,
        "ack_errors": ack_errors,
        "chain_errors": chain_errors,
        "failures": [r["candidate"] for r in results if r["verdict"] != "admit"],
    }


def _batch_ack_errors(
    epoch_acks: list[dict[str, Any]], base_epochs: frozenset[str] | None
) -> list[str]:
    """``corpus_epoch.ack_ledger_base_errors`` scoped to this corpus's acks
    (admission's corpus is append-only); no base chain, no rule."""
    if base_epochs is None:
        return []
    from quant_fund.research.corpus_epoch import ack_ledger_base_errors

    keys = frozenset(str(a["corpus"]) for a in epoch_acks)
    return ack_ledger_base_errors(epoch_acks, [], dict.fromkeys(keys, base_epochs), keys)


def _batch_chain_errors(
    corpus_dir: Path,
    expected_head: Mapping[str, Any] | None,
    epoch_acks: list[dict[str, Any]] | None,
) -> list[str]:
    """Hard epoch-chain errors of the real corpus (``admission_check``'s rule:
    an unstamped corpus is benign unless a heads pin says a chain exists)."""
    from quant_fund.research.corpus_epoch import check_epoch_chain

    errors = check_epoch_chain(corpus_dir, acks=epoch_acks, expected_head=expected_head)["errors"]
    return [e for e in errors if e != "no_epoch_receipts" or expected_head is not None]


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
    epoch_root = payload.get("corpus_epoch_root")
    if epoch_root is not None and not (isinstance(epoch_root, str) and len(epoch_root) == 64):
        errors.append("corpus_epoch_root")
    epoch_receipt = payload.get("corpus_epoch_receipt")
    if epoch_receipt is not None and not (
        isinstance(epoch_receipt, str) and epoch_receipt.startswith("corpus_epoch_")
    ):
        errors.append("corpus_epoch_receipt")
    if isinstance(checks, list) and errors == []:
        seal_check: Mapping[str, Any] = next((c for c in checks if c.get("name") == "seal"), {})
        honesty: Mapping[str, Any] = next((c for c in checks if c.get("name") == "honesty"), {})
        lattice: Mapping[str, Any] = next((c for c in checks if c.get("name") == "lattice"), {})
        epoch_chain: Mapping[str, Any] = next(
            (c for c in checks if c.get("name") == "epoch_chain"), {}
        )
        tombstone: Mapping[str, Any] = next((c for c in checks if c.get("name") == "tombstone"), {})
        verdict = payload["verdict"]
        rejectable = (
            not seal_check.get("ok", True)
            or bool(honesty.get("reject_errors"))
            or bool(tombstone.get("reject_errors"))
        )
        quarantinable = (
            bool(honesty.get("quarantine_errors"))
            or not lattice.get("ok", True)
            or not epoch_chain.get("ok", True)
            or bool(tombstone.get("findings"))
        )
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
