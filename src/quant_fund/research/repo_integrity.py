"""Repo integrity capstone: one sealed attestation over the whole evidence
integrity substrate.

The pieces each gate one surface — ``crown_jewels`` byte-pins the gate-defining
files, ``corpus_epoch`` chains the evidence corpora, ``check_epoch_chain``
verifies them. ``verify_repo`` composes them into a single verdict and seals it
as a ``repo_integrity.v1`` receipt: a point-in-time proof that the repo's
evidence corpus and the config that gates it were all intact *at the same
instant*.

The committed attestation is itself a ``quality/`` chain member, so a stale
attestation can't be silently edited — the next epoch stamp records the drift.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import (
    check_epoch_chain,
    epoch_heads_key,
    load_heads_pin,
)
from quant_fund.research.crown_jewels import (
    DEFAULT_PIN_PATH as JEWELS_PIN,
)
from quant_fund.research.crown_jewels import (
    crown_jewels_errors,
)
from quant_fund.research.gate_signatures import verify_pin_signatures
from quant_fund.research.integrity_checkpoint import verify_checkpoint
from quant_fund.research.integrity_witness import verify_witnesses
from quant_fund.research.timestamp_anchor import verify_timestamps
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import hash_bytes

REPO_INTEGRITY_SCHEMA = "repo_integrity.v1"

# (corpus_dir, member glob, require_stamped, allow_member_updates,
#  coverage_exempt globs) — the same policy the evidence-audit Makefile
# target enforces; keep in sync.
CORPORA: tuple[tuple[str, str, bool, bool, tuple[str, ...]], ...] = (
    # require_stamped=True everywhere: an unstamped member is a silent
    # injection vector — the fuzz drill demonstrated arrivals passed as
    # notes. Arrivals are errors until the epoch restamp commits them.
    ("receipts", "*.json", True, False, ("legacy-unsealed/README.md",)),
    ("verifier", "*.md", True, False, ()),
    # quality's non-.json residents are jewel-pinned keys/certs or
    # self-authenticating TSA anchors — pinned elsewhere, exempt here.
    (
        "quality",
        "*.json",
        True,
        True,
        ("*.pub", "*.pem", "*.crt", "*.tsr", "*.txt", "timestamps/*"),
    ),
    # Closed world: a `.yaml` workflow beside `*.yml` members would run on
    # GitHub while dodging the epoch chain — uncovered is an error.
    (".github/workflows", "*.yml", True, True, ("README.md",)),
    # Declared experiment inputs — a post-hoc config edit silently rewrites
    # what a sealed bench measured; mutable corpus, stamped arrivals only.
    ("configs", "*", True, True, ()),
    # Committed claim artifacts (champion selects, dev grids) — regenerated
    # between runs, so mutable; every committed member must be stamped.
    ("artifacts", "*", True, True, ("*.gitkeep",)),
    # Committed fleet evidence (the SOTA input stream). Live dir — arrivals
    # land between stamps (unstamped = informational, not error), but a
    # stamped member is immutable: .npz mutation after stamping is tamper.
    (".dsh-24x7", "*", False, False, ()),
    # Committed dataset manifests + validation outputs — inputs the
    # data_manifest receipts pin; write-once dated dirs, stamped arrivals.
    ("data/metadata", "*", True, False, ()),
)


def verify_repo(
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
    evidence_only: bool = False,
) -> dict[str, Any]:
    """Run every integrity gate against the live tree.

    Returns ``{"gates": {name: {"ok": bool, "errors": [...]}}, "ok": bool}``.
    ``ok`` is true iff every gate reports zero errors. ``heads_pin`` is the
    committed epoch-heads pin required by every corpus chain.

    ``evidence_only`` verifies an evidence bundle — a tree carrying only the
    evidence dirs (receipts/, quality/, verifier/, configs/,
    .github/workflows/) with no source checkout: the crown-jewels byte pins
    cover ``src/`` and can't be checked, so that gate reports a skipped
    marker instead of a pass. The attestation records the mode so a partial
    verdict can never masquerade as a full-tree one.
    """
    root = Path(root)
    pin_path = root / heads_pin if not Path(heads_pin).is_absolute() else Path(heads_pin)
    # Committed removal acknowledgments — moving a stamped member (e.g. to
    # legacy-unsealed/) or a transient corpus blip is recorded history, not
    # an error, when name->sha256 is declared here. The file is itself a
    # corpus member: silently editing it still trips the epoch gate.
    allowed_removals: dict[str, str] = {}
    ar_path = root / "quality/epoch_allowed_removals.json"
    if ar_path.is_file():
        try:
            raw_ar = json.loads(ar_path.read_text())
            if isinstance(raw_ar, dict):
                allowed_removals = {str(k): str(v) for k, v in raw_ar.items() if isinstance(v, str)}
        except (OSError, json.JSONDecodeError):
            allowed_removals = {}
    gates: dict[str, dict[str, Any]] = {}

    if evidence_only:
        gates["crown_jewels"] = {
            "ok": True,
            "errors": [],
            "skipped": "evidence_only",
        }
    else:
        jewel_errs = crown_jewels_errors(root, pin_path=root / JEWELS_PIN)
        gates["crown_jewels"] = {"ok": not jewel_errs, "errors": jewel_errs}

    sig = verify_pin_signatures(root)
    gates["pin_signatures"] = {
        "ok": bool(sig["ok"]),
        "signed": bool(sig["signed"]),
        "errors": sig["errors"],
    }

    ts = verify_timestamps(root)
    gates["timestamp_anchors"] = {
        "ok": bool(ts["ok"]),
        "anchored": bool(ts["anchored"]),
        "fresh": ts.get("fresh", {}),
        "errors": ts["errors"],
    }

    cp = verify_checkpoint(root)
    gates["checkpoint"] = {
        "ok": bool(cp["ok"]),
        "signed": bool(cp["signed"]),
        "anchored": bool(cp.get("anchored", False)),
        "current": bool(cp.get("current", False)),
        "revision_ancestor": cp.get("revision_ancestor"),
        "errors": cp["errors"],
    }

    wit = verify_witnesses(root)
    gates["witness"] = {
        "ok": bool(wit["ok"]),
        "witnessed": bool(wit["witnessed"]),
        "errors": wit["errors"],
    }

    from quant_fund.research.checkpoint_chain import checkpoint_spine

    spine = checkpoint_spine(root)
    gates["spine"] = {
        "ok": bool(spine["ok"]),
        "signed": bool(spine.get("signed", False)),
        "spine_length": spine.get("spine_length", 0),
        "errors": spine["errors"],
    }

    from quant_fund.research.key_rotation import verify_rotations

    rot = verify_rotations(root)
    gates["key_rotation"] = {
        "ok": bool(rot["ok"]),
        "n_rotations": rot["n_rotations"],
        "errors": rot["errors"],
    }

    pin_present = pin_path.is_file()
    pin_parse_error: str | None = None
    if pin_present:
        try:
            heads = load_heads_pin(pin_path)
        except (OSError, ValueError, json.JSONDecodeError):
            heads = {}
            pin_parse_error = f"heads_pin_malformed:{heads_pin}"
    else:
        heads = {}
    for corpus_dir, pattern, require_stamped, allow_updates, exempt in CORPORA:
        cdir = root / corpus_dir
        if not cdir.is_dir():
            gates[f"epoch:{corpus_dir}"] = {
                "ok": False,
                "errors": [f"corpus_missing:{corpus_dir}"],
            }
            continue
        if not pin_present:
            expected = None
            pin_errors = [f"heads_pin_missing:{heads_pin}"]
        elif pin_parse_error is not None:
            expected = None
            pin_errors = [pin_parse_error]
        else:
            expected = heads.get(epoch_heads_key(corpus_dir, pattern))
            pin_errors = (
                [] if expected is not None else [f"heads_pin_entry_missing:{corpus_dir}/{pattern}"]
            )
        res = check_epoch_chain(
            cdir,
            pattern=pattern,
            expected_head=expected,
            require_stamped=require_stamped,
            allow_member_updates=allow_updates,
            allowed_removals=allowed_removals,
        )
        errs = pin_errors + list(res["errors"])
        # Coverage closure at the corpus policy level: every file in the
        # dir must be a pattern member, an epoch record, a declared
        # exemption, or on the corpus's allowlist — a non-matching file
        # (e.g. a `.yaml` workflow beside `*.yml` members, which GitHub
        # would still run) carries zero chain evidence and is invisible to
        # every check above. Multi-chain dirs are fine here: coverage is
        # per-corpus entry, and each dir has exactly one.
        from fnmatch import fnmatch

        from quant_fund.research.corpus_epoch import (
            EXEMPT_BASENAMES,
            EXEMPT_RELPATHS,
            _exempt_member,
        )

        for f in sorted(cdir.rglob("*")):
            if not f.is_file() or f.name.startswith("corpus_epoch_"):
                continue
            rel = f.relative_to(cdir).as_posix()
            if (
                f.name in EXEMPT_BASENAMES
                or rel in EXEMPT_RELPATHS
                or _exempt_member(cdir, rel)
                or fnmatch(rel, pattern)
                or any(fnmatch(rel, g) for g in exempt)
            ):
                continue
            errs.append(f"uncovered_member:{rel}")
        gates[f"epoch:{corpus_dir}"] = {"ok": not errs, "errors": errs}

    return {"gates": gates, "ok": all(g["ok"] for g in gates.values())}


# Gates every attestation must carry — the neutral/absent states are legal
# (unsigned pins, unanchored timestamps report ok=True) but the gate itself
# must be *run*: a receipt missing a gate hid the surface, not passed it.
_REQUIRED_GATES = (
    "crown_jewels",
    "pin_signatures",
    "timestamp_anchors",
    "spine",
    "key_rotation",
)


def repo_integrity_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Contract check for a ``repo_integrity.v1`` receipt body.

    Re-verifying the live gates would need the whole repo; the contract
    instead proves the attestation is internally coherent — that the claimed
    ``ok`` verdict is what the recorded gate verdicts actually imply, and
    that the digest pins can't disagree with the gates they summarize.
    """
    errors: list[str] = []
    if payload.get("schema") != REPO_INTEGRITY_SCHEMA:
        errors.append("schema_mismatch")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        errors.append("gates_missing")
        gates = {}
    for name in _REQUIRED_GATES:
        if name not in gates:
            errors.append(f"gate_missing:{name}")
    mode = payload.get("mode", "full")
    if mode not in ("full", "evidence_only"):
        errors.append(f"mode_unknown:{mode}")
        mode = "full"
    for name, gate in gates.items():
        if not isinstance(gate, Mapping):
            errors.append(f"gate_malformed:{name}")
            continue
        g_errors = gate.get("errors")
        if not isinstance(gate.get("ok"), bool) or not isinstance(g_errors, list):
            errors.append(f"gate_malformed:{name}")
            continue
        if not all(isinstance(e, str) for e in g_errors):
            errors.append(f"gate_errors_not_strings:{name}")
        # The gate's own verdict must match its listed errors — a receipt
        # can't claim ok while listing the errors it failed on.
        if bool(gate["ok"]) != (not g_errors):
            errors.append(f"gate_ok_incoherent:{name}")
        if "revision_ancestor" in gate and not (
            gate["revision_ancestor"] is None or isinstance(gate["revision_ancestor"], bool)
        ):
            errors.append(f"gate_field_malformed:{name}:revision_ancestor")
        if "skipped" in gate:
            # Only crown_jewels may be skipped, and only under evidence_only:
            # a full-tree attestation silently skipping a code gate would
            # downgrade a partial verdict into an implied full pass.
            if gate["skipped"] != "evidence_only" or name != "crown_jewels":
                errors.append(f"gate_skip_invalid:{name}")
            elif mode != "evidence_only":
                errors.append(f"gate_skipped_in_full_mode:{name}")
    if mode == "evidence_only":
        cj = gates.get("crown_jewels")
        if not isinstance(cj, Mapping) or cj.get("skipped") != "evidence_only":
            errors.append("evidence_only_skip_missing:crown_jewels")
    expected_ok = bool(gates) and all(
        bool(g.get("ok")) for g in gates.values() if isinstance(g, Mapping)
    )
    if payload.get("ok") != expected_ok:
        errors.append("ok_incoherent")
    pins = payload.get("pins")
    if isinstance(pins, Mapping):
        sig_gate = gates.get("pin_signatures", {})
        ts_gate = gates.get("timestamp_anchors", {})
        if (
            isinstance(sig_gate, Mapping)
            and "signed" in sig_gate
            and pins.get("gate_pins_signed") != sig_gate["signed"]
        ):
            errors.append("pins_sig_incoherent")
        if (
            isinstance(ts_gate, Mapping)
            and "anchored" in ts_gate
            and pins.get("timestamps_anchored") != ts_gate["anchored"]
        ):
            errors.append("pins_anchor_incoherent")
        for key in ("epoch_heads_sha256", "crown_jewels_sha256"):
            v = pins.get(key)
            if v is not None and not (isinstance(v, str) and len(v) == 64):
                errors.append(f"pin_malformed:{key}")
    else:
        errors.append("pins_missing")
    params = payload.get("params")
    if isinstance(params, Mapping):
        corpora = params.get("corpora")
        if isinstance(corpora, list):
            for c in corpora:
                if f"epoch:{c}" not in gates:
                    errors.append(f"corpus_gate_missing:{c}")
    if not isinstance(payload.get("data_label"), str):
        errors.append("data_label_missing")
    return errors


def repo_integrity_receipt(
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
    evidence_only: bool = False,
) -> dict[str, Any]:
    """Sealed ``repo_integrity.v1`` attestation over the live gate verdicts."""
    from quant_fund.research.receipt_v2 import seal_receipt

    root = Path(root)
    verdict = verify_repo(root, heads_pin=heads_pin, evidence_only=evidence_only)
    pin_path = root / heads_pin
    body: dict[str, Any] = {
        "kind": "repo_integrity",
        "schema": REPO_INTEGRITY_SCHEMA,
        "data_label": "SYNTHETIC",
        "ok": verdict["ok"],
        "mode": "evidence_only" if evidence_only else "full",
        "gates": {
            # signed/anchored are verdict state, not metadata — an unsigned-
            # tree attestation must be distinguishable from a signed one.
            name: {
                "ok": g["ok"],
                "errors": sorted(g["errors"]),
                **{
                    k: g[k]
                    for k in ("signed", "anchored", "fresh", "skipped", "revision_ancestor")
                    if k in g
                },
            }
            for name, g in verdict["gates"].items()
        },
        "pins": {
            "epoch_heads_sha256": hash_bytes(pin_path.read_bytes()) if pin_path.is_file() else None,
            "crown_jewels_sha256": (
                hash_bytes((root / JEWELS_PIN).read_bytes())
                if (root / JEWELS_PIN).is_file()
                else None
            ),
            "gate_pins_signed": verdict["gates"]["pin_signatures"]["signed"],
            "timestamps_anchored": verdict["gates"]["timestamp_anchors"]["anchored"],
        },
        "params": {"corpora": [c[0] for c in CORPORA]},
    }
    return seal_receipt(body)


def write_repo_integrity_receipt(
    out_path: Path | str,
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
    evidence_only: bool = False,
) -> Path:
    """Write the sealed attestation to ``out_path`` (atomic)."""
    receipt = repo_integrity_receipt(root, heads_pin=heads_pin, evidence_only=evidence_only)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return out
