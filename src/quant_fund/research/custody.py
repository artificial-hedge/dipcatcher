"""Chain-of-custody bundle: one artifact proving *when* a file entered the
signed evidence spine, verifiable with no repo access.

The layers each exist separately — ``corpus-proof`` (member→epoch inclusion),
``corpus-consistency`` (epoch→head extension), ``sign-pins`` (signed head),
``checkpoint`` (signed pin state), ``witness-checkpoint`` (Rekor inclusion of
the checkpoint). ``dipcatcher custody`` composes them into a single sealed
``custody_proof.v1`` bundle, and ``--check`` re-verifies every layer against
the *embedded* bytes — the auditor needs nothing but the bundle and the file
whose provenance is being proven:

    custody <file> --corpus-dir receipts --out custody.json
    custody --check custody.json --member-file <file>

The bundle embeds raw bytes for each chain hop (the inclusion epoch through
the current head), the signed pin files, the checkpoint, and the Rekor
witness proof. Verification recomputes every digest and signature; nothing
is trusted on claim alone.
"""

from __future__ import annotations

import base64
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

CUSTODY_SCHEMA = "custody_proof.v1"

# Files embedded verbatim — the exact inventory the pin/checkpoint/witness
# verifiers read under a repo root.
_PIN_FILES = ("quality/epoch_heads.json", "quality/crown_jewels.json")
_SIG_FILES = ("gate_pins.sig", "quality/gate_signing.pub")
_CP_FILE = "quality/checkpoint.json"
# Both pubkeys verify_witness_file reads: ours (signature over the logged
# artifact) and Rekor's (SET + checkpoint note).
_WITNESS_KEY_FILES = ("quality/witness_signing.pub", "quality/rekor_pubkey.pem")
_WITNESS_DIR = Path("quality/witness")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _unb64(s: Any) -> bytes | None:
    if not isinstance(s, str):
        return None
    try:
        return base64.b64decode(s)
    except ValueError:
        return None


def _chain_order(index: Mapping[str, Any]) -> list[str]:
    """Epoch receipt names in genesis→head order (prev-link walk)."""
    by_prev: dict[str, str] = {}
    genesis: list[str] = []
    for name, (_, payload, _) in index.items():
        prev = payload.get("prev_epoch_receipt")
        if prev is None:
            genesis.append(name)
        else:
            by_prev[prev] = name
    if len(genesis) != 1:
        return []  # forked or headless — no linear order to report
    order = [genesis[0]]
    while order[-1] in by_prev:
        order.append(by_prev[order[-1]])
    if len(order) != len(index):
        return []  # unreachable nodes — forked
    return order


def custody_proof(
    member: str,
    corpus_dir: Path | str,
    *,
    pattern: str = "*.json",
    root: Path | str = ".",
) -> dict[str, Any]:
    """Compose the full custody bundle for corpus member ``member``.

    ``member`` is the corpus-relative name (e.g. a receipt filename). The
    epoch bound is the *earliest* chained epoch whose member map pins the
    member's current bytes — proof of first committed state, not merely
    presence at head.
    """
    from quant_fund.research.epoch_consistency import chain_index
    from quant_fund.research.epoch_merkle import member_proof

    root_p = Path(root)
    corpus_rel = Path(corpus_dir)
    if corpus_rel.is_absolute():
        corpus_rel = corpus_rel.resolve().relative_to(root_p.resolve())
    corpus = root_p / corpus_rel
    member_path = corpus / member
    if not member_path.is_file():
        raise ValueError(f"member {member_path} does not exist")
    member_bytes = member_path.read_bytes()
    member_sha = hash_bytes(member_bytes)

    index = chain_index(corpus, pattern=pattern)
    if not index:
        raise ValueError(f"no epoch chain under {corpus} for {pattern}")
    order = _chain_order(index)
    if not order:
        raise ValueError(f"epoch chain under {corpus} is forked or headless")

    def _members(payload: Mapping[str, Any]) -> dict[str, str]:
        return {
            str(m["name"]): str(m["sha256"])
            for m in payload.get("members") or []
            if isinstance(m, Mapping) and "name" in m and "sha256" in m
        }

    first: str | None = None
    for name in order:
        if _members(index[name][1]).get(member) == member_sha:
            first = name
            break
    if first is None:
        raise ValueError(
            f"{member} is not stamped in any epoch for {corpus_dir}/{pattern} "
            "(unstamped or the member mutated since stamping)"
        )

    inclusion = member_proof(corpus, member, epoch_receipt=first)

    # Embed the hop files raw: verifier recomputes file digests, so parsed
    # JSON alone would not do. Hops run from the inclusion epoch to head.
    hops: list[dict[str, str]] = []
    seen = False
    for name in order:
        if name == first:
            seen = True
        if seen:
            path, _payload, digest = index[name]
            hops.append({"name": name, "sha256": digest, "bytes_b64": _b64(path.read_bytes())})

    embedded: dict[str, str] = {}
    for rel in (*_PIN_FILES, *_SIG_FILES, _CP_FILE, *_WITNESS_KEY_FILES):
        p = root_p / rel
        if p.is_file():
            embedded[rel] = _b64(p.read_bytes())
    witness_dir = root_p / _WITNESS_DIR
    witnesses: dict[str, str] = {}
    if witness_dir.is_dir():
        for p in sorted(witness_dir.glob(f"{Path(_CP_FILE).name}_*.json")):
            witnesses[f"{_WITNESS_DIR.as_posix()}/{p.name}"] = _b64(p.read_bytes())

    return {
        "kind": CUSTODY_SCHEMA,
        "schema": CUSTODY_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "member": member,
        "member_sha256": member_sha,
        "corpus_dir": corpus_rel.as_posix(),
        "pattern": pattern,
        "first_epoch": first,
        "chain_head": hops[-1]["name"],
        "n_hops": len(hops),
        "inclusion": inclusion,
        "hops": hops,
        "embedded_files": embedded,
        "witness_proofs": witnesses,
        "generated_at_commit": _git_rev(),
    }


def _git_rev() -> str | None:
    try:
        from quant_fund.research.receipt_v2 import git_revision

        return git_revision()
    except Exception:  # noqa: BLE001 — provenance garnish, never gate
        return None


def custody_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``custody_proof.v1`` structural coherence; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != CUSTODY_SCHEMA:
        errors.append("kind")
    if payload.get("schema") != CUSTODY_SCHEMA:
        errors.append("schema")
    sha = payload.get("member_sha256")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        errors.append("member_sha256")
    for key in ("member", "corpus_dir", "pattern", "first_epoch", "chain_head"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            errors.append(key)
    hops = payload.get("hops")
    if not isinstance(hops, list) or not hops:
        errors.append("hops_empty")
    elif isinstance(hops, list):
        for i, h in enumerate(hops):
            if (
                not isinstance(h, Mapping)
                or not isinstance(h.get("name"), str)
                or _unb64(h.get("bytes_b64")) is None
                or not isinstance(h.get("sha256"), str)
            ):
                errors.append(f"hop_malformed:{i}")
        if errors == []:
            if isinstance(hops, list) and hops[0].get("name") != payload.get("first_epoch"):
                errors.append("first_epoch_mismatch")
            if isinstance(hops, list) and hops[-1].get("name") != payload.get("chain_head"):
                errors.append("chain_head_mismatch")
    inclusion = payload.get("inclusion")
    if not isinstance(inclusion, Mapping):
        errors.append("inclusion_missing")
    embedded = payload.get("embedded_files")
    if not isinstance(embedded, Mapping):
        errors.append("embedded_missing")
    else:
        for rel, b64v in embedded.items():
            if not isinstance(rel, str) or _unb64(b64v) is None:
                errors.append("embedded_malformed")
    witnesses = payload.get("witness_proofs")
    if not isinstance(witnesses, Mapping):
        errors.append("witness_proofs_missing")
    else:
        for rel, b64v in witnesses.items():
            if not isinstance(rel, str) or _unb64(b64v) is None:
                errors.append("witness_malformed")
    return errors


def verify_custody_bundle(
    bundle: Mapping[str, Any],
    member_bytes: bytes,
) -> dict[str, Any]:
    """Zero-repo verification: re-derive every layer from embedded bytes.

    Layers, in order — a failure at any layer is an error string, never a
    silent pass:

    1. ``member``      — supplied file bytes hash to the pinned digest.
    2. ``inclusion``   — Merkle path recomputes the inclusion epoch's root.
    3. ``chain``       — embedded hop files: each digest re-derives, each
       successor's ``prev`` names its predecessor *and* pins the
       predecessor's file bytes in its member map.
    4. ``pins``        — the signed pin file binds the head hop digest.
    5. ``signature``   — ``verify_pin_signatures`` on the materialized tree.
    6. ``checkpoint``  — ``verify_checkpoint`` (signature + pinned digests
       current against the embedded pin files).
    7. ``witness``     — ``verify_witness_file`` on each embedded Rekor
       proof; at least one must be authentic for a witnessed bundle.
    """
    errors: list[str] = []
    contract = custody_contract_errors(bundle)
    if contract:
        return {"ok": False, "errors": [f"contract:{e}" for e in contract], "layers": {}}
    layers: dict[str, Any] = {}

    member_sha = str(bundle["member_sha256"])
    if hash_bytes(member_bytes) != member_sha:
        errors.append("member_bytes_mismatch")
    layers["member"] = {"ok": not errors}

    corpus_dir = str(bundle["corpus_dir"])
    hops = bundle["hops"]
    assert isinstance(hops, list)
    hop_docs: list[dict[str, Any]] = []
    hop_bytes: list[bytes] = []
    for i, hop in enumerate(hops):
        raw = _unb64(hop["bytes_b64"])
        assert raw is not None
        if hash_bytes(raw) != hop["sha256"]:
            errors.append(f"hop_digest_mismatch:{i}")
        try:
            doc = json.loads(raw)
        except ValueError:
            errors.append(f"hop_unparseable:{i}")
            doc = {}
        hop_docs.append(doc if isinstance(doc, dict) else {})
        hop_bytes.append(raw)

    # Inclusion: the Merkle path recomputes the root *and* that root equals
    # the Merkle root of the embedded epoch's member map (the epoch's own
    # epoch_root_sha256 is a different digest — canonical member map, not
    # the RFC 6962 tree — so the path must land on merkle_root, not it).
    inc = bundle["inclusion"]
    assert isinstance(inc, Mapping)
    from quant_fund.research.epoch_merkle import merkle_root, verify_inclusion

    epoch_doc = hop_docs[0]
    members_map = {
        str(m["name"]): str(m["sha256"])
        for m in epoch_doc.get("members") or []
        if isinstance(m, Mapping) and "name" in m and "sha256" in m
    }
    inc_ok = (
        isinstance(inc.get("merkle_root"), str)
        and members_map.get(str(inc.get("member"))) == inc.get("member_sha256") == member_sha
        and verify_inclusion(
            str(inc["member"]), str(inc["member_sha256"]), inc, str(inc["merkle_root"])
        )
        and merkle_root(members_map) == inc["merkle_root"]
    )
    if not inc_ok:
        errors.append("inclusion_invalid")
    layers["inclusion"] = {"ok": inc_ok}

    # Chain extension: embedded successors link back and pin predecessor
    # bytes — the same binding verify_consistency enforces on a live dir.
    chain_ok = True
    for i in range(1, len(hops)):
        prev, cur = hops[i - 1], hop_docs[i]
        if cur.get("prev_epoch_receipt") != prev["name"]:
            errors.append(f"hop_link_broken:{i}")
            chain_ok = False
        cur_members = {
            str(m["name"]): str(m["sha256"])
            for m in cur.get("members") or []
            if isinstance(m, Mapping) and "name" in m and "sha256" in m
        }
        if cur_members.get(str(prev["name"])) != prev["sha256"]:
            errors.append(f"hop_member_pin_mismatch:{i}")
            chain_ok = False
    layers["chain"] = {"ok": chain_ok}

    # Materialize a synthetic root and hand off to the existing verifiers.
    tmp = Path(tempfile.mkdtemp(prefix="custody_verify_"))
    corpus_root = tmp / corpus_dir
    corpus_root.mkdir(parents=True, exist_ok=True)
    for hop, raw in zip(hops, hop_bytes, strict=True):
        (corpus_root / str(hop["name"])).write_bytes(raw)
    (corpus_root / str(bundle["member"])).write_bytes(member_bytes)

    embedded = bundle["embedded_files"]
    assert isinstance(embedded, Mapping)
    for rel, b64v in embedded.items():
        raw = _unb64(b64v)
        assert raw is not None
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)

    # The signed pin must bind the claimed head hop.
    try:
        heads_doc = json.loads((tmp / "quality/epoch_heads.json").read_text())
        pin_key = f"{corpus_dir}/{bundle['pattern']}"
        pinned = (heads_doc.get("heads") or {}).get(pin_key) or {}
        pin_ok = (
            pinned.get("receipt") == hops[-1]["name"] and pinned.get("sha256") == hops[-1]["sha256"]
        )
    except (OSError, ValueError):
        pin_ok = False
    if not pin_ok:
        errors.append("pin_head_mismatch")
    layers["pins"] = {"ok": pin_ok}

    if (tmp / "gate_pins.sig").is_file():
        from quant_fund.research.gate_signatures import verify_pin_signatures

        sig = verify_pin_signatures(tmp)
        layers["signature"] = {"ok": bool(sig["ok"]), "signed": sig["signed"]}
        if not sig["ok"] or not sig["signed"]:
            errors.append(f"signature:{sig['errors']}")
    else:
        layers["signature"] = {"ok": True, "signed": False}

    if (tmp / _CP_FILE).is_file():
        from quant_fund.research.integrity_checkpoint import verify_checkpoint

        cp = verify_checkpoint(tmp)
        layers["checkpoint"] = {"ok": bool(cp["ok"]), "current": cp.get("current")}
        if not cp["ok"]:
            errors.append(f"checkpoint:{cp['errors']}")
    else:
        layers["checkpoint"] = {"ok": True, "absent": True}

    witnesses = bundle["witness_proofs"]
    assert isinstance(witnesses, Mapping)
    if witnesses:
        from quant_fund.research.integrity_witness import verify_witness_file

        w_ok = False
        w_errors: list[str] = []
        for rel, b64v in witnesses.items():
            raw = _unb64(b64v)
            assert raw is not None
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            res = verify_witness_file(tmp, rel)
            if res["ok"]:
                w_ok = True
            else:
                w_errors.extend(f"{rel}:{e}" for e in res["errors"])
        layers["witness"] = {"ok": w_ok}
        if not w_ok:
            errors.extend(w_errors or ["witness_invalid"])
    else:
        layers["witness"] = {"ok": True, "absent": True}

    return {"ok": not errors, "errors": errors, "layers": layers}
