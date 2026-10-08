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
_SIG_FILES = ("gate_pins.sig", "quality/gate_signing.pub", "quality/gate_quorum.json")
_CP_FILE = "quality/checkpoint.json"
# Both pubkeys verify_witness_file reads: ours (signature over the logged
# artifact) and Rekor's (SET + checkpoint note).
_WITNESS_KEY_FILES = ("quality/witness_signing.pub", "quality/rekor_pubkey.pem")
_WITNESS_DIR = Path("quality/witness")
# RFC 3161 timestamp anchors: manifest + .tsr tokens + pinned TSA certs —
# binding the pins to wall-clock. The dir is epoch-exempt by design (the .tsr
# imprints authenticate it), so custody embeds it wholesale.
_TIMESTAMPS_DIR = Path("quality/timestamps")


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
    ts_dir = root_p / _TIMESTAMPS_DIR
    if ts_dir.is_dir():
        for p in sorted(ts_dir.iterdir()):
            if p.is_file():
                embedded[f"{_TIMESTAMPS_DIR.as_posix()}/{p.name}"] = _b64(p.read_bytes())

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


def digest_custody(
    digest_hex: str,
    *,
    root: Path | str = ".",
    corpora: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, Any]:
    """Custody for a *referenced digest*, not a file: prove that a committed
    corpus member's bytes contain ``digest_hex``.

    A ``custody`` bundle proves a file exists in the spine; this variant
    proves the corpus *attested a digest* — e.g. a data manifest pinning a
    dataset sha256, or a receipt embedding an artifact digest. The carrier
    file's current bytes get the full eight-layer bundle plus a
    ``subject_digest`` layer verified by containment. The earliest-pinning
    carrier across corpora wins (earliest chain epoch = longest-attested
    reference).

    ``corpora`` narrows the search to ``(dir, glob)`` pairs; default scans
    every corpus with a chain under ``root``.
    """
    root_p = Path(root)
    digest_hex = digest_hex.strip().lower()
    if len(digest_hex) != 64 or any(c not in "0123456789abcdef" for c in digest_hex):
        raise ValueError("digest must be 64 lowercase hex chars")
    from quant_fund.research.corpus_epoch import member_digests
    from quant_fund.research.epoch_consistency import chain_index
    from quant_fund.research.repo_integrity import CORPORA

    pairs = corpora or tuple((str(c[0]), str(c[1])) for c in CORPORA)
    needles = digest_hex.encode()
    best: dict[str, Any] | None = None
    best_rank: tuple[int, str, str] | None = None
    for rel_dir, pattern in pairs:
        corpus = root_p / rel_dir
        if not corpus.is_dir():
            continue
        index = chain_index(corpus, pattern=pattern)
        order = _chain_order(index)
        if not order:
            continue
        # ``member_digests`` is the exact member set (recursive, with the
        # exemption rules applied). Epoch receipts are members too, but a
        # digest inside one's bytes is chain bookkeeping, not a content
        # claim — exclude them as carriers.
        for member, member_sha in member_digests(corpus, pattern=pattern).items():
            if Path(member).name.startswith("corpus_epoch_"):
                continue
            raw = (corpus / member).read_bytes()
            if needles not in raw:
                continue
            first_i = next(
                (
                    i
                    for i, name in enumerate(order)
                    if any(
                        str(m.get("name")) == member and str(m.get("sha256")) == member_sha
                        for m in index[name][1].get("members") or []
                        if isinstance(m, Mapping)
                    )
                ),
                None,
            )
            if first_i is None:
                continue  # unstamped or mutated since stamping
            rank = (first_i, rel_dir, member)
            if best_rank is None or rank < best_rank:
                bundle = custody_proof(member, corpus, pattern=pattern, root=root_p)
                bundle["subject_digest"] = digest_hex
                bundle["carrier"] = f"{rel_dir}/{member}"
                best, best_rank = bundle, rank
    if best is None:
        raise ValueError(
            f"digest {digest_hex[:16]}… is not referenced by any stamped corpus member"
        )
    return best


def member_timeline(
    member: str, corpus_dir: Path | str, *, pattern: str = "*.json"
) -> list[dict[str, str | int]]:
    """Ordered byte-lineage of ``member`` across the (dir, pattern) chain.

    Each entry is ``{epoch, sha256, epoch_index}`` for the epochs in which
    the member appears — the committed provenance of a mutable corpus
    member (e.g. ``quality/epoch_heads.json``, which legitimately changes).
    Immutable members yield a single repeated digest; an entry whose digest
    differs from its predecessor's is a committed byte change, an absent
    epoch is a committed removal (re-additions appear as a new run).
    """
    from quant_fund.research.epoch_consistency import chain_index

    index = chain_index(corpus_dir, pattern=pattern)
    order = _chain_order(index)
    if not order:
        raise ValueError(f"epoch chain under {corpus_dir} is forked or headless")
    timeline: list[dict[str, str | int]] = []
    for i, name in enumerate(order):
        members = {
            str(m["name"]): str(m["sha256"])
            for m in index[name][1].get("members") or []
            if isinstance(m, Mapping) and "name" in m and "sha256" in m
        }
        if member in members:
            timeline.append({"epoch": name, "epoch_index": i, "sha256": members[member]})
    return timeline


def _git_rev() -> str | None:
    try:
        from quant_fund.utils.reproducibility import git_revision

        return git_revision()
    except (ImportError, OSError, ValueError):  # provenance garnish, never gate
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
    subj = payload.get("subject_digest")
    if subj is not None and not (
        isinstance(subj, str) and len(subj) == 64 and all(c in "0123456789abcdef" for c in subj)
    ):
        errors.append("subject_digest")
    carrier = payload.get("carrier")
    if carrier is not None and not isinstance(carrier, str):
        errors.append("carrier")
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
    8. ``timestamps``  — ``verify_timestamps``: RFC 3161 tokens bind the
       pinned files to wall-clock (freshness = token commits to the
       embedded bytes).
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

    subj = bundle.get("subject_digest")
    if isinstance(subj, str):
        # Digest custody: the queried digest must sit inside the carrier's
        # verified bytes — the claim is containment, not just presence.
        contained = subj.encode() in member_bytes
        layers["subject_digest"] = {"ok": contained, "carrier": bundle.get("carrier")}
        if not contained:
            errors.append("subject_digest_absent")

    corpus_dir = str(bundle["corpus_dir"])
    hops = bundle["hops"]
    if not (isinstance(hops, list)):
        raise ValueError("isinstance(hops, list)")
    hop_docs: list[dict[str, Any]] = []
    hop_bytes: list[bytes] = []
    for i, hop in enumerate(hops):
        raw = _unb64(hop["bytes_b64"])
        if not (raw is not None):
            raise ValueError("raw is not None")
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
    if not (isinstance(inc, Mapping)):
        raise ValueError("isinstance(inc, Mapping)")
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
    if not (isinstance(embedded, Mapping)):
        raise ValueError("isinstance(embedded, Mapping)")
    for rel, b64v in embedded.items():
        raw = _unb64(b64v)
        if not (raw is not None):
            raise ValueError("raw is not None")
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
    if not (isinstance(witnesses, Mapping)):
        raise ValueError("isinstance(witnesses, Mapping)")
    if witnesses:
        from quant_fund.research.integrity_witness import verify_witness_file

        w_ok = False
        w_errors: list[str] = []
        for rel, b64v in witnesses.items():
            raw = _unb64(b64v)
            if not (raw is not None):
                raise ValueError("raw is not None")
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

    # RFC 3161 layer: the anchored targets (checkpoint/pins) are embedded, so
    # verify_timestamps gets a live synthetic root — ``fresh`` confirms each
    # token still commits to those bytes, and the openssl chain check pins the
    # TSA certs.
    ts_manifest = tmp / _TIMESTAMPS_DIR / "anchors.json"
    if ts_manifest.is_file():
        from quant_fund.research.timestamp_anchor import verify_timestamps

        ts = verify_timestamps(tmp)
        layers["timestamps"] = {
            "ok": bool(ts["ok"]),
            "anchored": ts.get("anchored"),
            "fresh": ts.get("fresh"),
        }
        if not ts["ok"]:
            errors.extend(f"timestamps:{e}" for e in ts["errors"])
    else:
        layers["timestamps"] = {"ok": True, "absent": True}

    return {"ok": not errors, "errors": errors, "layers": layers}
