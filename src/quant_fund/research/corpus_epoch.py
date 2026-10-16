"""Corpus epochs: a hash-chained integrity root over the evidence store.

``verify-receipt`` seals individual receipts; ``receipt_lattice`` cross-checks
claims inside them. Neither watches the *container*: today a receipt can be
deleted or rewritten under ``receipts/`` and nothing in the receipt format
notices — only git history does, and git is outside the sealed-evidence model.

A ``corpus_epoch.v1`` receipt stamps the corpus's full membership:

- ``members`` — every ``*.json`` in the corpus dir as ``{name, sha256}``
  (digest over raw file bytes, sealing-agnostic — pre-seal artifacts count),
  sorted by name;
- ``epoch_root_sha256`` — sha256 over the canonical member map, a Merkle-style
  root for the corpus at this instant;
- ``prev_epoch_sha256`` / ``prev_epoch_receipt`` — link to the previous epoch
  receipt, forming a chain over corpus *states*;
- ``members_added`` / ``members_removed`` — the delta vs that previous epoch.

``check_epoch_chain`` walks the committed chain and enforces the corpus
discipline: epochs form one fork-free chain, membership is non-decreasing
(a deleted receipt flips ``members_removed``), and the head epoch's root
matches the live corpus — so a receipt modified or dropped after the last
epoch stamp surfaces as ``corpus_drift_since_head_epoch``.

The chain alone cannot see its *own* head being deleted: rewind to an older
epoch still verifies internally while unstamping any members committed since.
``quality/epoch_heads.json`` therefore pins each chain's newest receipt —
stamping updates the pin in the same commit, and ``--check --heads-pin``
turns head deletion into ``epoch_head_missing``/``epoch_head_rollback``
instead of a silent rewind.

Verdicts: ``genesis`` (no previous epoch), ``advancing`` (monotone growth),
``shrinking`` (members removed — recorded fact, honest when intentional).
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_fund.research.epoch_merkle import merkle_root
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def member_tree_root(members: Mapping[str, str]) -> str:
    """RFC 6962-style Merkle root (domain-separated) over ``{name: sha256}``.

    Bound into the epoch payload and the heads pin so a ``corpus_proof.v1``
    verifies against the pin alone — no epoch-receipt fetch needed.
    """
    return merkle_root(members) if members else hash_bytes(b"")


EPOCH_SCHEMA = "corpus_epoch.v1"
GENESIS_PREV = "0" * 64

# ``epoch_heads.json`` is the chain's own bookkeeping (see check_epoch_chain):
# stamping updates it, so as a member it would show permanent post-stamp drift.
# Reserved name — it is never a member under any pattern in any corpus dir.
HEADS_PIN_BASENAME = "epoch_heads.json"

# Basenames never admitted as members — bookkeeping that the chain governs
# rather than measures: the heads pin is rewritten on every stamp.
EXEMPT_BASENAMES = frozenset({HEADS_PIN_BASENAME, ".epoch_stamp.lock"})

# Member rel-paths exempt per corpus dir — exact paths, not basenames, so a
# real member can never hide behind a shared filename: the timestamp-anchor
# manifests are rewritten by each anchor request and authenticate themselves —
# anchors.json via the .tsr imprints + pinned TSA certs, ots_anchors.json via
# the committed digests inside each referenced .ots proof — and the integrity
# checkpoint is a
# digest-of-pins (chained membership would stale itself instantly).
EXEMPT_RELPATHS: frozenset[str] = frozenset(
    {"timestamps/anchors.json", "timestamps/ots_anchors.json", "checkpoint.json"}
)

# Member rel-path *prefixes* exempt per corpus dir. Rekor witness proofs are
# self-authenticating (RFC 6962 inclusion + log-signed timestamps inside the
# file) and churn with every checkpoint rewrite — a new proof replaces the
# retired one each cycle, so chain membership would force an endless
# restamp-rewrite loop. Exemption is prefix-wide because filenames embed the
# per-entry log index.
EXEMPT_PREFIXES: frozenset[str] = frozenset({"witness/", "checkpoints/", "quorum_rotations/"})

# Exempt prefixes are corpus-scoped by the corpus dir's basename: a
# ``witness/`` drop under ``receipts`` is a normal member, and a
# subdirectory that is its own corpus (``.github/workflows``) must not be
# double-chained by the parent corpus.
EXEMPT_PREFIXES_BY_CORPUS: dict[str, frozenset[str]] = {
    "quality": EXEMPT_PREFIXES,
    ".github": frozenset({"workflows/"}),
    # Gitignored vendored clones inside the third_party corpus — the
    # committed vendored tree is chained; locally re-fetched copies are not.
    "third_party": frozenset({"kronos_src/", "kronos_weights/"}),
}

# Directory names never admitted as members in ANY corpus: machine-local
# build/cache artifacts (bytecode, JS deps, notebook checkpoints, tool
# caches, bundler output) — chaining them would break the chain across
# hosts and flag ordinary dev state as tamper. Verified: no tracked file
# lives under any of these names.
EXEMPT_DIRNAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        "test-results",
        "playwright-report",
    }
)

# Directory-name *suffixes* exempt in every corpus (packaging metadata dirs
# like ``dipcatcher.egg-info/`` embed the project name in the dir name).
EXEMPT_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)


def _machine_local(relname: str) -> bool:
    parts = Path(relname).parts
    return any(part in EXEMPT_DIRNAMES or part.endswith(EXEMPT_DIR_SUFFIXES) for part in parts[:-1])


def gitignored_relpaths(root: Path, rels: list[str]) -> set[str]:
    """Corpus-relative paths ``git check-ignore`` drops.

    Tracked files are never ignored. Outside a worktree, or when git is
    unavailable, nothing is ignored — membership stays the directory
    listing (exported bundles, unit fixtures). ``.gitignore`` is a crown
    jewel, so a pattern that hides a path is a pin mutation.
    """
    if not rels:
        return set()
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-z", "--stdin"],
            input="\0".join(rels) + "\0",
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    # 0 = some ignored, 1 = none, anything else = not a worktree.
    if proc.returncode not in (0, 1):
        return set()
    return {name for name in proc.stdout.split("\0") if name}


def member_digests(corpus_dir: Path | str, *, pattern: str = "*.json") -> dict[str, str]:
    """``{rel-path: sha256-of-bytes}`` for every file matching ``pattern``.

    Recursive — member keys are POSIX relative paths so nested evidence dirs
    (``verifier/runs/*.md``) are covered and digests are stable across OSes.
    On a flat dir the keys equal the plain filenames, so existing chains are
    unchanged. ``corpus_epoch_*.json`` receipts are members like any other —
    epochs stamp each other, which is what lets the chain detect a stamped
    epoch's own deletion or mutation.     ``EXEMPT_BASENAMES`` and
    ``EXEMPT_RELPATHS`` are skipped: they are chain bookkeeping, not corpus
    content. Exempt rel-paths are exact corpus-relative paths — a basename
    exemption would let a real member hide under a shared filename.
    Gitignored untracked files are skipped too (tracked files are not):
    a clean checkout must see the same members as the machine that stamped.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"corpus dir {root} does not exist")
    # Sort the POSIX names, not the Path objects. Windows compares paths
    # case-insensitively, so Path order puts ``momentum_…`` before
    # ``README.md``. The epoch contract requires ``sorted(names)`` — the
    # case-sensitive order a Linux stamp already produces.
    named: list[str] = []
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name in EXEMPT_BASENAMES or rel in EXEMPT_RELPATHS or _exempt_member(root, rel):
            continue
        named.append(rel)
    candidates = sorted(named)
    ignored = gitignored_relpaths(root, candidates)
    return {rel: hash_bytes((root / rel).read_bytes()) for rel in candidates if rel not in ignored}


def _exempt_member(root: Path, relname: str) -> bool:
    """True for membership exemptions — must mirror ``member_digests`` exactly.

    Basenames and relpaths exempt unconditionally (chain bookkeeping); the
    prefix set is corpus-scoped — a ``witness/`` drop under ``receipts`` stays
    a normal member. Machine-local dir segments (bytecode caches, JS deps,
    tool caches, bundler output) are exempt in every corpus — they are not
    content, and ``parts[:-1]`` keeps a like-named *file* a member."""
    return (
        Path(relname).name in EXEMPT_BASENAMES
        or relname in EXEMPT_RELPATHS
        or _machine_local(relname)
        or any(
            relname.startswith(prefix) for prefix in EXEMPT_PREFIXES_BY_CORPUS.get(root.name, ())
        )
    )


def _nfc_errors(members: Mapping[str, str]) -> list[str]:
    """Member names must be NFC-canonical — Unicode-equivalent spellings
    (NFC ``café`` vs NFD ``café``) alias to one file on APFS/HFS+ and
    Windows-lookup filesystems while remaining distinct strings on Linux,
    which would make inclusion/absence semantics platform-dependent.

    Also rejects distinct names that collide under NFC+casefold
    (``A.json``/``a.json``) — they can't coexist on default macOS/Windows
    checkouts, so a corpus admitting both wouldn't be portable."""
    errors = [f"member_name_not_nfc:{n!a}" for n in members if unicodedata.normalize("NFC", n) != n]
    seen: dict[str, str] = {}
    for n in members:
        folded = unicodedata.normalize("NFC", n).casefold()
        other = seen.get(folded)
        if other is not None and other != n:
            errors.append(f"member_name_alias:{other!a}|{n!a}")
        else:
            seen[folded] = n
    return errors


def _symlink_errors(root: Path, pattern: str) -> list[str]:
    """Symlink members are refused: ``is_file()`` follows links, so a
    link's *target* content is hashed — retargeting is a mutation vector
    (and the target may live outside the corpus entirely). A corpus member
    must be a real file whose bytes the name unambiguously owns."""
    # Scan every entry, not only glob matches: a symlinked *directory* never
    # matches the member pattern, but could hide content under the corpus.
    return [
        f"member_is_symlink:{p.relative_to(root).as_posix()!a}"
        for p in sorted(root.rglob("*"))
        if p.is_symlink()
    ]


_PORTABLE_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


def _portable_name_errors(members: Mapping[str, str]) -> list[str]:
    """Member names must not carry control, format, or line-separator
    characters — C0/DEL inject fake lines into human-facing verifier output,
    bidi/format codepoints (U+202E…) visually rename members, and Zl/Zp break
    line-oriented tooling. ASCII-repr'd in labels so the name can't inject
    even while being reported."""
    return [
        f"member_name_not_portable:{n!a}"
        for n in members
        if any(unicodedata.category(c) in _PORTABLE_FORBIDDEN_CATEGORIES for c in n)
    ]


def epoch_root(members: Mapping[str, str]) -> str:
    """Merkle-style root over the member map — order-free via canonical JSON."""
    for name, digest in members.items():
        if not isinstance(name, str) or not name:
            raise ValueError("member name must be a non-empty string")
        if not (isinstance(digest, str) and len(digest) == 64):
            raise ValueError(f"member {name} digest must be 64-hex")
        int(digest, 16)  # raises on non-hex
    return hash_bytes(canonical_json_bytes(dict(sorted(members.items()))))


def _member_maps(payload: Mapping[str, Any]) -> dict[str, str]:
    """`members` list → `{name: sha256}`; malformed entries map to empty."""
    out: dict[str, str] = {}
    for entry in payload.get("members") or []:
        if isinstance(entry, Mapping):
            name, sha = entry.get("name"), entry.get("sha256")
            if isinstance(name, str) and isinstance(sha, str):
                out[name] = sha
    return out


def _epoch_receipts(
    corpus_dir: Path,
) -> tuple[list[tuple[Path, Mapping[str, Any]]], list[str]]:
    """Epoch candidates among top-level ``*.json``, plus prefix squatters.

    Any file whose body (or v2 ``payload``) claims ``schema``/``kind`` =
    ``corpus_epoch.v1`` is a candidate — the ``corpus_epoch_*`` filename
    convention is writer-side, not a reader filter. Files *named*
    ``corpus_epoch_*.json`` that fail to yield an epoch payload squat the
    reserved prefix: they can't forge a chain, but they must not pass
    silently as ordinary members."""
    out: list[tuple[Path, Mapping[str, Any]]] = []
    squatters: list[str] = []
    for path in sorted(corpus_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, UnicodeError, ValueError, RecursionError):
            if path.name.startswith("corpus_epoch_"):
                squatters.append(path.name)
            continue
        body: object = doc.get("payload") if isinstance(doc, Mapping) else None
        candidate = body if isinstance(body, Mapping) else doc
        if isinstance(candidate, Mapping) and (
            candidate.get("schema") == EPOCH_SCHEMA or candidate.get("kind") == EPOCH_SCHEMA
        ):
            out.append((path, candidate))
        elif path.name.startswith("corpus_epoch_"):
            squatters.append(path.name)
    return out, squatters


class EpochStampLocked(RuntimeError):
    """Raised when another process holds the corpus's stamp lock."""


def _acquire_stamp_lock(root: Path) -> Any:
    """Exclusive non-blocking lock over the head-read -> write -> pin sequence.

    Two concurrent stamps that both observe the same head would each write
    ``prev = head`` — a chain fork (``epoch_multiple_heads``) created by
    correct software. The lock serializes the whole stamp; a contended lock
    fails closed instead of forking. POSIX uses ``fcntl.flock``; Windows
    uses ``msvcrt.locking`` over one byte of the lock file. A process that
    never takes the lock is unaffected — the chain checker still catches a fork.
    """
    lock_path = root / ".epoch_stamp.lock"
    # sys.platform, not os.name: mypy drops the other branch. Both locks are
    # per open-file description, so a second handle in this process fails.
    if sys.platform == "win32":
        import msvcrt

        handle = lock_path.open("a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise EpochStampLocked(
                f"another corpus-epoch stamp is in flight on {root} (lock: {lock_path})"
            ) from exc
        return handle

    import fcntl

    handle = lock_path.open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise EpochStampLocked(
            f"another corpus-epoch stamp is in flight on {root} (lock: {lock_path})"
        ) from exc
    return handle


def corpus_epoch(
    corpus_dir: Path | str,
    *,
    head_sha: str | None = None,
    pattern: str = "*.json",
    allow_new_pattern: bool = False,
) -> dict[str, Any]:
    """Build the epoch receipt over the corpus's current membership.

    Links to the newest committed epoch receipt (highest chain position) as
    ``prev``; membership delta is computed against it. ``pattern`` selects
    which files count as members (default ``*.json``); evidence dirs that
    aren't JSON (e.g. ``verifier/`` markdown reports) stamp the same way.
    """
    root = Path(corpus_dir)
    members = member_digests(root, pattern=pattern)
    nfc = _nfc_errors(members)
    if nfc:
        raise ValueError(
            f"refusing to stamp {root}: non-NFC member names "
            f"{[e.split(':', 1)[1] for e in nfc]} — rename to the NFC form"
        )
    links = _symlink_errors(root, pattern)
    if links:
        raise ValueError(
            f"refusing to stamp {root}: symlink members "
            f"{[e.split(':', 1)[1] for e in links]} — replace with real files"
        )
    unportable = _portable_name_errors(members)
    if unportable:
        raise ValueError(
            f"refusing to stamp {root}: non-portable member names "
            f"{[e.split(':', 1)[1] for e in unportable]}"
        )
    candidates, squatters = _epoch_receipts(root)
    if squatters:
        raise ValueError(
            f"refusing to stamp {root}: files squat the corpus_epoch_* prefix "
            f"without carrying an epoch payload: {[ascii(s) for s in squatters]}"
        )
    # Chains are per-(dir, pattern): only epochs stamped with the same member
    # glob participate. Absent params.pattern means the default "*.json". A
    # receipt claiming corpus_epoch.v1 with non-mapping params can't carry a
    # chain pattern — the checker flags it; it neither parents a stamp nor
    # constrains patterns.
    epochs: list[tuple[Path, Mapping[str, Any]]] = []
    foreign_patterns: set[str] = set()
    for p, e in candidates:
        params = e.get("params")
        if params is not None and not isinstance(params, Mapping):
            continue
        pat = str((params or {}).get("pattern", "*.json"))
        if pat == pattern:
            epochs.append((p, e))
        else:
            foreign_patterns.add(pat)
    # Fail closed on a *new* pattern in a dir that already has an established
    # chain under another glob — a mismatched --glob mints a parallel
    # (dir, pattern) chain whose records then read as unstamped members of
    # the real chain, exactly the drift the heads pin exists to catch.
    if foreign_patterns and not allow_new_pattern:
        raise ValueError(
            f"refusing to stamp {root}: corpus already has epoch chains under "
            f"{sorted(foreign_patterns)}; a new pattern needs --allow-new-pattern"
        )
    # The chain head is the epoch no other epoch names as prev.
    prevs = {e.get("prev_epoch_receipt") for _, e in epochs}
    heads = [(p, e) for p, e in epochs if p.name not in prevs]
    prev_name: str | None = None
    prev_root = GENESIS_PREV
    prev_members: dict[str, str] = {}
    if heads:
        # Deterministic pick if a fork already exists (chain check flags it).
        prev_path, prev_payload = sorted(heads, key=lambda t: t[0].name)[-1]
        prev_name = prev_path.name
        prev_members = _member_maps(prev_payload)
        prev_root = str(prev_payload.get("epoch_root_sha256") or GENESIS_PREV)
    added = sorted(set(members) - set(prev_members))
    removed = sorted(set(prev_members) - set(members))
    if prev_name is None:
        verdict = "genesis"
    elif removed:
        verdict = "shrinking"
    else:
        verdict = "advancing"
    return {
        "kind": EPOCH_SCHEMA,
        "schema": EPOCH_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "inputs_sha256": hash_bytes(
            canonical_json_bytes({"corpus_dir": str(root), "pattern": pattern})
        ),
        "params": ({"head_sha": head_sha} if head_sha else {}) | {"pattern": pattern},
        "epoch_root_sha256": epoch_root(members),
        "member_tree_root": member_tree_root(members),
        "members": [{"name": n, "sha256": s} for n, s in members.items()],
        "n_members": len(members),
        "prev_epoch_sha256": prev_root,
        "prev_epoch_receipt": prev_name,
        "members_added": added,
        "members_removed": removed,
        "verdict": verdict,
    }


def epoch_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``corpus_epoch.v1`` internal consistency; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != EPOCH_SCHEMA:
        errors.append("kind_not_corpus_epoch")
    if payload.get("schema") != EPOCH_SCHEMA:
        errors.append("schema_not_corpus_epoch")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    members = payload.get("members")
    if not isinstance(members, list):
        errors.append("members_not_list")
        members = []
    member_map = _member_maps(payload)
    if len(member_map) != len(members):
        errors.append("members_malformed_or_dup_names")
    names = [
        m["name"] for m in members if isinstance(m, Mapping) and isinstance(m.get("name"), str)
    ]
    if len(names) != len(members):
        errors.append("member_names_not_strings")
    elif names != sorted(names):
        errors.append("members_not_sorted")
    n_members = payload.get("n_members")
    if not isinstance(n_members, int) or n_members != len(members):
        errors.append("n_members_mismatch")
    root = payload.get("epoch_root_sha256")
    if not (isinstance(root, str) and len(root) == 64):
        errors.append("epoch_root_sha256")
    elif member_map:
        try:
            if epoch_root(member_map) != root:
                errors.append("epoch_root_mismatch")
        except ValueError:
            errors.append("member_digest_not_hex")
    tree = payload.get("member_tree_root")
    if tree is not None:
        if not (isinstance(tree, str) and len(tree) == 64):
            errors.append("member_tree_root")
        elif member_map:
            try:
                if member_tree_root(member_map) != tree:
                    errors.append("member_tree_root_mismatch")
            except (ValueError, KeyError):
                errors.append("member_digest_not_hex")
    prev = payload.get("prev_epoch_sha256")
    if not (isinstance(prev, str) and len(prev) == 64):
        errors.append("prev_epoch_sha256")
    for field in ("members_added", "members_removed"):
        lst = payload.get(field)
        if not isinstance(lst, list) or not all(isinstance(x, str) for x in lst):
            errors.append(f"{field}_not_name_list")
            continue
        if lst != sorted(lst):
            errors.append(f"{field}_not_sorted")
        if field == "members_removed" and set(lst) & set(names):
            errors.append("members_removed_still_present")
    verdict = payload.get("verdict")
    if verdict not in ("genesis", "advancing", "shrinking"):
        errors.append("verdict_unknown")
    else:
        removed = payload.get("members_removed") or []
        prev_name = payload.get("prev_epoch_receipt")
        if verdict == "genesis" and prev_name is not None:
            errors.append("verdict_genesis_with_prev")
        if verdict == "shrinking" and not removed:
            errors.append("verdict_shrinking_without_removals")
        if verdict == "advancing" and (removed or prev_name is None):
            errors.append("verdict_advancing_invalid")
    params = payload.get("params")
    if params is not None and not isinstance(params, Mapping):
        errors.append("params_not_mapping")
    elif isinstance(params, Mapping):
        pat = params.get("pattern")
        if pat is not None and not isinstance(pat, str):
            errors.append("params_pattern_not_str")
        head_sha = params.get("head_sha")
        if head_sha is not None:
            hex40 = (
                isinstance(head_sha, str)
                and len(head_sha) == 40
                and all(c in "0123456789abcdef" for c in head_sha)
            )
            if not hex40:
                errors.append("params_head_sha_not_sha1")
    inputs = payload.get("inputs_sha256")
    if not (isinstance(inputs, str) and len(inputs) == 64):
        errors.append("inputs_sha256")
    return errors


def epoch_heads_key(corpus_dir: Path | str, pattern: str) -> str:
    """Pin-file key for one (dir, pattern) chain, e.g. ``receipts/*.json``."""
    return f"{Path(corpus_dir).as_posix()}/{pattern}"


def load_heads_pin(pin_path: Path | str) -> dict[str, dict[str, str]]:
    """Load ``quality/epoch_heads.json`` → ``{key: {"receipt", "sha256"}}``."""
    path = Path(pin_path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("heads"), dict):
        raise ValueError("heads pin must be an object with a 'heads' map")
    heads: dict[str, dict[str, Any]] = {}
    for key, entry in raw["heads"].items():
        if (
            isinstance(key, str)
            and isinstance(entry, Mapping)
            and isinstance(entry.get("receipt"), str)
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
        ):
            record = {"receipt": entry["receipt"], "sha256": entry["sha256"]}
            tree_root = entry.get("tree_root")
            if tree_root is not None:
                if not (isinstance(tree_root, str) and len(tree_root) == 64):
                    raise ValueError(f"heads pin entry {key!r} malformed")
                record["tree_root"] = tree_root
            chain_root = entry.get("chain_root")
            if chain_root is not None:
                if not (isinstance(chain_root, str) and len(chain_root) == 64):
                    raise ValueError(f"heads pin entry {key!r} malformed")
                record["chain_root"] = chain_root
            n_epochs = entry.get("n_epochs")
            if n_epochs is not None:
                if not (isinstance(n_epochs, int) and n_epochs > 0):
                    raise ValueError(f"heads pin entry {key!r} malformed")
                record["n_epochs"] = n_epochs
            heads[key] = record
        else:
            raise ValueError(f"heads pin entry {key!r} malformed")
    return heads


EPOCH_ACKS_SCHEMA = "epoch_acknowledgments.v1"
_ACK_FIELDS = frozenset(
    {"corpus", "member", "transition", "from_sha256", "to_sha256", "epoch", "reason"}
)
_ACK_TRANSITIONS = frozenset({"removed", "mutated"})


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def load_epoch_acks(path: Path | str) -> list[dict[str, Any]]:
    """Load the committed epoch-history acknowledgment ledger; fail closed.

    An append-only chain cannot forget a recorded removal or mutation, and
    rewriting the chain to hide one is exactly the tamper it exists to catch.
    The ledger is how a reviewed historical incident stays *visible* without
    failing every future check: each entry binds exactly one transition
    already recorded in a sealed epoch — ``corpus`` (heads-pin key, e.g.
    ``receipts/*.json``), ``member``, ``transition`` (``removed`` /
    ``mutated``), ``from_sha256`` (the digest the previous epoch stamped),
    ``to_sha256`` (the digest the recording epoch stamped; ``None`` for a
    removal), ``epoch`` (the epoch receipt that records it) and a non-empty
    ``reason``. Live drift is never acknowledgeable — a removal or rewrite
    must be stamped first, so the ack names sealed history, not intent.

    Raises ``ValueError`` on any malformed, unknown-field or duplicate entry:
    an acknowledgment that cannot be matched exactly must not be silently
    dropped or widened.
    """
    from quant_fund.utils.receipt import load_json_strict

    raw = load_json_strict(Path(path).read_bytes())
    if not isinstance(raw, Mapping) or raw.get("schema") != EPOCH_ACKS_SCHEMA:
        raise ValueError(f"epoch acks must be an object with schema {EPOCH_ACKS_SCHEMA!r}")
    entries = raw.get("acks")
    if not isinstance(entries, list) or set(raw) != {"schema", "acks"}:
        raise ValueError("epoch acks must carry exactly 'schema' and an 'acks' list")
    acks: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(entries):
        problem = _ack_entry_problem(entry)
        if problem is not None:
            raise ValueError(f"epoch ack #{index}: {problem}")
        key = (entry["corpus"], entry["member"], entry["epoch"])
        if key in seen:
            raise ValueError(f"epoch ack #{index}: duplicate acknowledgment {key!a}")
        seen.add(key)
        acks.append(dict(entry))
    return acks


def _ack_epoch_name_ok(epoch: object) -> bool:
    return (
        isinstance(epoch, str)
        and epoch.startswith("corpus_epoch_")
        and epoch.endswith(".json")
        and len(epoch) == len("corpus_epoch_") + 16 + len(".json")
    )


def _ack_digests_problem(entry: Mapping[str, Any]) -> str | None:
    if not _is_hex64(entry["from_sha256"]):
        return "from_sha256 must be 64 lowercase hex"
    to_sha = entry["to_sha256"]
    if entry["transition"] == "removed":
        return None if to_sha is None else "a removal carries to_sha256 null"
    if _is_hex64(to_sha) and to_sha != entry["from_sha256"]:
        return None
    return "a mutation needs a distinct 64-hex to_sha256"


def _ack_entry_problem(entry: object) -> str | None:
    """Why one ledger entry is malformed, or ``None`` when it is well-formed."""
    if not isinstance(entry, Mapping) or set(entry) != _ACK_FIELDS:
        return f"must carry exactly {sorted(_ACK_FIELDS)}"
    corpus, member = entry["corpus"], entry["member"]
    if not (isinstance(corpus, str) and "/" in corpus and isinstance(member, str) and member):
        return "corpus/member malformed"
    if entry["transition"] not in _ACK_TRANSITIONS:
        return "transition must be removed|mutated"
    if not _ack_epoch_name_ok(entry["epoch"]):
        return "epoch must name a corpus_epoch_<hex16>.json"
    reason = entry["reason"]
    if not (isinstance(reason, str) and reason.strip()):
        return "reason must be non-empty"
    return _ack_digests_problem(entry)


def _ack_explains(
    ack: Mapping[str, Any] | None, transition: str, from_sha: str, to_sha: str | None
) -> bool:
    """True iff ``ack`` names exactly this recorded transition's digests."""
    return (
        ack is not None
        and ack["transition"] == transition
        and ack["from_sha256"] == from_sha
        and ack["to_sha256"] == to_sha
    )


def _unmatched_ack_errors(
    ack_index: Mapping[tuple[str, str], Any], used: set[tuple[str, str]]
) -> list[str]:
    """An acknowledgment that matched no recorded transition explains nothing
    — stale, mistyped, or pre-authorizing a future rewrite. Fail closed."""
    return [f"epoch_ack_unmatched:{m!a}@{e!a}" for m, e in sorted(set(ack_index) - used)]


@dataclass
class _AckState:
    """Shared bookkeeping for transition findings in one chain walk."""

    index: Mapping[tuple[str, str], Mapping[str, Any]]
    used: set[tuple[str, str]]
    errors: list[str]
    acknowledged: list[str]

    def record(
        self, label: str, member: str, epoch: str, transition: str, frm: str, to: str | None
    ) -> None:
        """File ``label`` as acknowledged iff an ack names exactly this transition."""
        explained = _ack_explains(self.index.get((member, epoch)), transition, frm, to)
        if explained:
            self.used.add((member, epoch))
        (self.acknowledged if explained else self.errors).append(label)


def _check_transition(
    root: Path,
    prev: Mapping[str, Any],
    cur_name: str,
    cur: Mapping[str, Any],
    acks: _AckState,
    allow_member_updates: bool,
) -> None:
    """Membership findings for one ``prev -> cur`` link: membership must be
    non-decreasing and the declared deltas honest. Members under a
    corpus-scoped exempt prefix churn by rule (self-authenticating files like
    Rekor witness proofs replace per checkpoint rewrite) — their
    epoch-to-epoch presence delta is bookkeeping, not evidence loss."""
    prev_members = {k: v for k, v in _member_maps(prev).items() if not _exempt_member(root, k)}
    cur_members = {k: v for k, v in _member_maps(cur).items() if not _exempt_member(root, k)}
    actual_removed = set(prev_members) - set(cur_members)
    for gone in sorted(actual_removed):
        label = f"member_removed:{gone!a}@{cur_name!a}"
        acks.record(label, gone, cur_name, "removed", prev_members[gone], None)
    declared_removed = {
        n for n in (cur.get("members_removed") or []) if not _exempt_member(root, n)
    }
    if declared_removed != actual_removed:
        acks.errors.append(f"members_removed_dishonest:{cur_name!a}")
    declared_added = {n for n in (cur.get("members_added") or []) if not _exempt_member(root, n)}
    if declared_added != set(cur_members) - set(prev_members):
        acks.errors.append(f"members_added_dishonest:{cur_name!a}")
    if allow_member_updates:
        return
    # Mutated members: same name, different digest.
    for kept in sorted(k for k in set(prev_members) & set(cur_members)):
        if prev_members[kept] != cur_members[kept]:
            label = f"member_mutated:{kept!a}@{cur_name!a}"
            acks.record(label, kept, cur_name, "mutated", prev_members[kept], cur_members[kept])


def _fold(name: str) -> str:
    """Name identity on case-insensitive / normalizing filesystems (the
    Windows fleet host): ``a.json``, ``A.json``, an NFD spelling and (NTFS)
    ``a.json.`` or ``a.json `` can open the same file, so history treats them
    as one name. Over-folding is safe: only a fold-equal name with
    *different* bytes is ever flagged."""
    folded = unicodedata.normalize("NFC", name).casefold()
    return "/".join(part.rstrip(". ") for part in folded.split("/"))


def _resurrections(
    root: Path, by_name: Mapping[str, Mapping[str, Any]], child_of: Mapping[str, str], genesis: str
) -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    """``(member, epoch, last_sha, new_sha)`` for append-only names that were
    removed and later re-added with different bytes, plus the last stamped
    digest of every name the chain ever held (keyed by ``_fold``).

    Removal-then-re-add is a mutation spread over two epochs: an acknowledged
    removal must not launder a rewrite of a stamped name — nor may a re-add
    under another case or normalization of that name. Re-adding the same
    bytes is not flagged (restoring evidence is not a rewrite). Walks from the
    single genesis along ``child_of``; forks are reported elsewhere."""
    last: dict[str, str] = {}
    prev: dict[str, str] = {}
    found: list[tuple[str, str, str, str]] = []
    cur: str | None = genesis
    seen: set[str] = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        members = {
            k: v for k, v in _member_maps(by_name[cur]).items() if not _exempt_member(root, k)
        }
        found += [
            (k, cur, last[_fold(k)], members[k])
            for k in sorted(set(members) - set(prev))
            if _fold(k) in last and last[_fold(k)] != members[k]
        ]
        last.update({_fold(k): v for k, v in members.items()})
        prev = members
        cur = child_of.get(cur)
    return found, last


def _live_resurrection_errors(
    live: Mapping[str, str], unstamped: list[str], history: Mapping[str, str]
) -> list[str]:
    """Unstamped live members reusing a name (``_fold``-equal) the chain
    already stamped with other bytes. Not acknowledgeable — like all live
    drift — and, on an append-only chain, never stampable through a PR
    either: new evidence takes a new name."""
    return [
        f"member_resurrected_live:{name!a}"
        for name in unstamped
        if _fold(name) in history and history[_fold(name)] != live[name]
    ]


def acks_for_corpus(acks: list[dict[str, Any]], corpus_key: str) -> list[dict[str, Any]]:
    """The ledger entries scoped to one (dir, pattern) chain key."""
    return [a for a in acks if a["corpus"] == corpus_key]


def pinned_head_members(corpus_dir: Path | str, pinned: Mapping[str, Any]) -> dict[str, str]:
    """``{member: sha256}`` stamped by the head epoch a heads-pin entry names.

    Fails closed (``ValueError``) unless the named epoch receipt exists in
    ``corpus_dir``, its bytes hash to the pinned digest, and it verifies as a
    sealed ``corpus_epoch.v1`` — a pin is only an authority over bytes it
    actually authenticates.
    """
    from quant_fund.research.receipt_v2 import verify_receipt_file

    name = str(pinned.get("receipt") or "")
    path = Path(corpus_dir) / name
    if not name or "/" in name or not path.is_file():
        raise ValueError(f"pinned head epoch {name!a} is missing from {corpus_dir}")
    if hash_bytes(path.read_bytes()) != pinned.get("sha256"):
        raise ValueError(f"pinned head epoch {name!a} does not match the pinned digest")
    doc = json.loads(path.read_text(encoding="utf-8"))
    body = doc.get("payload") if isinstance(doc.get("payload"), Mapping) else doc
    if not verify_receipt_file(path)["valid"] or epoch_contract_errors(body):
        raise ValueError(f"pinned head epoch {name!a} is not a valid sealed epoch")
    return _member_maps(body)


def _epoch_link(corpus_dir: Path, name: str) -> tuple[str, str | None]:
    """``(file_sha256, prev_epoch_receipt)`` for one epoch file; ``ValueError``
    when it is missing, path-like or unparseable."""
    path = corpus_dir / name
    if not name or "/" in name or "\\" in name or not path.is_file():
        raise ValueError(f"chain epoch {name!a} is missing from {corpus_dir}")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"chain epoch {name!a} is not parseable JSON") from exc
    body = doc.get("payload") if isinstance(doc, Mapping) else None
    body = body if isinstance(body, Mapping) else doc
    prev = body.get("prev_epoch_receipt") if isinstance(body, Mapping) else None
    return hash_bytes(raw), prev if isinstance(prev, str) else None


def chain_ancestry(corpus_dir: Path | str, pinned: Mapping[str, Any]) -> frozenset[str]:
    """Epoch names in the chain a heads-pin entry commits to: the pinned head
    and every epoch it transitively names as ``prev``.

    Authenticated, not walked on trust: the head's bytes must match the
    pinned ``sha256`` and the genesis→head sequence of ``(name, file_sha)``
    must reproduce the pinned ``chain_root`` (and ``n_epochs``). Any gap,
    cycle, unreadable link or root mismatch raises ``ValueError`` — a caller
    that grants authority to "epochs the base already held" must never be
    handed a set an attacker shaped by editing older epoch files."""
    from quant_fund.research.epoch_merkle import chain_tree_root

    root = Path(corpus_dir)
    head = str(pinned.get("receipt") or "")
    chain: list[tuple[str, str]] = []
    cur: str | None = head
    while cur is not None:
        if any(cur == n for n, _ in chain):
            raise ValueError(f"chain epoch {cur!a} repeats (cycle)")
        sha, prev = _epoch_link(root, cur)
        chain.append((cur, sha))
        cur = prev
    chain.reverse()
    if chain[-1][1] != pinned.get("sha256"):
        raise ValueError(f"pinned head epoch {head!a} does not match the pinned digest")
    exp_root = pinned.get("chain_root")
    if not isinstance(exp_root, str) or chain_tree_root(chain) != exp_root:
        raise ValueError(f"chain ending at {head!a} does not reproduce the pinned chain_root")
    exp_n = pinned.get("n_epochs")
    if isinstance(exp_n, int) and exp_n != len(chain):
        raise ValueError(f"chain ending at {head!a} has {len(chain)} epochs, pin says {exp_n}")
    return frozenset(n for n, _ in chain)


def ack_ledger_base_errors(
    acks: list[dict[str, Any]],
    base_acks: list[dict[str, Any]],
    base_chains: Mapping[str, frozenset[str]],
    append_only: frozenset[str],
) -> list[str]:
    """PR rule for the acknowledgment ledger on append-only chains.

    An entry the base ledger did not already hold may only explain history
    the *base* chain recorded (its ``epoch`` is in ``base_chains[corpus]``,
    from ``chain_ancestry`` over the base pin). Otherwise one diff could stamp
    a removal or rewrite of stamped evidence in a new epoch and excuse it in
    the same breath — every transition type, not only mutations. Mutable
    chains (``allow_member_updates``) are outside this rule: their removals
    are ordinary, stamped and acknowledged together."""
    base = {canonical_json_bytes(dict(a)) for a in base_acks}
    return [
        f"epoch_ack_postdates_base:{a['corpus']!a}:{a['member']!a}@{a['epoch']!a}"
        for a in acks
        if a["corpus"] in append_only
        and canonical_json_bytes(dict(a)) not in base
        and a["epoch"] not in base_chains.get(a["corpus"], frozenset())
    ]


def update_heads_pin(
    pin_path: Path | str,
    corpus_dir: Path | str,
    pattern: str,
    head_receipt: Path,
) -> Path:
    """Record ``head_receipt`` as the pinned chain head for (dir, pattern).

    Called by the stamping path so the pin and the epoch receipt land in the
    same commit — the committed pin is then the authoritative statement of
    which epoch is the newest, making head deletion a detectable rollback.
    """
    path = Path(pin_path)
    key = epoch_heads_key(corpus_dir, pattern)
    heads: dict[str, dict[str, Any]] = (
        {k: dict(v) for k, v in load_heads_pin(path).items()} if path.is_file() else {}
    )
    try:
        head_doc = json.loads(head_receipt.read_bytes())
    except (OSError, ValueError):
        head_doc = {}
    payload = head_doc.get("payload") if isinstance(head_doc, Mapping) else None
    body = payload if isinstance(payload, Mapping) else head_doc
    tree_root = body.get("member_tree_root") if isinstance(body, Mapping) else None
    heads[key] = {
        "receipt": head_receipt.name,
        "sha256": hash_bytes(head_receipt.read_bytes()),
    }
    if isinstance(tree_root, str) and len(tree_root) == 64:
        heads[key]["tree_root"] = tree_root
    # Position-tree commitment: a Merkle root over the ordered epoch chain
    # (leaves bind (position, name, file-sha)) so `epoch_position.v1`
    # proofs verify offline against this pin in O(log n).
    from quant_fund.research.epoch_merkle import chain_tree_root, ordered_epoch_chain

    try:
        ordered, chain_errors = ordered_epoch_chain(corpus_dir, pattern=pattern)
    except (OSError, ValueError):
        ordered, chain_errors = [], ["chain_compute_failed"]
    if not chain_errors and ordered and ordered[-1][0] == head_receipt.name:
        names_sha = [(name, file_sha) for name, file_sha, _, _, _ in ordered]
        heads[key]["chain_root"] = chain_tree_root(names_sha)
        heads[key]["n_epochs"] = len(names_sha)
    from quant_fund.utils.atomicio import atomic_write_text

    payload = {"schema": "epoch_heads.v1", "heads": heads}
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def check_epoch_chain(
    corpus_dir: Path | str,
    *,
    acks: list[dict[str, Any]] | None = None,
    pattern: str = "*.json",
    expected_head: Mapping[str, Any] | None = None,
    require_stamped: bool = False,
    allow_member_updates: bool = False,
) -> dict[str, Any]:
    """Walk the committed epoch chain against the live corpus.

    Returns ``{"errors": [...], "acknowledged": [...], "unstamped": [...],
    "head": <name|None>, "head_epoch_root": <hex|None>}``. Errors are
    integrity violations the chain is authoritative over: forked/disconnected/
    invalid epoch receipts, stamped members removed or mutated, and dishonest
    delta fields. ``acks`` (this chain's entries from ``load_epoch_acks``)
    move a recorded removal/mutation from ``errors`` to ``acknowledged`` only
    when the entry names the exact member, epoch and digests the chain
    records; an entry matching nothing is itself an error
    (``epoch_ack_unmatched``), so the ledger cannot pre-authorize future
    tamper or outlive the history it explains. ``unstamped`` lists corpus
    members not covered by the head epoch — arrivals between stamps are the
    normal state, recorded not flagged.

    ``expected_head`` (from ``load_heads_pin``) pins which epoch receipt is
    the committed head: missing file → ``epoch_head_missing``, byte-drift →
    ``epoch_head_mutated``, and an observed head that isn't the pin or one
    of its descendants → ``epoch_head_rollback`` — closing the hole where
    deleting the newest epoch receipts silently reverts the chain to an
    older (pre-tamper) head.

    ``require_stamped`` promotes ``unstamped`` members to
    ``unstamped_member:`` errors — for corpora whose *every* member is
    security-critical (e.g. CI workflow definitions), an unstamped arrival
    is itself the tamper, not a pending stamp.

    ``allow_member_updates`` relaxes ``member_mutated`` for *mutable*
    corpora (quality manifests, workflow definitions) whose members
    legitimately change between stamps — the chain then provides ordered
    history attestation (every committed state, honestly declared deltas,
    pinned head) while post-stamp edits still surface as
    ``head_member_digest_drift`` until re-stamped. Append-only corpora
    (receipts, verifier records) must leave it False: a digest change
    between epochs there is tamper evidence.
    """
    root = Path(corpus_dir)
    errors: list[str] = []
    acknowledged: list[str] = []
    unstamped: list[str] = []
    ack_index = {(str(a["member"]), str(a["epoch"])): a for a in acks or []}
    ack_used: set[tuple[str, str]] = set()

    # Committed head pin (phase 1, chain-free): the pin file states which
    # epoch receipt is the newest — a missing or byte-drifted pinned file is
    # a violation even when no chain is left to check. The ancestry walk runs
    # below once the chain resolves, catching rollback to an older head.
    exp_name: str | None = None
    if expected_head is not None:
        exp_name = str(expected_head.get("receipt") or "") or None
        exp_sha = str(expected_head.get("sha256") or "")
        exp_path = root / exp_name if exp_name else root / ""
        if exp_name is None or not exp_path.is_file():
            errors.append(f"epoch_head_missing:{(exp_name or '')!a}")
        elif hash_bytes(exp_path.read_bytes()) != exp_sha:
            errors.append(f"epoch_head_mutated:{exp_name!a}")

    # Chains are per-(dir, pattern) — epochs stamped under a different member
    # glob form their own chain and are ignored here. Files squatting the
    # corpus_epoch_* reserved prefix (no epoch payload) are flagged, not
    # ignored: they can't forge a chain but must not pass silently.
    candidates, squatters = _epoch_receipts(root)
    errors.extend(f"epoch_prefix_squat:{name!a}" for name in squatters)
    epochs: list[tuple[Path, Mapping[str, Any]]] = []
    for epoch_path, epoch_payload in candidates:
        params = epoch_payload.get("params")
        if params is not None and not isinstance(params, Mapping):
            errors.append(f"epoch_params_malformed:{epoch_path.name}")
            continue
        if (params or {}).get("pattern", "*.json") == pattern:
            epochs.append((epoch_path, epoch_payload))
    if not epochs:
        if exp_name is not None:
            errors.append(f"epoch_head_rollback:{exp_name!a}")
        errors.append("no_epoch_receipts")
        return {
            "errors": errors,
            "acknowledged": acknowledged,
            "unstamped": [],
            "head": None,
            "head_epoch_root": None,
        }

    # Seal-verify every epoch receipt before trusting its claims.
    from quant_fund.research.receipt_v2 import verify_receipt_file

    sealed: dict[str, Mapping[str, Any]] = {}
    for path, payload in epochs:
        ver = verify_receipt_file(path)
        if not ver["valid"]:
            errors.append(f"epoch_receipt_invalid:{path.name!a}")
            continue
        sealed[path.name] = payload
    if not sealed:
        errors.append("no_valid_epoch_receipts")
        return {
            "errors": errors,
            "acknowledged": acknowledged,
            "unstamped": unstamped,
            "head": None,
            "head_epoch_root": None,
        }

    # Link the chain: each non-genesis epoch names prev_epoch_receipt.
    by_name = sealed
    child_of: dict[str, str] = {}  # prev_name -> epoch name (for fork check)
    genesis: list[str] = []
    for name, payload in by_name.items():
        prev = payload.get("prev_epoch_receipt")
        if prev is None:
            genesis.append(name)
            continue
        if prev not in by_name:
            errors.append(f"epoch_orphan:{name!a}")
            continue
        if payload.get("prev_epoch_sha256") != by_name[prev].get("epoch_root_sha256"):
            errors.append(f"epoch_prev_root_mismatch:{name!a}")
        if prev in child_of:
            errors.append(f"epoch_fork:{prev!a}->{child_of[prev]!a},{name!a}")
        child_of[prev] = name
    if len(genesis) > 1:
        errors.append(f"epoch_multiple_genesis:{','.join(sorted(ascii(g) for g in genesis))}")

    # Membership must be non-decreasing along the chain. Members under a
    # corpus-scoped exempt prefix churn by rule (self-authenticating files
    # like Rekor witness proofs replace per checkpoint rewrite) — their
    # epoch-to-epoch presence delta is bookkeeping, not evidence loss.
    ack_state = _AckState(ack_index, ack_used, errors, acknowledged)
    for prev_name, cur_name in child_of.items():
        _check_transition(
            root, by_name[prev_name], cur_name, by_name[cur_name], ack_state, allow_member_updates
        )
    history: dict[str, str] = {}
    if not allow_member_updates and len(genesis) == 1:
        found, history = _resurrections(root, by_name, child_of, genesis[0])
        for name, epoch, last_sha, new_sha in found:
            ack_state.record(
                f"member_resurrected:{name!a}@{epoch!a}", name, epoch, "mutated", last_sha, new_sha
            )

    # Head vs live corpus: stamped membership must hold exactly; files added
    # after the head stamp are unstamped (normal), not violations.
    heads = set(by_name) - set(child_of)
    head_name: str | None = None
    head_root: str | None = None
    if len(heads) == 1:
        head_name = next(iter(heads))
        head = by_name[head_name]
        root_val = head.get("epoch_root_sha256")
        head_root = root_val if isinstance(root_val, str) else None
        live = member_digests(root, pattern=pattern)
        errors += _nfc_errors(live)
        errors += _symlink_errors(root, pattern)
        errors += _portable_name_errors(live)
        head_members = {k: v for k, v in _member_maps(head).items() if not _exempt_member(root, k)}
        stamped = set(head_members)
        # Live drift is never acknowledgeable: a removal must be stamped
        # (recorded in a sealed epoch) before the ledger can explain it.
        for name in sorted(stamped - set(live)):
            errors.append(f"head_member_missing_live:{name!a}")
        for name, sha in head_members.items():
            if name in live and live[name] != sha:
                errors.append(f"head_member_digest_drift:{name!a}")
        unstamped = sorted(set(live) - stamped - set(by_name))
        errors += _live_resurrection_errors(live, unstamped, history)
        if require_stamped:
            errors.extend(f"unstamped_member:{name!a}" for name in unstamped)
    elif len(heads) > 1:
        errors.append(f"epoch_multiple_heads:{','.join(sorted(ascii(h) for h in heads))}")

    # Committed head pin (phase 2): the observed head must be the pinned head
    # or one of its descendants — a shorter or forked chain means head epochs
    # were deleted to rewind past a committed corpus state.
    if exp_name is not None:
        cur = head_name
        seen: set[str] = set()
        ancestor = False
        while cur is not None and cur not in seen:
            if cur == exp_name:
                ancestor = True
                break
            seen.add(cur)
            prev_ref = by_name.get(cur)
            prev = prev_ref.get("prev_epoch_receipt") if isinstance(prev_ref, Mapping) else None
            cur = prev if isinstance(prev, str) else None
        if not ancestor:
            errors.append(f"epoch_head_rollback:{exp_name!a}")

    # Bound chain fields: when the pin carries the position-commitment
    # fields written by ``update_heads_pin`` (member ``tree_root``, ordered
    # ``chain_root``, ``n_epochs``), recompute them against the observed
    # chain truncated at the pinned head — a pin naming a live head while
    # lying about its committed position root must fail closed, since
    # offline position proofs anchor on these values.
    if exp_name is not None and ancestor and expected_head is not None:
        from quant_fund.research.epoch_merkle import (
            chain_tree_root,
            ordered_epoch_chain,
        )

        ordered, order_errors = ordered_epoch_chain(root, pattern=pattern)
        if not order_errors:
            tip_idx = next((i for i, e in enumerate(ordered) if e[0] == exp_name), None)
            if tip_idx is not None:
                sub = ordered[: tip_idx + 1]
                names_sha = [(e[0], e[1]) for e in sub]
                pinned_head = by_name[exp_name]
                exp_tree = expected_head.get("tree_root")
                if isinstance(exp_tree, str) and pinned_head.get("member_tree_root") != exp_tree:
                    errors.append(f"epoch_head_tree_root_mismatch:{exp_name!a}")
                exp_chain = expected_head.get("chain_root")
                if isinstance(exp_chain, str) and chain_tree_root(names_sha) != exp_chain:
                    errors.append(f"epoch_head_chain_root_mismatch:{exp_name!a}")
                exp_n = expected_head.get("n_epochs")
                if isinstance(exp_n, int) and len(names_sha) != exp_n:
                    errors.append(f"epoch_pin_n_epochs_mismatch:{exp_name!a}")
    errors += _unmatched_ack_errors(ack_index, ack_used)
    return {
        "errors": errors,
        "acknowledged": acknowledged,
        "unstamped": unstamped,
        "head": head_name,
        "head_epoch_root": head_root,
    }


def write_epoch_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal + atomically write a corpus-epoch receipt; fails closed on contract violation."""
    errors = epoch_contract_errors(receipt)
    if errors:
        raise ValueError(f"corpus_epoch receipt violates contract: {errors}")
    dirpath = Path(receipts_dir)
    dirpath.mkdir(parents=True, exist_ok=True)

    from quant_fund.research.fleet_eval import _atomic_write_text
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass" if receipt.get("verdict") in ("genesis", "advancing") else "fail",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = dirpath / f"corpus_epoch_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
