#!/usr/bin/env python3
"""Standalone evidence-bundle auditor — stdlib only, no repo imports.

Verifies a bundle produced by ``dipcatcher evidence-export``:

* ``BUNDLE.json`` declares ``evidence_bundle.v1``, covers exactly the eight
  evidence corpora, and records the same member glob the policy table below
  expects — a manifest that drifts from the auditor's own spec is reported,
  not trusted (the spec here is the auditor's code, not the exporter's data).
* Every corpus directory's epoch chain re-verifies via ``check_chain``
  (imported from ``verify_epoch_chain.py`` — the same independent
  reimplementation, so a bundle that verifies here and fails under
  ``verify-repo --evidence-only``, or vice versa, is itself a finding).
* Nothing else may sit inside a corpus dir: every file must be a chain
  member, an epoch receipt (``corpus_epoch_*.json`` bookkeeping), or a
  declared exempt — an extra file is an unaudited injection.
* The chain bookkeeping the library exempts from membership —
  ``quality/epoch_heads.json`` — must exist (the pin is what the chains
  verify against). ``--checkpoint`` mode additionally authenticates
  ``--pubkey``'s signature against the quorum registry the bundle carries
  as an ordinary quality member.

The policy table mirrors ``repo_integrity.CORPORA[:8]`` (EVIDENCE_CORPORA);
keep in sync — a deliberate spec change in the library is a spec change here.

    python scripts/verify_evidence_bundle.py --root /path/to/bundle \
        [--checkpoint <bundle>/quality/checkpoint.json \
         --pubkey <bundle>/quality/gate_signing.pub]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_epoch_chain import (  # noqa: E402
    _checkpoint_heads,
    _exempt_member,
    _sha256,
    check_chain,
)

EVIDENCE_SPEC: tuple[tuple[str, str, bool, bool, tuple[str, ...]], ...] = (
    ("receipts", "*.json", True, False, ("legacy-unsealed/README.md",)),
    ("verifier", "*.md", True, False, ()),
    (
        "quality",
        "*.json",
        True,
        True,
        ("*.pub", "*.pem", "*.crt", "*.tsr", "*.txt", "timestamps/*", "timestamps/ots/*"),
    ),
    (".github/workflows", "*.yml", True, True, ("README.md",)),
    ("configs", "*", True, True, ()),
    ("artifacts", "*", True, True, ("*.gitkeep",)),
    (".dsh-24x7", "*", False, False, ()),
    ("data/metadata", "*", True, False, ()),
)

BUNDLE_MANIFEST = "BUNDLE.json"
_SIGNATURE = "gate_pins.sig"
HEADS_PIN = "quality/epoch_heads.json"
_BOOKKEEPING = (HEADS_PIN,)
# Corpora whose epoch receipts are gitignored (never clone). A bundle
# exported from a checkout that never stamped them holds members but no
# chain records — explicit skip, mirroring repo_integrity's local_only skip.
LOCAL_ONLY = frozenset({"data/metadata"})


def _load_heads(root: Path, checkpoint: Path | None, pubkey: Path | None) -> tuple[dict, list[str]]:
    """Pin heads — from the signed checkpoint when offered, else the pin file.

    v2 checkpoints authenticate against the quorum registry — the bundle
    carries ``quality/gate_quorum.json`` as a normal member, so the auditor's
    trust is the registry + signature, not the pin file's bytes.
    """
    if checkpoint is not None:
        assert pubkey is not None
        quorum = root / "quality" / "gate_quorum.json"
        heads, pins, errs = _checkpoint_heads(
            json.loads(checkpoint.read_bytes()),
            pubkey,
            quorum if quorum.is_file() else None,
        )
        pin = root / HEADS_PIN
        if pin.is_file():
            want = pins.get(HEADS_PIN)
            if want is not None and want != _sha256(pin.read_bytes()):
                errs.append(f"checkpoint_pin_mismatch:{HEADS_PIN}")
        return heads, errs
    pin = root / HEADS_PIN
    if not pin.is_file():
        return {}, [f"heads_pin_missing:{HEADS_PIN}"]
    raw = json.loads(pin.read_bytes())
    return (raw.get("heads", {}) if isinstance(raw, dict) else {}), []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--pubkey", type=Path, default=None)
    args = ap.parse_args()
    if args.checkpoint is not None and args.pubkey is None:
        print("--checkpoint requires --pubkey", file=sys.stderr)
        return 2

    root = args.root
    errors: list[str] = []
    manifest_path = root / BUNDLE_MANIFEST
    if not manifest_path.is_file():
        print(f"missing {BUNDLE_MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema") != "evidence_bundle.v1":
        errors.append(f"manifest_schema:{manifest.get('schema')!r}")
    declared = manifest.get("corpora", {})
    if set(declared) != {c[0] for c in EVIDENCE_SPEC}:
        errors.append(
            "manifest_corpora_mismatch:"
            + ",".join(sorted(set(declared) ^ {c[0] for c in EVIDENCE_SPEC}))
        )

    pin_heads, head_errs = _load_heads(root, args.checkpoint, args.pubkey)
    errors += head_errs

    # Retired members the chain is allowed to forget (name → last sha256);
    # the file travels with the quality corpus, same as the lib loads it.
    allowed_removals: dict[str, str] = {}
    ar_path = root / "quality" / "epoch_allowed_removals.json"
    if ar_path.is_file():
        raw_ar = json.loads(ar_path.read_bytes())
        if isinstance(raw_ar, dict):
            allowed_removals = {str(k): str(v) for k, v in raw_ar.items() if isinstance(v, str)}

    corpus_dirs = sorted((c[0] for c in EVIDENCE_SPEC), key=len, reverse=True)

    # Closed world: every bundle file is the manifest, the signature file, or
    # a file under a declared corpus. Corpora nest (``.github`` is the parent
    # of the ``.github/workflows`` corpus), so a file is claimed by the
    # LONGEST corpus-dir prefix — files under ``.github`` but outside
    # ``workflows/`` are foreign, not unclaimed extras.
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root).as_posix()
        if rel in (BUNDLE_MANIFEST, _SIGNATURE):
            continue
        owner = next((d for d in corpus_dirs if rel == d or rel.startswith(d + "/")), None)
        if owner is None:
            errors.append(f"foreign_member:{rel}")
            continue
        spec = next(c for c in EVIDENCE_SPEC if c[0] == owner)
        inner = rel[len(owner) + 1 :]
        if f.name.startswith("corpus_epoch_") or _exempt_member(root / owner, inner):
            continue
        if not fnmatch.fnmatch(inner, spec[1]) and not any(
            fnmatch.fnmatch(inner, e) for e in spec[4]
        ):
            errors.append(f"uncovered_member:{rel}")

    for corpus_dir, glob, require_stamped, allow_updates, _exempt in EVIDENCE_SPEC:
        cdir = root / corpus_dir
        if not cdir.is_dir():
            errors.append(f"corpus_missing:{corpus_dir}")
            continue
        if isinstance(declared, dict) and isinstance(declared.get(corpus_dir), dict):
            m_glob = declared[corpus_dir].get("glob")
            if m_glob != glob:
                errors.append(f"manifest_glob_mismatch:{corpus_dir}:{m_glob!r}")
        if corpus_dir in LOCAL_ONLY and not any(cdir.glob("corpus_epoch_*.json")):
            print(f"bundle-check {corpus_dir}: skipped(local_only_no_receipts)")
            continue
        key = f"{corpus_dir}/{glob}"
        expected_head = pin_heads.get(key)
        if pin_heads and expected_head is None:
            errors.append(f"heads_pin_key_missing:{key!r}")
        result = check_chain(
            cdir,
            pattern=glob,
            allowed_removals=allowed_removals,
            expected_head=expected_head,
            require_stamped=require_stamped,
            allow_member_updates=allow_updates,
        )
        ok = not result["errors"]
        print(f"bundle-check {corpus_dir}: {'ok' if ok else 'FAIL'}")
        errors += [f"{corpus_dir}: {e}" for e in result["errors"]]

    for rel in _BOOKKEEPING:
        if not (root / rel).is_file():
            errors.append(f"bookkeeping_missing:{rel}")

    if errors:
        for e in errors:
            print(f"bundle-check FAIL {e}")
        return 1
    print("evidence bundle: all chains intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
