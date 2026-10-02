"""sbom_audit — adversarial probes on the release SBOM generator.

One real coverage hole fixed: a ``[[package]]`` block with a ``name`` but
a malformed/missing ``version`` (or vice versa) was silently dropped —
the SBOM showed fewer dependencies than the lockfile held. Half-formed
blocks now fail closed with ``ValueError``.

Pinned edges: missing lockfile raises ``FileNotFoundError``; a lockfile
with no package blocks raises ``ValueError``; the uv.lock *header* block
(unquoted ``version = 1``) is correctly skipped rather than crashing the
strict parser; invalid quoted ``name = "x"; evil`` injections fail
TOML parsing; entries come out sorted by name; ``lockfile_sha256`` binds
the parsed bytes. Sealed ``sbom_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["sbom_audit", "sbom_audit_bench"]

_HEADER = 'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n'
_PKG = (
    '[[package]]\nname = "{name}"\nversion = "{ver}"\nsource = '
    '{{ registry = "https://pypi.org/simple" }}\n{extra}'
)
_HASH = 'wheels = [\n    {{ url = "u", hash = "sha256:{h}" }},\n]\n'


def _raises(fn: Any) -> str:
    try:
        fn()
    except (ValueError, FileNotFoundError) as exc:
        return f"raise:{type(exc).__name__}"
    return "no-raise"


def sbom_audit() -> dict[str, Any]:
    from fx1.sbom import generate_sbom

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lock = root / "uv.lock"
        h1 = hashlib.sha256(b"w1").hexdigest()
        lock.write_text(
            _HEADER
            + _PKG.format(name="alpha", ver="1.0", extra=_HASH.format(h=h1))
            + _PKG.format(name="beta", ver="2.0", extra="")
        )
        sbom = generate_sbom(lock)
        out["n_entries"] = len(sbom.entries)
        out["sorted"] = [e.name for e in sbom.entries] == sorted(e.name for e in sbom.entries)
        out["hash_carried"] = sbom.entries[0].sha256 == h1
        out["hashless_none"] = sbom.entries[1].sha256 is None
        out["lockfile_sha"] = sbom.lockfile_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
        out["missing_raises"] = _raises(lambda: generate_sbom(root / "ghost.lock"))

        empty = root / "empty.lock"
        empty.write_text(_HEADER)
        out["no_packages_raises"] = _raises(lambda: generate_sbom(empty))

        # name without version — the fixed silent-drop hole
        half = root / "half.lock"
        half.write_text(_HEADER + '[[package]]\nname = "orphan"\n')
        out["half_block_raises"] = _raises(lambda: generate_sbom(half))

        # injection: quote inside a name cannot escape the field
        inj = root / "inj.lock"
        inj.write_text(
            _HEADER
            + _PKG.format(name='evil"; version = "9.9', ver="1.0", extra="")
            + _PKG.format(name="real", ver="1.1", extra="")
        )
        # Invalid TOML must fail closed rather than emitting a partial SBOM.
        out["injection_raises"] = _raises(lambda: generate_sbom(inj))
        out["injection_contained"] = out["injection_raises"] == "raise:ValueError"
    return out


def sbom_audit_bench() -> dict[str, Any]:
    r = sbom_audit()
    ok = (
        r["n_entries"] == 2
        and r["sorted"] is True
        and r["hash_carried"] is True
        and r["hashless_none"] is True
        and r["lockfile_sha"] is True
        and r["missing_raises"] == "raise:FileNotFoundError"
        and r["no_packages_raises"] == "raise:ValueError"
        and r["half_block_raises"] == "raise:ValueError"
        and r["injection_contained"] is True
    )
    payload: dict[str, Any] = {
        "kind": "sbom_audit",
        "schema": "sbom_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "SBOM contract holds: sorted hash-pinned entries, lockfile digest "
            "bound, malformed/half-formed package blocks fail closed, "
            "quoted-name injection contained."
            if ok
            else f"SBOM DEFECT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
