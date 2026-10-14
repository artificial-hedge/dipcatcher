"""RFC 6962 witness for the numeric kernels.

The Student-t objective, the realized-kernel reductions, and the
high-frequency realized-variance contract are the pieces a copy would
have to reproduce to claim the same proper scores. This module binds
those files into an ordered Merkle tree:

- each leaf is ``SHA-256(0x00 || domain || path || bytes || structural)``
  via :func:`quant_fund.audit.merkle.hash_leaf`;
- the structural digest is an abstract-syntax walk that ignores comments
  and docstrings, so a comment-only edit and a logic edit are different
  events;
- inclusion proofs are the Certificate Transparency audit path, so a
  witness for one kernel does not verify under a different root.

The witness is a proof about source identity. It is not a cipher, not a
packer, and not a market result. The kernels stay readable because the
proper-score contract has to be auditable. A witness copied onto a tree
whose kernel bytes differ does not verify.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_fund.audit.merkle import hash_leaf, inclusion_proof, merkle_root, verify_inclusion
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

SCHEMA = "kernel_lineage.v1"
KIND = "kernel_lineage"
DOMAIN = b"dipcatcher.kernel_lineage.v1"
_MAGIC = b"DKLW"
_VERSION = 1
_MAX_PROOF_NODES = 64

KERNEL_PATHS: tuple[str, ...] = (
    "src/quant_fund/audit/kernel_lineage.py",
    "src/quant_fund/flowbars/realized.py",
    "src/quant_fund/metrics/risk_parametric.py",
    "src/quant_fund/metrics/scoring.py",
    "src/quant_fund/models/realized.py",
    "src/quant_fund/realized/hf_rv.py",
)

_PATH_SET = frozenset(KERNEL_PATHS)
_DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SKIPPED_FIELDS = frozenset(
    {"lineno", "col_offset", "end_lineno", "end_col_offset", "type_comment"}
)


@dataclass(frozen=True)
class KernelWitnessModule:
    """One kernel inside a parsed witness. Digests are raw SHA-256."""

    path: str
    byte_digest: bytes
    structural_digest: bytes
    leaf_hash: bytes
    proof: tuple[bytes, ...]


@dataclass(frozen=True)
class KernelWitness:
    """Parsed witness. ``root`` is the RFC 6962 Merkle root."""

    root: bytes
    modules: tuple[KernelWitnessModule, ...]


class _Cursor:
    """Big-endian cursor over a witness body. Over-reads fail closed."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._at = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self._at + n > len(self._data):
            raise ValueError("kernel witness truncated")
        chunk = self._data[self._at : self._at + n]
        self._at += n
        return chunk

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "big")

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def finished(self) -> bool:
        return self._at == len(self._data)


def default_repo_root() -> Path:
    """Walk parents until the dipcatcher checkout that contains this package."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "quant_fund").is_dir():
            return parent
    raise ValueError("kernel lineage could not find the repo root")


def _sha256(data: bytes) -> bytes:
    """Raw SHA-256, the same function the receipt hex digests use."""
    return bytes.fromhex(hash_bytes(data))


def _is_docstring(node: object) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _emit_value(value: object, parts: list[str]) -> None:
    if isinstance(value, ast.AST):
        _emit_tree(value, parts)
        return
    if isinstance(value, list):
        parts.append("[")
        for item in value:
            _emit_value(item, parts)
        parts.append("]")
        return
    parts.append(repr(value))


def _emit_body(body: list[object], parts: list[str]) -> None:
    statements = body[1:] if body and _is_docstring(body[0]) else body
    parts.append("[")
    for statement in statements:
        _emit_value(statement, parts)
    parts.append("]")


def _emit_tree(node: ast.AST, parts: list[str]) -> None:
    parts.append(type(node).__name__)
    skip_doc = isinstance(node, _DOC_OWNERS)
    for field, value in ast.iter_fields(node):
        if field in _SKIPPED_FIELDS:
            continue
        parts.append(field)
        if skip_doc and field == "body" and isinstance(value, list):
            _emit_body(value, parts)
        else:
            _emit_value(value, parts)


def structural_digest(source: str) -> bytes:
    """SHA-256 of a docstring-free syntax walk.

    Comments never enter the AST, so they do not move this digest. A
    changed constant, name, or control-flow node does.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("kernel source is not valid Python") from exc
    parts: list[str] = []
    _emit_tree(tree, parts)
    return _sha256("\n".join(parts).encode("utf-8"))


def leaf_preimage(rel: str, data: bytes, structural: bytes) -> bytes:
    """Domain-separated leaf bytes. The path length is part of the preimage."""
    if len(structural) != 32:
        raise ValueError("structural digest must be 32 bytes")
    path = rel.encode("utf-8")
    if not path or len(path) > 65535:
        raise ValueError("kernel path is empty or too long")
    return DOMAIN + b"\x00" + len(path).to_bytes(2, "big") + path + data + structural


def _read_member(root: Path, rel: str) -> bytes:
    if rel not in _PATH_SET:
        raise ValueError(f"path is not a kernel member: {rel}")
    raw = root / rel
    if raw.is_symlink():
        raise ValueError(f"kernel path is a symlink: {rel}")
    if not raw.is_file():
        raise ValueError(f"kernel path missing: {rel}")
    resolved = raw.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"kernel path escapes the repo root: {rel}")
    return raw.read_bytes()


def _module_parts(root: Path, rel: str) -> dict[str, Any]:
    data = _read_member(root, rel)
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"kernel source is not utf-8: {rel}") from exc
    structural = structural_digest(text)
    preimage = leaf_preimage(rel, data, structural)
    return {
        "path": rel,
        "byte_sha256": hash_bytes(data),
        "structural_sha256": structural.hex(),
        "preimage": preimage,
        "leaf": hash_leaf(preimage),
    }


def _receipt_digest(body: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in body.items() if key != "receipt_sha256"}
    return hash_bytes(canonical_json_bytes(payload))


def seal_kernel_lineage(root: Path | str | None = None) -> dict[str, Any]:
    """Seal the live kernel files. Raises if a member is missing or a symlink."""
    base = default_repo_root() if root is None else Path(root)
    parts = [_module_parts(base, rel) for rel in KERNEL_PATHS]
    # merkle_root applies the RFC 6962 leaf prefix itself. Pass preimages,
    # not leaves that were already hashed, or every node is hashed twice.
    root_bytes = merkle_root([item["preimage"] for item in parts])
    body: dict[str, Any] = {
        "kind": KIND,
        "schema": SCHEMA,
        "research_only": True,
        "data_label": "SOURCE",
        "claim": {
            "algorithm": "rfc6962-sha256",
            "domain": DOMAIN.decode("ascii"),
            "n_modules": len(parts),
            "merkle_root": root_bytes.hex(),
            "modules": [
                {
                    "path": item["path"],
                    "byte_sha256": item["byte_sha256"],
                    "structural_sha256": item["structural_sha256"],
                }
                for item in parts
            ],
        },
        "interpretation": (
            "Binds the numeric kernels to this tree. A witness copied onto "
            "different kernel sources does not verify. This is source identity, "
            "not a market result."
        ),
    }
    body["receipt_sha256"] = _receipt_digest(body)
    return body


def _modules_by_path(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]] | None:
    claim = receipt.get("claim")
    if not isinstance(claim, Mapping):
        return None
    modules = claim.get("modules")
    if not isinstance(modules, list):
        return None
    out: dict[str, Mapping[str, Any]] = {}
    for item in modules:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            return None
        out[str(item["path"])] = item
    return out


def _drifted_paths(receipt: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[str]:
    left = _modules_by_path(receipt)
    right = _modules_by_path(fresh)
    if left is None or right is None:
        return list(KERNEL_PATHS)
    drifted: list[str] = []
    for rel in KERNEL_PATHS:
        a = left.get(rel)
        b = right.get(rel)
        if a is None or b is None:
            drifted.append(rel)
            continue
        if a.get("byte_sha256") != b.get("byte_sha256") or a.get("structural_sha256") != b.get(
            "structural_sha256"
        ):
            drifted.append(rel)
    return drifted


def verify_kernel_lineage(
    receipt: Mapping[str, Any],
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Recompute the seal and report drifted kernel paths. Does not raise."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return {"ok": False, "errors": ["receipt_not_a_mapping"], "drifted": []}
    if receipt.get("schema") != SCHEMA or receipt.get("kind") != KIND:
        errors.append("schema")
    if receipt.get("research_only") is not True:
        errors.append("research_only")
    if receipt.get("data_label") != "SOURCE":
        errors.append("data_label")
    try:
        fresh = seal_kernel_lineage(root)
    except ValueError as exc:
        errors.append(f"unreadable:{exc}")
        return {"ok": False, "errors": errors, "drifted": list(KERNEL_PATHS)}
    drifted = _drifted_paths(receipt, fresh)
    if drifted or receipt.get("claim", {}).get("merkle_root") != fresh["claim"]["merkle_root"]:
        errors.append("kernel_drift")
    if receipt.get("receipt_sha256") != _receipt_digest(receipt):
        errors.append("receipt_sha256")
    return {"ok": not errors, "errors": errors, "drifted": drifted}


def compare_kernel_lineage(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Separate comment-level byte drift from syntax drift between two seals."""
    a = _modules_by_path(left)
    b = _modules_by_path(right)
    if a is None or b is None:
        raise ValueError("kernel lineage seal is missing its module list")
    byte_drift: list[str] = []
    structural_drift: list[str] = []
    for rel in KERNEL_PATHS:
        if rel not in a or rel not in b:
            byte_drift.append(rel)
            structural_drift.append(rel)
            continue
        if a[rel].get("byte_sha256") != b[rel].get("byte_sha256"):
            byte_drift.append(rel)
        if a[rel].get("structural_sha256") != b[rel].get("structural_sha256"):
            structural_drift.append(rel)
    left_root = (
        left.get("claim", {}).get("merkle_root") if isinstance(left.get("claim"), Mapping) else None
    )
    right_root = (
        right.get("claim", {}).get("merkle_root")
        if isinstance(right.get("claim"), Mapping)
        else None
    )
    return {
        "same_root": left_root == right_root and left_root is not None,
        "n_byte_drift": len(byte_drift),
        "n_structural_drift": len(structural_drift),
        "byte_drift": byte_drift,
        "structural_drift": structural_drift,
    }


def encode_kernel_witness(root: Path | str | None = None) -> bytes:
    """Binary witness: magic, root, per-kernel inclusion proof, SHA-256 trailer."""
    base = default_repo_root() if root is None else Path(root)
    parts = [_module_parts(base, rel) for rel in KERNEL_PATHS]
    preimages = [item["preimage"] for item in parts]
    root_bytes = merkle_root(preimages)
    body = bytearray()
    body += _MAGIC
    body += _VERSION.to_bytes(2, "big")
    body += len(parts).to_bytes(4, "big")
    body += root_bytes
    for index, item in enumerate(parts):
        proof = inclusion_proof(preimages, index)
        path = item["path"].encode("utf-8")
        body += len(path).to_bytes(2, "big")
        body += path
        body += bytes.fromhex(item["byte_sha256"])
        body += bytes.fromhex(item["structural_sha256"])
        body += item["leaf"]
        body += len(proof).to_bytes(2, "big")
        for node in proof:
            body += node
    return bytes(body) + hashlib.sha256(body).digest()


def _parse_module(cursor: _Cursor) -> KernelWitnessModule:
    path = cursor.take(cursor.u16()).decode("utf-8")
    byte_digest = cursor.take(32)
    structural = cursor.take(32)
    leaf = cursor.take(32)
    proof_len = cursor.u16()
    if proof_len > _MAX_PROOF_NODES:
        raise ValueError("kernel witness proof is too long")
    proof = tuple(cursor.take(32) for _ in range(proof_len))
    return KernelWitnessModule(path, byte_digest, structural, leaf, proof)


def parse_kernel_witness(blob: bytes) -> KernelWitness:
    """Parse a witness. A bad trailer, magic, version, or length raises."""
    if len(blob) < 32:
        raise ValueError("kernel witness truncated")
    body, trailer = blob[:-32], blob[-32:]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), trailer):
        raise ValueError("kernel witness trailer mismatch")
    cursor = _Cursor(body)
    if cursor.take(4) != _MAGIC:
        raise ValueError("kernel witness magic mismatch")
    if cursor.u16() != _VERSION:
        raise ValueError("kernel witness version mismatch")
    count = cursor.u32()
    if count != len(KERNEL_PATHS):
        raise ValueError("kernel witness module count mismatch")
    root = cursor.take(32)
    modules = tuple(_parse_module(cursor) for _ in range(count))
    if not cursor.finished():
        raise ValueError("kernel witness has trailing bytes")
    if tuple(item.path for item in modules) != KERNEL_PATHS:
        raise ValueError("kernel witness paths are not the kernel set")
    return KernelWitness(root=root, modules=modules)


def verify_kernel_witness(blob: bytes, root: Path | str | None = None) -> dict[str, Any]:
    """Check a witness against the live kernel files. Does not raise."""
    try:
        parsed = parse_kernel_witness(blob)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)], "drifted": []}
    try:
        parts = [
            _module_parts(default_repo_root() if root is None else Path(root), rel)
            for rel in KERNEL_PATHS
        ]
    except ValueError as exc:
        return {"ok": False, "errors": [f"unreadable:{exc}"], "drifted": list(KERNEL_PATHS)}
    preimages = [item["preimage"] for item in parts]
    live_root = merkle_root(preimages)
    errors: list[str] = []
    drifted: list[str] = []
    if not hmac.compare_digest(parsed.root, live_root):
        errors.append("merkle_root")
    for index, (parsed_mod, live) in enumerate(zip(parsed.modules, parts, strict=True)):
        byte_digest = bytes.fromhex(live["byte_sha256"])
        structural = bytes.fromhex(live["structural_sha256"])
        if parsed_mod.byte_digest != byte_digest or parsed_mod.structural_digest != structural:
            drifted.append(live["path"])
        if parsed_mod.leaf_hash != live["leaf"] or not verify_inclusion(
            live["leaf"], index, list(parsed_mod.proof), len(preimages), live_root
        ):
            errors.append(f"inclusion:{live['path']}")
    if drifted:
        errors.append("kernel_drift")
    return {"ok": not errors, "errors": errors, "drifted": drifted}


__all__ = [
    "DOMAIN",
    "KERNEL_PATHS",
    "KIND",
    "SCHEMA",
    "KernelWitness",
    "KernelWitnessModule",
    "compare_kernel_lineage",
    "default_repo_root",
    "encode_kernel_witness",
    "leaf_preimage",
    "parse_kernel_witness",
    "seal_kernel_lineage",
    "structural_digest",
    "verify_kernel_lineage",
    "verify_kernel_witness",
]
