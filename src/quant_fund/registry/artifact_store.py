"""artifact_store — content-addressed evidence store for receipts/artifacts.

``mlflow_store`` tracks runs in an external service; this is the offline
complement: a directory of sha256-addressed blobs plus a hash-chained
append-only index — the same fail-closed discipline as the audit ledger
and corpus epoch chains.

Layout::

    root/objects/<sha[:2]>/<sha[2:]>   immutable blob (atomic create)
    root/index.jsonl                    one record per store():
        {"sha256", "kind", "bytes", "prev_index_sha256"}

- ``store`` dedups objects by digest but always appends an index record,
  so the index is a complete history of what was committed when.
- ``fetch`` re-hashes the blob before returning it — bit drift on disk
  fails closed instead of returning corrupt bytes.
- ``verify`` walks the index (chain intact, fields well-formed) and
  re-hashes every referenced object, plus flags objects on disk that no
  index record points at.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

INDEX_NAME = "index.jsonl"
OBJECTS_DIR = "objects"
GENESIS = "0" * 64


class ArtifactStoreError(RuntimeError):
    """Raised on integrity failures (tamper, chain break, missing object)."""


def _object_path(root: Path, sha: str) -> Path:
    return root / OBJECTS_DIR / sha[:2] / sha[2:]


def _append_index(root: Path, record: dict[str, Any]) -> None:
    line = (canonical_json_bytes(record) + b"\n").decode("utf-8")
    index_path = root / INDEX_NAME
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _index_sha256(root: Path) -> str:
    index_path = root / INDEX_NAME
    if not index_path.exists():
        return GENESIS
    return hash_bytes(index_path.read_bytes())


def _read_index(root: Path) -> list[dict[str, Any]]:
    """Parse index records; each carries ``raw_bytes`` (line + newline) so the
    byte-level chain is replayable."""
    index_path = root / INDEX_NAME
    if not index_path.exists():
        return []
    raw = index_path.read_bytes()
    records: list[dict[str, Any]] = []
    for i, line_bytes in enumerate(raw.splitlines(keepends=True)):
        if not line_bytes.strip():
            continue
        try:
            rec = json.loads(line_bytes)
        except json.JSONDecodeError as exc:
            raise ArtifactStoreError(f"index line {i} is not valid JSON") from exc
        rec["__raw_bytes"] = line_bytes
        records.append(rec)
    return records


class ArtifactStore:
    """Filesystem content-addressed store rooted at ``root``."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def store(self, payload: bytes, kind: str = "blob") -> str:
        """Write ``payload`` under its sha256; append an index record.

        Returns the digest. Object writes are atomic (tmp + rename); a
        pre-existing identical object is left untouched.
        """
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not kind or not isinstance(kind, str):
            raise ValueError("kind must be a non-empty string")
        sha = hash_bytes(payload)
        obj = _object_path(self._root, sha)
        if not obj.exists():
            obj.parent.mkdir(parents=True, exist_ok=True)
            tmp = obj.parent / f".{obj.name}.tmp-{os.getpid()}"
            tmp.write_bytes(payload)
            os.replace(tmp, obj)
        elif obj.read_bytes() != payload:
            raise ArtifactStoreError(
                f"sha256 collision or corruption at {obj.name}: same digest, different bytes"
            )
        record = {
            "sha256": sha,
            "kind": kind,
            "bytes": len(payload),
            "prev_index_sha256": _index_sha256(self._root),
        }
        _append_index(self._root, record)
        return sha

    def fetch(self, sha256: str) -> bytes:
        """Return the blob's bytes after re-verifying its digest."""
        obj = _object_path(self._root, sha256)
        if not obj.exists():
            raise ArtifactStoreError(f"object {sha256} not present in store")
        data = obj.read_bytes()
        if hash_bytes(data) != sha256:
            raise ArtifactStoreError(f"object {sha256} content drifted on disk")
        return data

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Index records, optionally filtered by kind."""
        records = _read_index(self._root)
        out = [{k: v for k, v in rec.items() if k != "__raw_bytes"} for rec in records]
        for rec in out:
            if "sha256" not in rec:
                raise ArtifactStoreError("index record missing sha256")
        if kind is None:
            return out
        return [r for r in out if r.get("kind") == kind]

    def verify(self) -> dict[str, Any]:
        """Full audit: index well-formed + chained, objects re-hashed.

        Returns a report; ``ok`` is False if any check failed. Raises
        ArtifactStoreError only for an unreadable/corrupt index itself.
        """
        records = _read_index(self._root)
        errors: list[str] = []
        seen: set[str] = set()
        prev = GENESIS
        running = b""
        n_objects = 0
        for i, rec in enumerate(records):
            sha = rec.get("sha256")
            kind = rec.get("kind")
            nbytes = rec.get("bytes")
            chain = rec.get("prev_index_sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                errors.append(f"index[{i}]: bad sha256")
                continue
            if not isinstance(kind, str) or not kind:
                errors.append(f"index[{i}]: bad kind")
            if not isinstance(nbytes, int) or nbytes < 0:
                errors.append(f"index[{i}]: bad bytes field")
            if chain != prev:
                errors.append(f"index[{i}]: chain break (prev={chain}, expected {prev})")
            obj = _object_path(self._root, sha)
            if sha not in seen:
                if not obj.exists():
                    errors.append(f"index[{i}]: object {sha[:12]} missing on disk")
                else:
                    data = obj.read_bytes()
                    if hash_bytes(data) != sha:
                        errors.append(f"index[{i}]: object {sha[:12]} digest drift")
                    elif isinstance(nbytes, int) and len(data) != nbytes:
                        errors.append(f"index[{i}]: object {sha[:12]} size mismatch")
            seen.add(sha)
            running += rec["__raw_bytes"]
            prev = hash_bytes(running)
        # orphan objects: on disk but unreferenced
        objects_root = self._root / OBJECTS_DIR
        orphans = 0
        if objects_root.exists():
            for p in objects_root.rglob("*"):
                if p.is_file() and not p.name.startswith("."):
                    full = p.parent.name + p.name
                    if full not in seen:
                        orphans += 1
                        errors.append(f"orphan object on disk: {full[:16]}...")
        n_objects = len(seen)
        return {
            "ok": not errors,
            "n_index_records": len(records),
            "n_objects": n_objects,
            "n_orphans": orphans,
            "errors": errors,
        }


__all__ = ["ArtifactStore", "ArtifactStoreError", "GENESIS", "INDEX_NAME", "OBJECTS_DIR"]
