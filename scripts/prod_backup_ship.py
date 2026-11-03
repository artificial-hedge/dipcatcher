#!/usr/bin/env python3
"""Write one verified paper-state stamp and copy it off the host.

The stamp is ``BACKUP_ROOT/<UTC>``. That root is not ``BACKUP_DIR``: ``prod-backup``
uses ``BACKUP_DIR`` as the archive itself. A root that already holds
``manifest.json`` is refused so a stamp is not nested inside an archive.
Every payload is rehashed before a copy. ``DIPCATCHER_BACKUP_COPY`` is
``none``, ``rsync``, or ``s3``. A failed verify does not copy. Previous
stamps stay. This does not stop the writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from quant_fund.paper.state_backup import SCHEMA, PaperBackupError, backup_paper_state

_STAMP = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
_CHUNK = 1 << 20
Runner = Callable[[Sequence[str]], object]


def _positive_part(part: str) -> bool:
    return part.startswith("dipcatcher")


def locate(path: Path) -> Path | None:
    """Absolute dipcatcher path. Symlinks and path escapes are refused."""
    if not path.is_absolute() or path.is_symlink():
        return None
    resolved = path.resolve()
    if path.exists() and resolved != path.absolute():
        return None
    if not any(_positive_part(part) for part in resolved.parts):
        return None
    text = resolved.as_posix().rstrip("/")
    if text in {"/tmp", "/var", "/var/lib", "/var/backups", "/var/log"}:
        return None
    allowed = (
        text.startswith("/tmp/dipcatcher")
        or text.startswith("/var/lib/dipcatcher")
        or text.startswith("/var/log/dipcatcher")
        or text.startswith("/var/backups/")
    )
    if not allowed:
        return None
    return resolved


def rsync_target_ok(target: str) -> bool:
    """Local absolute path, or ``user@host:/absolute``."""
    if not target or target.startswith("-") or any(char.isspace() for char in target):
        return False
    if ".." in target.split("/"):
        return False
    if "@" not in target and ":" not in target:
        return target.startswith("/")
    if target.count("@") != 1 or target.count(":") != 1:
        return False
    user, rest = target.split("@", 1)
    host, dest = rest.split(":", 1)
    if not user or not host or ":" in user or user.startswith("-") or host.startswith("-"):
        return False
    return dest.startswith("/") and ".." not in dest.split("/")


def s3_uri_ok(uri: str) -> bool:
    """``s3://bucket`` or ``s3://bucket/prefix``. No spaces, no ``@``."""
    if not uri.startswith("s3://") or "@" in uri or any(char.isspace() for char in uri):
        return False
    if ".." in uri.split("/"):
        return False
    rest = uri[len("s3://") :]
    if not rest or rest.startswith("/"):
        return False
    bucket, _sep, _key = rest.partition("/")
    return bool(bucket) and ":" not in bucket and not bucket.startswith("-")


def copy_plan(env: Mapping[str, str]) -> tuple[str, str] | None:
    """``(mode, target)``. ``None`` when the mode or target is refused."""
    mode = env.get("DIPCATCHER_BACKUP_COPY", "").strip() or "none"
    if mode == "none":
        return ("none", "")
    if mode == "rsync":
        target = env.get("DIPCATCHER_BACKUP_RSYNC_TARGET", "").strip()
        if rsync_target_ok(target):
            return ("rsync", target)
        return None
    if mode == "s3":
        uri = env.get("DIPCATCHER_BACKUP_S3_URI", "").strip()
        if s3_uri_ok(uri):
            return ("s3", uri)
        return None
    return None


def _hash_file(path: Path) -> tuple[str, int] | None:
    if path.is_symlink() or not path.is_file():
        return None
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return None
    return digest.hexdigest(), size


def _entry_ok(archive: Path, entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    relative = entry.get("path")
    digest = entry.get("sha256")
    size = entry.get("size")
    if not isinstance(relative, str) or not isinstance(digest, str):
        return False
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return False
    if not relative or relative.startswith("/") or ".." in relative.split("/"):
        return False
    payload = archive / "payload" / relative
    if payload.is_symlink():
        return False
    found = _hash_file(payload)
    return found == (digest, size)


def verify_archive(archive: Path) -> dict[str, object]:
    """Rehash every payload named by ``manifest.json``."""
    manifest = archive / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        return {"error": "schema", "ok": False, "verified": False}
    try:
        raw = manifest.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"error": "schema", "ok": False, "verified": False}
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return {"error": "schema", "ok": False, "verified": False}
    files = payload.get("files")
    if not isinstance(files, list) or not all(_entry_ok(archive, entry) for entry in files):
        return {"error": "hash_mismatch", "ok": False, "verified": False}
    return {
        "archive": str(archive),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "ok": True,
        "verified": True,
    }


def rsync_argv(source: Path, target: str, stamp: str, *, dry: bool) -> list[str]:
    dest = f"{target.rstrip('/')}/{stamp}/"
    argv = ["rsync", "-a", "--checksum"]
    if dry:
        argv.extend(["--dry-run", "--itemize-changes"])
    argv.extend([f"{source}/", dest])
    return argv


def s3_sync_argv(source: Path, uri: str, stamp: str) -> list[str]:
    return [
        "aws",
        "s3",
        "sync",
        str(source),
        f"{uri.rstrip('/')}/{stamp}",
        "--only-show-errors",
    ]


def s3_manifest_argv(uri: str, stamp: str) -> list[str]:
    return ["aws", "s3", "cp", f"{uri.rstrip('/')}/{stamp}/manifest.json", "-"]


def _code(result: object) -> int:
    code = getattr(result, "returncode", 1)
    return code if isinstance(code, int) else 1


def _text(result: object) -> str:
    stdout = getattr(result, "stdout", "")
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", "replace")
    return stdout if isinstance(stdout, str) else ""


def _copy(mode: str, target: str, archive: Path, stamp: str, run: Runner) -> str | None:
    if mode == "none":
        return None
    if mode == "rsync":
        copied = run(rsync_argv(archive, target, stamp, dry=False))
        if _code(copied) != 0:
            return "rsync"
        checked = run(rsync_argv(archive, target, stamp, dry=True))
        if _code(checked) != 0 or _text(checked).strip():
            return "rsync drift"
        return None
    synced = run(s3_sync_argv(archive, target, stamp))
    if _code(synced) != 0:
        return "s3"
    fetched = run(s3_manifest_argv(target, stamp))
    local = (archive / "manifest.json").read_text(encoding="utf-8")
    if _code(fetched) != 0 or _text(fetched) != local:
        return "s3 manifest"
    return None


def ship(
    source: Path,
    root: Path,
    env: Mapping[str, str],
    run: Runner,
    stamp: str = "",
    verify: Callable[[Path], dict[str, object]] = verify_archive,
) -> dict[str, object]:
    """Back up ``source`` under ``root`` and copy only after ``verify``."""
    plan = copy_plan(env)
    if plan is None:
        return {"error": "copy", "ok": False, "verified": False}
    mode, target = plan
    located_source = locate(source)
    located_root = locate(root)
    if located_source is None or located_root is None:
        return {"error": "path", "ok": False, "verified": False}
    if (located_root / "manifest.json").exists():
        return {"error": "archive_root", "ok": False, "verified": False}
    chosen = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not _STAMP.fullmatch(chosen):
        return {"error": "stamp", "ok": False, "verified": False}
    dest = located_root / chosen
    if dest.exists() or dest.is_symlink():
        return {"error": "dest_exists", "ok": False, "verified": False}
    try:
        located_root.mkdir(parents=True, exist_ok=True)
        backup_paper_state(located_source, dest)
    except PaperBackupError as exc:
        return {"error": str(exc), "ok": False, "verified": False}
    except OSError:
        return {"error": "io", "ok": False, "verified": False}
    checked = verify(dest)
    if not checked.get("ok"):
        checked["copy"] = mode
        return checked
    failure = _copy(mode, target, dest, chosen, run)
    if failure is not None:
        return {**checked, "copy": mode, "error": failure, "ok": False}
    return {**checked, "copy": mode, "ok": True}


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ship one verified paper-state stamp")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stamp", default="")
    args = parser.parse_args(argv)
    report = ship(args.source, args.root, os.environ, _run, args.stamp)
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
