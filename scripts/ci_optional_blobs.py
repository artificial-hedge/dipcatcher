#!/usr/bin/env python3
"""Build the CI optional-blob pickle once and reuse it across shards.

Pytest-split runs four shards at once. Each shard has its own disk, and the
workflow cache of ``~/.cache/dipcatcher`` is saved only after a job finishes,
so the four shards all miss and each spends a cold fill building the same
``ci.pkl``. The loader reads only ``ci-optional/<stamp>/ci.pkl``. A restored
older tree does not count: the stamp changes when ``quant_fund`` sources,
``pyproject.toml``, ``uv.lock``, or the interpreter change.

This script fills that file once. ``pack`` writes a tarball whose name is the
stamp; ``install`` extracts it only when the stamp still matches this tree.
Production budgets never call it. Do not point it at a live trading host.

The GitHub Actions workflow is intentionally unchanged (``runs-on`` stays
``instabox-large``). Wire the commands in front of the shard matrix when you
are ready; see ``docs/CI_OPTIONAL_BLOBS.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

_ARCHIVE_PREFIX = "ci-optional"


def _stamp() -> str:
    from quant_fund.research.ci_blob_cache import source_stamp

    return source_stamp()


def _cache_base() -> Path:
    override = os.environ.get("DIP_SESSION_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "dipcatcher"


def cmd_stamp() -> int:
    print(_stamp())
    return 0


def cmd_build() -> int:
    """Fill ``ci.pkl`` for this stamp. A hit returns without rebuilding."""
    os.environ["DIP_SCORECARD_BUDGET"] = "ci"
    os.environ.pop("DIP_DISABLE_SESSION_CACHE", None)
    from quant_fund.research.ci_blob_cache import load_blobs

    stamp = _stamp()
    loaded = load_blobs("ci")
    if isinstance(loaded, dict) and loaded:
        print(f"hit {stamp} keys {len(loaded)}")
        return 0
    from quant_fund.research.agent import _live_optional_family_blobs

    blobs = _live_optional_family_blobs()
    written = _cache_base() / _ARCHIVE_PREFIX / stamp / "ci.pkl"
    if not written.is_file():
        print(f"build did not write {written}", file=sys.stderr)
        return 1
    print(f"built {stamp} keys {len(blobs)} path {written}")
    return 0


def _member_ok(name: str, stamp: str) -> bool:
    parts = Path(name).parts
    if not parts or parts[0] == "/" or ".." in parts:
        return False
    expected = (_ARCHIVE_PREFIX, stamp, "ci.pkl")
    return parts == expected


def cmd_pack(out: Path) -> int:
    from quant_fund.research.ci_blob_cache import blob_dir

    stamp = _stamp()
    source = blob_dir() / "ci.pkl"
    if not source.is_file():
        print(f"missing {source}; run build first", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    arcname = f"{_ARCHIVE_PREFIX}/{stamp}/ci.pkl"
    with tarfile.open(out, "w:gz") as archive:
        archive.add(source, arcname=arcname)
    print(f"packed {stamp} {out}")
    return 0


def cmd_install(archive_path: Path) -> int:
    if not archive_path.is_file():
        print(f"missing archive {archive_path}", file=sys.stderr)
        return 2
    stamp = _stamp()
    base = _cache_base()
    dest_dir = base / _ARCHIVE_PREFIX / stamp
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != 1 or not _member_ok(members[0].name, stamp):
            print(
                f"archive stamp does not match {stamp}; refusing to install",
                file=sys.stderr,
            )
            return 2
        if not members[0].isfile():
            print("archive member is not a regular file", file=sys.stderr)
            return 2
        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted = archive.extractfile(members[0])
        if extracted is None:
            print("archive member is empty", file=sys.stderr)
            return 2
        target = dest_dir / "ci.pkl"
        tmp = dest_dir / f".ci.pkl.{os.getpid()}.tmp"
        try:
            with tmp.open("wb") as handle:
                handle.write(extracted.read())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"installed {stamp} {dest_dir / 'ci.pkl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stamp", help="print the cache stamp for this tree and interpreter")
    sub.add_parser("build", help="build ci.pkl if this stamp is not already cached")
    pack = sub.add_parser("pack", help="tar the current stamp's ci.pkl")
    pack.add_argument("--out", type=Path, required=True)
    install = sub.add_parser("install", help="install a tarball if its stamp matches")
    install.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "stamp":
        return cmd_stamp()
    if args.cmd == "build":
        return cmd_build()
    if args.cmd == "pack":
        return cmd_pack(args.out)
    return cmd_install(args.archive)


if __name__ == "__main__":
    raise SystemExit(main())
