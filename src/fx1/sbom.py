"""SBOM generation for fx-1 releases (SPDX-lite, CycloneDX-compatible shape).

An auditor asks "what exactly is in this thing?" — the SBOM answers with
locked dependency names/versions/hashes from uv.lock plus the repo's own
artifact hashes. Generated, never hand-maintained.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from fx1 import __version__


class SBOMEntry(BaseModel):
    name: str
    version: str
    sha256: str | None = None


class SBOM(BaseModel):
    spec: str = "spdx-lite/2.3"
    generated_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    package: str = "fx-1"
    entries: list[SBOMEntry]
    lockfile_sha256: str = Field(min_length=64, max_length=64)


def _package_entry(package: object) -> SBOMEntry:
    """Read one TOML package without mistaking nested metadata for its identity."""
    if not isinstance(package, dict):
        raise ValueError("malformed [[package]] block in lockfile")
    name, version = package.get("name"), package.get("version")
    # uv omits the version of the editable project when pyproject declares
    # dynamic versioning. Only our root package may use the canonical release
    # version; an unversioned dependency must still fail closed.
    if version is None and name == "fx-1" and package.get("source") == {"editable": "."}:
        version = __version__
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError(
            "malformed [[package]] block in lockfile: name without version "
            "or vice versa — refusing to emit an incomplete SBOM"
        )
    return SBOMEntry(name=name, version=version, sha256=_package_hash(package))


def _package_hash(package: dict[str, object]) -> str | None:
    """Keep the existing single-artifact hash convention (sdist, then wheels)."""
    wheels = package.get("wheels", [])
    if not isinstance(wheels, list):
        raise ValueError("malformed wheels in lockfile package")
    artifacts = [package.get("sdist"), *wheels]
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        digest = artifact.get("hash")
        if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            return digest.removeprefix("sha256:")
    return None


def generate_sbom(lockfile: str | Path = "uv.lock") -> SBOM:
    """Parse uv.lock into an SBOM for the current fx-1 release.

    Locked dependencies retain their recorded versions. The dynamically
    versioned editable fx-1 root uses ``fx1.__version__`` when uv omits it.
    """
    path = Path(lockfile)
    if not path.exists():
        raise FileNotFoundError(f"lockfile not found: {path}")
    raw = path.read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"no packages parsed from {path}: invalid TOML lockfile") from exc
    packages = document.get("package", [])
    if not isinstance(packages, list) or not packages:
        raise ValueError(f"no packages parsed from {path}")
    entries = [_package_entry(package) for package in packages]
    return SBOM(
        entries=sorted(entries, key=lambda e: e.name),
        lockfile_sha256=hashlib.sha256(raw).hexdigest(),
    )
