"""Fail when a runtime image's Python distributions drift from ``uv.lock``.

The image is built with ``uv sync --frozen --no-dev``. Every installed
distribution must be the locked version. Every requirement ``uv export
--frozen --no-dev`` emits must be installed. The local project (the lock
entry whose source is the checkout, not a registry sdist) has a dynamic
version, so its presence is required and its version is not compared.

Stdlib only. Images are read with ``docker run --entrypoint python``; set
``DOCKER`` when the client is not ``docker`` on ``PATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_NAME = re.compile(r"[-_.]+")
_DIST_SCRIPT = (
    "import importlib.metadata, json; "
    "print(json.dumps([[d.metadata['Name'], d.version] "
    "for d in importlib.metadata.distributions()]))"
)


def canonicalize(name: str) -> str:
    """PEP 503 name: lowercase, runs of ``-_.`` collapsed to one hyphen."""
    return _NAME.sub("-", name).lower()


def locked_versions(lock_text: str) -> tuple[dict[str, str], set[str]]:
    """Return ``(name -> version, local project names)`` from a uv lock."""
    document = tomllib.loads(lock_text)
    versions: dict[str, str] = {}
    local: set[str] = set()
    packages = document.get("package", [])
    if not isinstance(packages, list):
        raise ValueError("uv.lock package table is not a list")
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("uv.lock package entry is not a table")
        raw_name = package.get("name")
        if not isinstance(raw_name, str):
            raise ValueError("uv.lock package is missing a name")
        name = canonicalize(raw_name)
        source = package.get("source")
        if isinstance(source, dict) and _local_source(source):
            local.add(name)
        version = package.get("version")
        if isinstance(version, str):
            versions[name] = version
    return versions, local


def _local_source(source: dict[str, object]) -> bool:
    return any(key in source for key in ("editable", "virtual", "path"))


def parse_requirements(text: str) -> dict[str, str]:
    """Parse ``uv export`` lines. Markers that are false here are skipped."""
    required: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            raise ValueError(f"unparsed requirement: {line}")
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate():
            continue
        pinned = [spec.version for spec in req.specifier if spec.operator == "=="]
        if len(pinned) != 1:
            raise ValueError(f"unparsed requirement: {line}")
        required[canonicalize(req.name)] = pinned[0]
    return required


def _installed_problem(
    name: str,
    version: str,
    locked: dict[str, str],
    required: dict[str, str],
) -> str | None:
    expected = locked.get(name)
    if expected is None:
        return f"image has {name}=={version} which is not in uv.lock"
    if name not in required:
        return f"image has {name}=={version} outside the runtime lock export"
    if expected != version:
        return f"{name} image={version} lock={expected}"
    return None


def lock_mismatches(
    installed: dict[str, str],
    locked: dict[str, str],
    local_projects: set[str],
    required: dict[str, str],
) -> list[str]:
    """Names whose image version disagrees with the lock, or that are absent."""
    problems: list[str] = []
    normalized = {canonicalize(name): version for name, version in installed.items()}
    for name in sorted(local_projects):
        if name not in normalized:
            problems.append(f"image is missing the local project {name}")
    for name, version in sorted(normalized.items()):
        if name in local_projects:
            continue
        problem = _installed_problem(name, version, locked, required)
        if problem is not None:
            problems.append(problem)
    for name, version in sorted(required.items()):
        if name not in normalized:
            problems.append(f"image is missing locked requirement {name}=={version}")
    return problems


def image_distributions(image: str, docker: str | None = None) -> dict[str, str]:
    """Read ``Name``/``Version`` metadata from an image. No package is executed."""
    client = docker if docker is not None else os.environ.get("DOCKER", "docker")
    completed = subprocess.run(
        [
            client,
            "run",
            "--rm",
            "--entrypoint",
            "python",
            "--network",
            "none",
            image,
            "-c",
            _DIST_SCRIPT,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"could not read distributions from {image}: {detail}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError(f"distribution listing from {image} is not a list")
    installed: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError(f"distribution listing from {image} has a bad row")
        name, version = item
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError(f"distribution listing from {image} has a bad row")
        installed[canonicalize(name)] = version
    return installed


def export_requirements(root: Path) -> str:
    """Locked runtime requirements for this platform, without the local project."""
    completed = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"uv export failed: {detail}")
    return completed.stdout


def check_image(
    image: str,
    lock_text: str,
    requirements_text: str,
    docker: str | None = None,
) -> list[str]:
    locked, local = locked_versions(lock_text)
    required = parse_requirements(requirements_text)
    installed = image_distributions(image, docker=docker)
    return lock_mismatches(installed, locked, local, required)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True, help="Image tag to read")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("uv.lock"),
        help="Path to uv.lock",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root for uv export")
    args = parser.parse_args(argv)
    lock_text = args.lock.read_text(encoding="utf-8")
    requirements = export_requirements(args.root)
    failed = False
    for image in args.image:
        problems = check_image(image, lock_text, requirements)
        if problems:
            failed = True
            print(f"{image}: python deps do not match uv.lock")
            for problem in problems:
                print(f"  {problem}")
        else:
            print(f"{image}: python deps match uv.lock")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
