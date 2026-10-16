#!/usr/bin/env python3
"""Require same-commit CI evidence before a release job may sign artifacts."""

from __future__ import annotations

import ast
import json
import os
import re
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class ReleaseGateError(RuntimeError):
    """The tag does not have the required release evidence."""


_REQUIRED = (
    ("ci.yml", "ci-required", "push"),
    ("fx1.yml", "fx1-required", "push"),
    ("proofcore.yml", "proofcore-required", "push"),
    ("matrix.yml", None, None),
)
_PLATFORMS = ("ubuntu-latest", "macos-latest", "windows-latest")
_PYTHONS = ("3.12", "3.13", "3.14")
_MATRIX_JOBS = {
    name
    for os_name in _PLATFORMS
    for python in _PYTHONS
    for name in (
        f"{os_name} / py{python} / inventory",
        *(f"{os_name} / py{python} / shard{shard}" for shard in range(4)),
    )
}
_VERSION_SOURCE = Path("src/fx1/__init__.py")
_TAG_REF = re.compile(
    r"refs/tags/v(?P<version>(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?)\Z"
)


def verify_tag_version(ref: str, root: Path) -> str:
    """Match an exact SemVer tag to Hatch's literal fx1 version source."""

    match = _TAG_REF.fullmatch(ref)
    if match is None:
        raise ReleaseGateError(f"release ref is not a SemVer tag: {ref!r}")
    tag_version = match.group("version")
    prerelease = tag_version.split("+", 1)[0].split("-", 1)
    if len(prerelease) == 2 and any(
        part.isascii() and part.isdigit() and len(part) > 1 and part[0] == "0"
        for part in prerelease[1].split(".")
    ):
        raise ReleaseGateError(f"release tag has a noncanonical numeric prerelease: {ref!r}")

    try:
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseGateError(f"cannot read Hatch version configuration: {exc}") from exc
    project = config.get("project", {})
    if (
        "version" not in project.get("dynamic", [])
        or "version" in project
        or config.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
        != _VERSION_SOURCE.as_posix()
    ):
        raise ReleaseGateError("pyproject must derive its version from fx1.__version__")
    try:
        source = ast.parse((root / _VERSION_SOURCE).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ReleaseGateError(f"cannot read fx1 version source: {exc}") from exc
    assignments = [
        node
        for node in ast.walk(source)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        )
        or (
            isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        )
    ]
    if (
        len(assignments) != 1
        or assignments[0] not in source.body
        or not isinstance(assignments[0], ast.Assign)
        or len(assignments[0].targets) != 1
        or not isinstance(assignments[0].targets[0], ast.Name)
        or not isinstance(assignments[0].value, ast.Constant)
    ):
        raise ReleaseGateError("fx1.__version__ must be one literal string assignment")
    package_version = assignments[0].value.value
    if not isinstance(package_version, str) or package_version != tag_version:
        raise ReleaseGateError(
            f"release tag version {tag_version!r} does not match fx1.__version__ {package_version!r}"
        )
    return package_version


def _api(repo: str, path: str, token: str) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (OSError, ValueError) as exc:
        raise ReleaseGateError(f"GitHub API request failed for {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ReleaseGateError(f"GitHub API returned a non-object for {path}")
    return result


def _complete_list(data: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    rows = data.get(key)
    if not isinstance(rows, list) or not isinstance(data.get("total_count"), int):
        raise ReleaseGateError(f"{label}: malformed GitHub API response")
    if data["total_count"] > len(rows):
        raise ReleaseGateError(f"{label}: GitHub API response was paginated")
    if not all(isinstance(row, dict) for row in rows):
        raise ReleaseGateError(f"{label}: malformed list entry")
    return rows


def _workflow_run(
    repo: str, sha: str, token: str, workflow: str, event: str | None
) -> dict[str, Any]:
    params = {"head_sha": sha, "per_page": 100}
    if event:
        params["event"] = event
        params["branch"] = "main"
    query = urllib.parse.urlencode(params)
    data = _api(repo, f"actions/workflows/{workflow}/runs?{query}", token)
    runs = _complete_list(data, "workflow_runs", workflow)
    candidates = [
        run
        for run in runs
        if run.get("head_sha") == sha
        and run.get("event") in ({event} if event else {"schedule", "workflow_dispatch"})
        and (event is None or run.get("head_branch") == "main")
    ]
    if not candidates:
        raise ReleaseGateError(f"{workflow}: no same-SHA run with the required event")
    latest = max(
        candidates,
        key=lambda run: (
            str(run.get("updated_at") or run.get("created_at") or ""),
            int(run.get("id", 0)),
        ),
    )
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ReleaseGateError(
            f"{workflow}: latest same-SHA run {latest.get('id')} is "
            f"{latest.get('status')}/{latest.get('conclusion')}"
        )
    return latest


def _jobs(repo: str, token: str, workflow: str, run_id: int) -> list[dict[str, Any]]:
    data = _api(repo, f"actions/runs/{run_id}/jobs?per_page=100&filter=latest", token)
    return _complete_list(data, "jobs", f"{workflow} jobs")


def verify_release(repo: str, sha: str, token: str) -> list[str]:
    """Return audit references if the tag has all required same-SHA evidence."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ReleaseGateError("GITHUB_REPOSITORY must be owner/repo")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ReleaseGateError("GITHUB_SHA must be a commit SHA")
    if not token:
        raise ReleaseGateError("GITHUB_TOKEN is required for release evidence checks")

    ancestry = _api(repo, f"compare/{sha}...main", token)
    if ancestry.get("status") not in {"ahead", "identical"}:
        raise ReleaseGateError(f"tag commit {sha} is not an ancestor of main")

    evidence = []
    for workflow, required_job, event in _REQUIRED:
        run = _workflow_run(repo, sha, token, workflow, event)
        run_id = run.get("id")
        if not isinstance(run_id, int):
            raise ReleaseGateError(f"{workflow}: run has no numeric ID")
        jobs = _jobs(repo, token, workflow, run_id)
        if required_job:
            gate = [job for job in jobs if job.get("name") == required_job]
            if len(gate) != 1 or gate[0].get("conclusion") != "success":
                raise ReleaseGateError(f"{workflow}: {required_job} did not pass in run {run_id}")
        else:
            names = {job.get("name") for job in jobs}
            if names != _MATRIX_JOBS or len(jobs) != len(_MATRIX_JOBS):
                raise ReleaseGateError(f"{workflow}: run {run_id} lacks the complete 45-job matrix")
            if any(job.get("conclusion") != "success" for job in jobs):
                raise ReleaseGateError(f"{workflow}: run {run_id} has a non-success matrix job")
        evidence.append(f"{workflow}: {run.get('html_url') or run_id}")
    return evidence


def main() -> int:
    try:
        package_version = verify_tag_version(
            os.environ.get("GITHUB_REF", ""), Path(__file__).resolve().parents[1]
        )
        evidence = verify_release(
            os.environ.get("GITHUB_REPOSITORY", ""),
            os.environ.get("GITHUB_SHA", ""),
            os.environ.get("GITHUB_TOKEN", ""),
        )
    except ReleaseGateError as exc:
        print(f"release preflight failed: {exc}")
        return 1
    for row in evidence:
        print(row)
    print(f"release preflight: v{package_version} matches fx1; same-SHA CI and matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
