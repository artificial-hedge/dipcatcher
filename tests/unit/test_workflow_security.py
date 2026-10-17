"""Regression guards for fail-closed, least-privilege GitHub workflows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_ci_does_not_grant_workflow_wide_actions_write() -> None:
    """Caches and artifacts must not widen the GITHUB_TOKEN permission set."""

    workflow = _workflow("ci.yml")

    top_level = workflow.split("\njobs:\n", maxsplit=1)[0]
    assert "actions: write" not in top_level


def test_fast_gates_report_merge_queue_checks() -> None:
    """A queued merge needs the bounded main-bound PR gate workflows."""

    for name in ("ci.yml", "fx1.yml", "proofcore.yml"):
        config = yaml.load(_workflow(name), Loader=yaml.BaseLoader)
        events = config["on"]
        assert "pull_request" in events, name
        assert events["merge_group"] == {
            "types": ["checks_requested"],
            "branches": ["main"],
        }, name
        assert config["concurrency"]["cancel-in-progress"] == "false", name

    ci = yaml.load(_workflow("ci.yml"), Loader=yaml.BaseLoader)
    base_step = next(
        step
        for job in ci["jobs"].values()
        for step in job.get("steps", [])
        if step.get("id") == "base"
    )
    assert "github.event_name == 'merge_group'" in base_step["if"]
    assert base_step["env"]["PR_BASE"] == "${{ github.event.pull_request.base.sha }}"
    assert base_step["env"]["MERGE_BASE"] == "${{ github.event.merge_group.base_sha }}"
    assert 'if [ "$EVENT" = merge_group ]' in base_step["run"]
    assert 'if [ "$EVENT" = merge_group ] || [ "$EVENT" = pull_request ]' in base_step["run"]


def test_platform_matrix_is_nightly_or_manual() -> None:
    """The full OS/Python matrix must not occupy merge queue capacity."""

    config = yaml.load(_workflow("matrix.yml"), Loader=yaml.BaseLoader)
    events = config["on"]
    assert set(events) == {"schedule", "workflow_dispatch"}
    assert events["schedule"] == [{"cron": "37 22 * * *"}]
    assert config["concurrency"]["cancel-in-progress"] == "false"
    for name in ("inventory", "test"):
        matrix = config["jobs"][name]["strategy"]["matrix"]
        assert set(matrix["os"]) == {"ubuntu-latest", "macos-latest", "windows-latest"}
        assert set(matrix["python"]) == {"3.12", "3.13", "3.14"}


def test_required_status_checks_cover_every_job() -> None:
    """Each unique branch check must fail if any present or future job fails."""

    for workflow, context in (
        ("ci.yml", "ci-required"),
        ("fx1.yml", "fx1-required"),
        ("proofcore.yml", "proofcore-required"),
    ):
        config = yaml.load(_workflow(workflow), Loader=yaml.BaseLoader)
        jobs = config["jobs"]
        gate = jobs["required-gate"]
        assert gate["name"] == context
        assert gate["if"] == "${{ always() }}"
        assert set(gate["needs"]) == set(jobs) - {"required-gate"}
        step = gate["steps"][0]
        assert step["env"]["NEEDS_JSON"] == "${{ toJSON(needs) }}"
        assert 'row.get("result") != "success"' in step["run"]
        assert "if not jobs or failures:" in step["run"]


def test_required_status_checks_fail_on_missing_or_non_success_jobs() -> None:
    """Execute the workflow's actual gate code against success and red cases."""

    for workflow in ("ci.yml", "fx1.yml", "proofcore.yml"):
        config = yaml.load(_workflow(workflow), Loader=yaml.BaseLoader)
        script = config["jobs"]["required-gate"]["steps"][0]["run"]
        code = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        for outcomes, expected_code in (
            ({"test": {"result": "success"}}, 0),
            ({"test": {"result": "failure"}}, 1),
            ({"test": {"result": "skipped"}}, 1),
            ({}, 1),
        ):
            result = subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "NEEDS_JSON": json.dumps(outcomes)},
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == expected_code, (workflow, outcomes, result.stderr)


def test_dependency_review_fails_closed() -> None:
    """Unavailable dependency metadata must not produce a green review."""

    workflow = _workflow("dependency-review.yml")

    assert "actions/dependency-review-action@" in workflow
    assert "continue-on-error:" not in workflow
    assert "steps.review.outcome" not in workflow
    assert 'if [ "$code" = "404" ]' not in workflow


def test_codeql_covers_supported_source_languages() -> None:
    """Python and the repository's JavaScript/TypeScript must be analyzed."""

    workflow = _workflow("codeql.yml")
    config = (REPO_ROOT / ".github" / "codeql" / "codeql-config.yml").read_text(encoding="utf-8")

    assert "language: [python, javascript-typescript]" in workflow
    assert "languages: ${{ matrix.language }}" in workflow
    for path in ("src", "tests", "scripts", "web", "replay", "clients/typescript"):
        assert f"  - {path}\n" in config
