"""Regression guards for fail-closed, least-privilege GitHub workflows."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
MAKEFILE = REPO_ROOT / "Makefile"
CHECKOUT_USES = "actions/checkout@"


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_ci_does_not_grant_workflow_wide_actions_write() -> None:
    """Caches and artifacts must not widen the GITHUB_TOKEN permission set."""

    workflow = _workflow("ci.yml")

    top_level = workflow.split("\njobs:\n", maxsplit=1)[0]
    assert "actions: write" not in top_level


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


def test_ci_and_release_workflows_parse() -> None:
    """Workflow edits must remain valid YAML (catches indentation and anchor drift)."""

    for name in ("ci.yml", "release.yml", "proofcore.yml", "simtest.yml", "matrix.yml"):
        doc = yaml.safe_load(_workflow(name))
        assert isinstance(doc, dict)
        assert doc.get("jobs")


def _job_pushes_to_origin(job: dict) -> bool:
    script = yaml.safe_dump(job)
    return "git push" in script


def _assert_checkouts_disable_credential_persistence(workflow_name: str) -> None:
    workflow = yaml.safe_load(_workflow(workflow_name))
    for job_id, job in workflow["jobs"].items():
        pushes = _job_pushes_to_origin(job)
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if CHECKOUT_USES not in uses:
                continue
            with_block = step.get("with") or {}
            if pushes:
                continue
            assert with_block.get("persist-credentials") is False, f"{workflow_name}:{job_id}"


def test_workflow_checkouts_disable_credential_persistence_unless_pushing() -> None:
    """Checkout must not leave a writable origin credential unless the job pushes."""

    for path in sorted(WORKFLOWS.glob("*.yml")):
        _assert_checkouts_disable_credential_persistence(path.name)


def test_make_ci_target_matches_ci_yml_lint_and_pr_test_gate() -> None:
    """`make ci` must stay aligned with the lint job and the PR pytest marker."""

    makefile = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^ci:\s+([^\n#]+)", makefile, flags=re.MULTILINE)
    assert match, "Makefile must declare a ci target"
    deps = match.group(1).split()
    assert deps == ["lint", "typecheck", "security", "coverage"]

    workflow = yaml.safe_load(_workflow("ci.yml"))
    lint_steps = yaml.safe_dump(workflow["jobs"]["lint"])
    assert "ruff check src tests" in lint_steps
    assert "mypy src/quant_fund" in lint_steps
    assert "bandit" in lint_steps

    test_steps = yaml.safe_dump(workflow["jobs"]["test"])
    assert "not network and not slow" in test_steps


def test_release_preflight_requires_green_ci_and_evidence_audit() -> None:
    """Tag builds must not run on a SHA that never passed CI + evidence-audit."""

    workflow = yaml.safe_load(_workflow("release.yml"))
    assert "preflight" in workflow["jobs"]
    needs = workflow["jobs"]["build"].get("needs")
    assert needs == ["preflight"] or needs == "preflight"
    preflight = yaml.safe_dump(workflow["jobs"]["preflight"])
    assert "ci.yml/runs" in preflight
    assert "evidence-audit" in preflight
