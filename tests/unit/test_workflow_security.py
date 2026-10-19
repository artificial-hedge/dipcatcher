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


def test_workflow_checkouts_disable_credential_persistence_unless_pushing() -> None:
    """Checkout must not leave a writable origin credential unless the job pushes."""

    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs = workflow.get("jobs") or {}
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                continue
            pushes = _job_pushes_to_origin(job)
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                uses = str(step.get("uses") or "")
                if CHECKOUT_USES not in uses:
                    continue
                if pushes:
                    continue
                persisted = (step.get("with") or {}).get("persist-credentials")
                assert persisted is False, f"{path.name}:{job_id}"


CI_JOB_MAKE_TARGETS: dict[str, tuple[str, ...]] = {
    "lint": ("lint", "typecheck", "security"),
    "audit": ("audit",),
    "package": (),
    "container": (),
    "test": ("test",),
    "coverage": ("coverage",),
    "perf": (),
    "smoke": (),
    "mc-engine-smoke": ("mc-engine-smoke",),
    "rust-accel": ("native-lint",),
    "market-sim": ("market-sim-test",),
    "parity-smoke": ("parity-smoke",),
    "examples": ("examples",),
    "audit-observability": ("audit-obs",),
    "diffbacktest": ("diffbacktest",),
    "robustness-smoke": ("robustness-smoke",),
    "formal": ("formal",),
    "evidence-audit": ("evidence-audit",),
    "pretrade-risk": ("pretrade-bench",),
    "stress-smoke": ("stress-smoke",),
    "refresh-test-durations": ("test-durations",),
}

MAKE_CI_DEPS = ("lint", "typecheck", "security", "coverage")
MAKE_INVOCATION_RE = re.compile(r"\bmake\s+([a-z0-9-]+)")
MAKE_TARGET_RE = re.compile(r"^([a-zA-Z0-9_-]+):", re.MULTILINE)
LINT_JOB_COMMANDS = (
    "uv run ruff check src tests",
    "uv run ruff format --check src tests",
    "uv run python scripts/check_mypy_strict_allowlist.py",
    "uv run python scripts/check_mccabe_ratchet.py",
    "uv run mypy src/quant_fund",
    "uv run mypy --strict --follow-imports=silent",
    "bandit",
)


def _makefile_targets() -> set[str]:
    return set(MAKE_TARGET_RE.findall(MAKEFILE.read_text(encoding="utf-8")))


def _makefile_recipe(target: str) -> str:
    text = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(target)}:.*?(?=^[a-zA-Z0-9_-]+:|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, target
    return match.group(0)


def _make_ci_deps() -> list[str]:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^ci:\s+([^\n#]+)", makefile, flags=re.MULTILINE)
    assert match, "Makefile must declare a ci target"
    return match.group(1).split()


def test_ci_yml_make_steps_exist_in_makefile() -> None:
    """Every `make <target>` in ci.yml run steps must be a real Makefile target."""

    workflow = yaml.safe_load(_workflow("ci.yml"))
    targets = _makefile_targets()
    invoked: set[str] = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps") or []:
            run = step.get("run") if isinstance(step, dict) else ""
            if not run:
                continue
            invoked.update(MAKE_INVOCATION_RE.findall(run))
    missing = sorted(invoked - targets)
    assert not missing, f"ci.yml invokes undefined make targets: {missing}"


def test_make_ci_target_matches_ci_yml_lint_and_pr_test_gate() -> None:
    """`make ci` mirrors the ci.yml lint job plus the PR test/coverage gate."""

    assert _make_ci_deps() == list(MAKE_CI_DEPS)
    workflow = yaml.safe_load(_workflow("ci.yml"))
    extra = set(workflow["jobs"]) - set(CI_JOB_MAKE_TARGETS)
    missing = set(CI_JOB_MAKE_TARGETS) - set(workflow["jobs"])
    assert not extra, f"new ci.yml jobs need Makefile mapping: {sorted(extra)}"
    assert not missing, f"ci.yml dropped jobs still mapped: {sorted(missing)}"
    targets = _makefile_targets()
    for job_id, mapped in CI_JOB_MAKE_TARGETS.items():
        absent = [name for name in mapped if name not in targets]
        assert not absent, f"{job_id}: {absent}"

    lint_steps = "\n".join(
        str(step.get("run") or "") for step in workflow["jobs"]["lint"].get("steps") or []
    )
    recipes = "".join(_makefile_recipe(name) for name in ("lint", "typecheck", "security"))
    for command in LINT_JOB_COMMANDS:
        assert command in lint_steps, command
        assert command in recipes, command

    test_steps = yaml.safe_dump(workflow["jobs"]["test"])
    assert "not network and not slow" in test_steps


def test_agents_gate_table_names_ci_yml_jobs() -> None:
    """The AGENTS.md gate table must name every ci.yml job or its Make target."""

    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    start = agents.index("## Gates (run before committing)")
    end = agents.index("## Honesty contract", start)
    table = agents[start:end]
    for job_id, mapped in CI_JOB_MAKE_TARGETS.items():
        named = job_id in table or any(target in table for target in mapped)
        assert named, f"AGENTS.md gate table missing {job_id} (or {mapped})"
    assert "make ci" in table
    for dep in MAKE_CI_DEPS:
        assert dep in table


def test_release_preflight_requires_green_ci_and_evidence_audit() -> None:
    """Tag builds must not sign until the shared predicate script says the SHA is green."""

    workflow = yaml.safe_load(_workflow("release.yml"))
    assert "preflight" in workflow["jobs"]
    needs = workflow["jobs"]["build"].get("needs")
    assert needs == ["preflight"] or needs == "preflight"
    preflight = workflow["jobs"]["preflight"]
    assert not preflight.get("continue-on-error")
    dumped = yaml.safe_dump(preflight)
    assert "scripts/require_release_gates.py" in dumped
    assert "--sha" in dumped
    assert "--repo" in dumped
