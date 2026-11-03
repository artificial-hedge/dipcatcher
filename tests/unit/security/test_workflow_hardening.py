"""Workflow run scripts do not interpolate Actions expressions."""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]
_RUN_GUARDS = (
    "ci.yml",
    "docs.yml",
    "full-coverage.yml",
    "matrix.yml",
    "reproduce_sota.yml",
)
_CHECKOUT_GUARDS = (
    "ci.yml",
    "proofcore.yml",
    "replay_viz.yml",
    "simtest.yml",
    "web_explorer.yml",
    "witness_monitor.yml",
)


def _load(name: str) -> dict[str, object]:
    text = (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    loaded = yaml.load(text, Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _jobs(workflow: dict[str, object]) -> dict[str, dict[str, object]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def test_run_scripts_do_not_interpolate_expressions() -> None:
    for name in _RUN_GUARDS:
        for job_name, job in _jobs(_load(name)).items():
            steps = job.get("steps") or []
            assert isinstance(steps, list)
            for index, step in enumerate(steps):
                assert isinstance(step, dict)
                run = step.get("run")
                if isinstance(run, str):
                    assert "${{" not in run, f"{name}:{job_name}:{index}"


def test_guarded_checkouts_do_not_persist_credentials() -> None:
    for name in _CHECKOUT_GUARDS:
        found = 0
        for job in _jobs(_load(name)).values():
            steps = job.get("steps") or []
            assert isinstance(steps, list)
            for step in steps:
                assert isinstance(step, dict)
                uses = step.get("uses")
                if not isinstance(uses, str) or not uses.startswith("actions/checkout@"):
                    continue
                found += 1
                options = step.get("with")
                assert isinstance(options, dict)
                assert options.get("persist-credentials") == "false"
        assert found > 0, name


def test_runner_labels_stay_on_instabox_except_the_os_matrix() -> None:
    names = sorted(set(_RUN_GUARDS) | set(_CHECKOUT_GUARDS))
    for name in names:
        for job_name, job in _jobs(_load(name)).items():
            runs_on = job["runs-on"]
            if name == "matrix.yml" and job_name == "test":
                assert runs_on == "${{ matrix.os }}"
            else:
                assert runs_on == "instabox-large"


def test_docs_deploy_does_not_print_the_pages_body() -> None:
    text = (_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "pages.json" not in text
    assert "cat /tmp/pages.json" not in text
    assert "/dev/null" in text
