"""SYNTHETIC tests for complete-lane coverage accounting and failure visibility."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from coverage import Coverage, CoverageData
from scripts import full_coverage as gate


@pytest.fixture
def project(tmp_path: Path) -> Path:
    for name in gate.SOURCE_ROOTS:
        (tmp_path / name).mkdir()
    (tmp_path / "src" / "sample.py").write_text(
        "def choose(flag):\n    if flag:\n        return 1\n    return 0\n"
    )
    (tmp_path / "src" / "never_imported.py").write_text("def unused():\n    return 8\n")
    (tmp_path / "src" / "__main__.py").write_text("print('SYNTHETIC entrypoint')\n")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests/unit", "tests/formal"]\n'
        '[tool.coverage.run]\nsource = ["src"]\nbranch = true\nomit = ["*/__main__.py"]\n'
        "[tool.coverage.report]\nfail_under = 81\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=SYNTHETIC",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-qm",
            "SYNTHETIC fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def _junit(path: Path, status: str = "passed") -> None:
    content = "" if status == "passed" else f"<{status} message='SYNTHETIC'/>"
    path.write_text(
        f"<testsuites><testsuite><testcase classname='suite' name='test_one'>{content}"
        "</testcase></testsuite></testsuites>"
    )


def _manifest(path: Path, **updates: object) -> None:
    value = json.loads(path.read_text())
    value.update(updates)
    path.write_text(json.dumps(value))


@pytest.fixture
def parts(project: Path) -> Path:
    root = project / "parts"
    root.mkdir()
    for lane in gate.LANES:
        directory = root / lane
        directory.mkdir()
        config = directory / "coverage.ini"
        config.write_text(gate.coverage_config(project))
        coverage = directory / ".coverage"
        data = CoverageData(basename=str(coverage))
        data.add_arcs({"src/sample.py": [(-1, 1), (1, -1), (-1, 2), (2, 3), (3, -1)]})
        data.write()
        data.close()
        _junit(directory / "junit.xml")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": gate.SCHEMA,
                    "lane": lane,
                    "revision": gate.revision(project),
                    "configuration_sha256": gate._sha256(config),
                    "pytest_exit_code": 0,
                    "coverage_sha256": gate._sha256(coverage),
                    "tests": {"passed": 999999},
                }
            )
        )
    return root


def test_full_configuration_keeps_the_floor_and_includes_entrypoints(project: Path) -> None:
    config = project / "full.ini"
    config.write_text(gate.coverage_config(project))
    cov = Coverage(config_file=str(config))
    assert cov.get_option("report:fail_under") == 81
    assert cov.get_option("run:source") == list(gate.SOURCE_ROOTS)
    assert cov.get_option("run:omit") == []
    assert cov.get_option("run:branch") is True
    assert cov.get_option("run:relative_files") is True
    assert cov.get_option("report:include_namespace_packages") is True
    assert "subprocess" in cov.get_option("run:patch")


@pytest.mark.parametrize("value", ["0", "-1", "101", "nan", "inf", "true", '"81"'])
def test_invalid_or_missing_floor_never_defaults_to_zero(project: Path, value: str) -> None:
    path = project / "pyproject.toml"
    path.write_text(path.read_text().replace("fail_under = 81", f"fail_under = {value}"))
    with pytest.raises(gate.CoverageGateError):
        gate.coverage_config(project)


@pytest.mark.parametrize("lane", gate.LANES)
def test_full_selection_includes_slow_and_separate_suites(project: Path, lane: str) -> None:
    args = gate.test_arguments(project, lane, project / "config.ini", project / "junit.xml")
    assert args[args.index("-m") + 1] == "pytest"  # Python's -m precedes pytest's marker.
    assert "not network" in args
    assert not any("not slow" in arg or "not perf_full" in arg for arg in args)
    if lane in gate.SEPARATE_LANES:
        assert gate.SEPARATE_LANES[lane] in args
        assert "--splits" not in args
    else:
        assert {"tests/unit", "tests/formal"}.issubset(args)
        assert not set(gate.SEPARATE_LANES.values()).intersection(args)
        assert args[args.index("--group") + 1] == lane[-1]


def test_invalid_lane_fails_before_creating_artifacts(project: Path) -> None:
    with pytest.raises(gate.CoverageGateError):
        gate.run_lane(project, project / "output", "../unreviewed")
    assert not (project / "output").exists()


def test_exact_lane_set_is_required(project: Path, parts: Path) -> None:
    config_sha = hashlib.sha256(gate.coverage_config(project).encode()).hexdigest()
    assert len(gate.validate_parts(parts, gate.revision(project), config_sha)) == len(gate.LANES)
    (parts / "native").rename(parts / "other")
    with pytest.raises(gate.CoverageGateError, match="missing=.*native.*unexpected=.*other"):
        gate.validate_parts(parts, gate.revision(project), config_sha)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "a" * 40),
        ("schema", "other"),
        ("lane", "python-99"),
        ("configuration_sha256", "b" * 64),
        ("coverage_sha256", "c" * 64),
    ],
)
def test_mismatched_manifest_is_rejected(
    project: Path, parts: Path, field: str, value: str
) -> None:
    _manifest(parts / "python-1" / "manifest.json", **{field: value})
    config_sha = hashlib.sha256(gate.coverage_config(project).encode()).hexdigest()
    with pytest.raises(gate.CoverageGateError, match="mismatch"):
        gate.validate_parts(parts, gate.revision(project), config_sha)


def test_worker_fragments_cannot_substitute_for_controller_result(
    project: Path, parts: Path
) -> None:
    (parts / "python-1" / ".coverage").rename(parts / "python-1" / ".coverage.worker")
    with pytest.raises(gate.CoverageGateError, match="controller"):
        gate.validate_parts(
            parts, gate.revision(project), gate._sha256(parts / "native" / "coverage.ini")
        )


def test_statement_only_data_is_not_branch_coverage(project: Path, parts: Path) -> None:
    path = parts / "python-1" / ".coverage"
    path.unlink()
    data = CoverageData(basename=str(path))
    data.add_lines({"src/sample.py": [1]})
    data.write()
    data.close()
    _manifest(path.parent / "manifest.json", coverage_sha256=gate._sha256(path))
    with pytest.raises(gate.CoverageGateError, match="branch"):
        gate.validate_parts(
            parts, gate.revision(project), gate._sha256(path.parent / "coverage.ini")
        )


@pytest.mark.parametrize(
    "status,code", [("failure", 1), ("error", 1), ("skipped", 0), ("passed", True)]
)
def test_failed_all_skipped_and_boolean_exit_codes_do_not_pass(
    parts: Path, status: str, code: object
) -> None:
    lane = parts / "python-1"
    _junit(lane / "junit.xml", status)
    _manifest(lane / "manifest.json", pytest_exit_code=code)
    result = gate.summarize(parts, {"files": {}, "totals": {}}, set())
    assert result["all_executed_tests_passed"] is False
    assert result["lanes"]["python-1"]["passed"] is False


def test_junit_not_claimed_test_counts_is_the_source_of_truth(parts: Path) -> None:
    result = gate.summarize(parts, {"files": {}, "totals": {}}, {"src/missing.py"})
    assert result["lanes"]["python-1"]["tests"]["passed"] == 1
    assert result["unrepresented_source_files"] == ["src/missing.py"]


def test_incomplete_input_writes_a_failed_summary(project: Path, parts: Path) -> None:
    (parts / "native" / ".coverage").unlink()
    out = project / "output"
    with pytest.raises(gate.CoverageGateError):
        gate.combine(project, parts, out)
    assert json.loads((out / "summary.json").read_text())["passed"] is False
    assert not (out / "coverage.json").exists()


def test_stale_output_directory_is_never_reused(project: Path) -> None:
    output = project / "output"
    output.mkdir()
    (output / "summary.json").write_text("old SYNTHETIC result")
    with pytest.raises(FileExistsError):
        gate.run_lane(project, output, "native")
    assert (output / "summary.json").read_text() == "old SYNTHETIC result"


def test_combine_from_another_cwd_keeps_config_and_unexecuted_sources(
    project: Path, parts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = project / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    out = project / "output"
    assert gate.combine(project, parts, out) == 1  # Under 81%, not a default-zero green.
    report = json.loads((out / "coverage.json").read_text())
    assert {"src/sample.py", "src/never_imported.py", "src/__main__.py"}.issubset(report["files"])
    assert report["files"]["src/never_imported.py"]["summary"]["covered_lines"] == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["coverage_gate_passed"] is False
    assert summary["unrepresented_source_files"] == []
    assert Path.cwd() == elsewhere
    assert all((parts / lane / ".coverage").exists() for lane in gate.LANES)


def test_tracked_source_inventory_is_not_test_file_inventory(project: Path) -> None:
    assert gate.tracked_sources(project) == {
        "src/sample.py",
        "src/never_imported.py",
        "src/__main__.py",
    }


def test_uploaded_configuration_is_verified(project: Path, parts: Path) -> None:
    expected = gate._sha256(parts / "native" / "coverage.ini")
    (parts / "python-1" / "coverage.ini").write_text("[report]\nfail_under = 0\n")
    with pytest.raises(gate.CoverageGateError, match="configuration digest mismatch"):
        gate.validate_parts(parts, gate.revision(project), expected)


def test_corrupt_database_produces_a_failed_summary(project: Path, parts: Path) -> None:
    from coverage.exceptions import CoverageException

    data = parts / "python-1" / ".coverage"
    data.write_bytes(b"SYNTHETIC corrupt SQLite file")
    _manifest(data.parent / "manifest.json", coverage_sha256=gate._sha256(data))
    output = project / "output"
    with pytest.raises(CoverageException):
        gate.combine(project, parts, output)
    assert json.loads((output / "summary.json").read_text())["passed"] is False


def test_successful_complete_aggregate(project: Path, parts: Path) -> None:
    # Deliberately constructed coverage data test the merger, not application behavior.
    for lane in gate.LANES:
        data_path = parts / lane / ".coverage"
        data_path.unlink()
        data = CoverageData(basename=str(data_path))
        data.add_arcs(
            {
                "src/sample.py": [(-1, 1), (1, -1), (-1, 2), (2, 3), (3, -1), (2, 4), (4, -1)],
                "src/never_imported.py": [(-1, 1), (1, -1), (-1, 2), (2, -1)],
                "src/__main__.py": [(-1, 1), (1, -1)],
            }
        )
        data.write()
        data.close()
        _manifest(data_path.parent / "manifest.json", coverage_sha256=gate._sha256(data_path))
    output = project / "output"
    assert gate.combine(project, parts, output) == 0
    result = json.loads((output / "summary.json").read_text())
    assert result["passed"] is True
    assert result["has_skipped_tests"] is False
    assert result["security_audit_claim"] is False
    assert set(result["lanes"]) == set(gate.LANES)
    assert (output / "html" / "index.html").is_file()
    assert (output / "coverage.xml").is_file()


@pytest.mark.parametrize("failing", [False, True])
def test_real_pytest_lane_preserves_results_and_exit_status(
    project: Path, monkeypatch: pytest.MonkeyPatch, failing: bool
) -> None:
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    test_dir = project / "tests" / "native"
    test_dir.mkdir(parents=True)
    (test_dir / "test_sample.py").write_text(
        "import runpy\nfrom pathlib import Path\n"
        "def test_synthetic():\n"
        "    module = runpy.run_path(str(Path('src/sample.py').resolve()))\n"
        f"    assert module['choose'](True) == {99 if failing else 1}\n"
    )
    output = project / "output"
    assert gate.run_lane(project, output, "native") == int(failing)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["pytest_exit_code"] == int(failing)
    assert manifest["tests"]["total"] == 1
    assert manifest["tests"]["failed"] == int(failing)
    assert manifest["coverage_sha256"] == gate._sha256(output / ".coverage")
    assert (output / "pytest.log").stat().st_size > 0


def test_one_lane_error_does_not_prevent_other_lanes(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed = []

    def fake_lane(root: Path, output: Path, lane: str) -> int:
        executed.append(lane)
        if lane == "python-2":
            raise OSError("SYNTHETIC process launch failure")
        output.mkdir()
        return 0

    monkeypatch.setattr(gate, "run_lane", fake_lane)
    monkeypatch.setattr(gate, "combine", lambda *args: 1)
    output = project / "output"
    assert gate.run_all(project, output) == 1
    assert executed == list(gate.LANES)
    error = output / "parts" / "python-2" / "runner-error.json"
    assert "SYNTHETIC process launch failure" in error.read_text()


def test_workflow_preserves_independent_checks_and_evidence() -> None:
    import re

    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.load(
        (root / ".github/workflows/full-coverage.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert config["permissions"] == {"contents": "read"}
    jobs = config["jobs"]
    assert set(jobs["python"]["strategy"]["matrix"]["lane"]) == set(gate.LANES)
    assert jobs["python"]["strategy"]["fail-fast"] == "false"
    assert config["concurrency"]["cancel-in-progress"] == "false"
    assert set(jobs["result"]["needs"]) == set(jobs) - {"result"}
    for job in jobs.values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.search(r"@[a-f0-9]{40}$", step["uses"])
            assert "continue-on-error" not in step
    assert "epoch-candidate" in jobs
    uploads = [
        s
        for s in jobs["python"]["steps"]
        if s.get("uses", "").startswith("actions/upload-artifact")
    ]
    assert uploads[0]["with"]["include-hidden-files"] == "true"
    assert uploads[0]["with"]["if-no-files-found"] == "error"


def test_git_revision_must_be_a_complete_object_id(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *args, **kwargs: "not-a-sha\n")
    with pytest.raises(gate.CoverageGateError, match="complete commit"):
        gate.revision(project)


@pytest.mark.parametrize("value", ["[]", "[7]", '"tests/unit"'])
def test_bad_test_root_configuration_is_not_silently_accepted(project: Path, value: str) -> None:
    path = project / "pyproject.toml"
    path.write_text(path.read_text().replace('["tests/unit", "tests/formal"]', value))
    with pytest.raises(gate.CoverageGateError, match="testpaths"):
        gate.test_arguments(project, "python-1", project / "c.ini", project / "j.xml")


def test_invalid_lane_cannot_construct_a_pytest_command(project: Path) -> None:
    with pytest.raises(gate.CoverageGateError, match="unknown coverage lane"):
        gate.test_arguments(project, "missing", project / "c.ini", project / "j.xml")


def test_empty_tracked_source_inventory_is_a_failure(project: Path) -> None:
    subprocess.run(
        ["git", "rm", "--cached", "-r", "src"], cwd=project, check=True, capture_output=True
    )
    with pytest.raises(gate.CoverageGateError, match="no tracked Python"):
        gate.tracked_sources(project)


@pytest.mark.parametrize("operation", ["run", "all", "combine"])
def test_cli_dispatch_preserves_failures_and_resolves_paths(
    project: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls = []

    def record(*args: object) -> int:
        calls.append(args)
        return 3

    monkeypatch.setattr(
        gate, {"run": "run_lane", "all": "run_all", "combine": "combine"}[operation], record
    )
    arguments = ["--root", str(project), operation, "--output", str(project / "out")]
    if operation == "run":
        arguments += ["--lane", "fx1"]
    elif operation == "combine":
        arguments += ["--parts", str(project / "parts")]
    assert gate.main(arguments) == 3
    assert len(calls) == 1
    assert calls[0][0] == project.resolve()


def test_command_line_entrypoint_help() -> None:
    import sys

    script = Path(gate.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "combine" in result.stdout


def test_reporter_initialization_failure_restores_cwd(
    project: Path, parts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import coverage

    def failed_constructor(*args: object, **kwargs: object) -> object:
        raise OSError("SYNTHETIC unavailable reporter configuration")

    before = Path.cwd()
    monkeypatch.setattr(coverage, "Coverage", failed_constructor)
    output = project / "out"
    with pytest.raises(OSError, match="unavailable reporter"):
        gate.combine(project, parts, output)
    assert Path.cwd() == before
    assert json.loads((output / "summary.json").read_text())["passed"] is False
