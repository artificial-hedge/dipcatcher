"""Run and combine the complete offline Python test lanes without partial-green reports.

This reports test execution and source coverage, not a security certification.
Run from a complete checkout in the locked environment (``make sync`` first).
Artifacts belong under data/metadata, never the immutable research receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LAB_LANES = ("python-1", "python-2", "python-3", "python-4")
SEPARATE_LANES = {
    "fx1": "tests/fx1",
    "examples": "tests/examples",
    "performance": "tests/perf",
    "native": "tests/native",
}
LANES = (*LAB_LANES, *SEPARATE_LANES)
# ``SOURCE_ROOTS`` enumerates the first-party Python roots whose
# coverage denominator is included in the full-offline report. Anything
# outside these roots is excluded and stamped as ``uncovered_path_outside``
# in the report (see #2851 — replay/scripts and web/scripts were omitted
# in the prior revision). To add a new root: append the path, then run a
# full lane to refresh ``unrepresented_source_files``.
SOURCE_ROOTS = ("src", "scripts", "examples", "replay/scripts", "web/scripts")
SCHEMA = "full-offline-coverage.v1"


class CoverageGateError(ValueError):
    """The evidence is missing, inconsistent, or insufficient for a full result."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _project(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def revision(root: Path) -> str:
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise CoverageGateError("git did not return a complete commit object ID")
    return value


def coverage_config(root: Path) -> str:
    """Keep the existing floor; widen measurement, never the exclusion list."""
    report = _project(root)["tool"]["coverage"]["report"]
    floor = report["fail_under"]
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        raise CoverageGateError("coverage fail_under must be a numeric project setting")
    if not math.isfinite(floor) or not 0 < floor <= 100:
        raise CoverageGateError("coverage fail_under must be finite and in (0, 100]")
    # Use a separate configuration: ordinary PR coverage and its ratchets are
    # unchanged. All first-party Python roots and __main__.py are included.
    # Retain coverage.py's non-executable defaults, not broad project omissions.
    return (
        "[run]\nbranch = true\nrelative_files = true\nparallel = true\n"
        "patch = subprocess\nsource =\n"
        + "".join(f"    {source}\n" for source in SOURCE_ROOTS)
        + "\n[report]\ninclude_namespace_packages = true\n"
        "show_missing = true\nskip_covered = false\n"
        f"fail_under = {floor}\n"
    )


def test_arguments(root: Path, lane: str, config: Path, junit: Path) -> list[str]:
    if lane not in LANES:
        raise CoverageGateError(f"unknown coverage lane: {lane!r}")
    args = [sys.executable, "-m", "pytest"]
    if lane in SEPARATE_LANES:
        args += [SEPARATE_LANES[lane]]
    else:
        roots = _project(root)["tool"]["pytest"]["ini_options"]["testpaths"]
        if not isinstance(roots, list) or not roots or not all(isinstance(p, str) for p in roots):
            raise CoverageGateError("pytest testpaths must be a nonempty list of paths")
        args += list(dict.fromkeys(roots))
        args += [
            "-n",
            "auto",
            "--dist",
            "loadfile",
            "--splits",
            "4",
            "--group",
            lane.rsplit("-", 1)[1],
            "--durations-path",
            ".test_durations",
            "--splitting-algorithm",
            "least_duration",
        ]
    # Slow and perf_full tests are deliberately included. Network tests are
    # an explicit coverage gap, not permission to use hosted APIs/live trades.
    args += [
        "-m",
        "not network",
        "--strict-markers",
        "--cov",
        f"--cov-config={config}",
        "--cov-report=",
        "--cov-fail-under=0",
        f"--junitxml={junit}",
    ]
    return args


def junit_counts(path: Path) -> dict[str, Any]:
    counts: dict[str, Any] = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    skipped: list[str] = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "testcase":
            continue
        if element.find("error") is not None:
            counts["errors"] += 1
        elif element.find("failure") is not None:
            counts["failed"] += 1
        elif element.find("skipped") is not None:
            counts["skipped"] += 1
            skipped.append(f"{element.get('classname', '')}::{element.get('name', '')}")
        else:
            counts["passed"] += 1
        element.clear()
    counts["total"] = sum(counts.values())
    counts["skipped_tests"] = skipped
    return counts


def run_lane(root: Path, output: Path, lane: str) -> int:
    if lane not in LANES:
        raise CoverageGateError(f"unknown coverage lane: {lane!r}")
    # Refuse stale evidence rather than silently reusing a prior run's data.
    output.mkdir(parents=True, exist_ok=False)
    config = output / "coverage.ini"
    config.write_text(coverage_config(root), encoding="utf-8")
    junit = output / "junit.xml"
    command = test_arguments(root, lane, config, junit)
    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(output / ".coverage")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH")]))
    started = time.monotonic()
    commit = revision(root)
    print(f"Running {lane}; complete output: {output / 'pytest.log'}", flush=True)
    with (output / "pytest.log").open("wb") as log:
        result = subprocess.run(
            command, cwd=root, env=env, check=False, stdout=log, stderr=subprocess.STDOUT
        )
    print(f"{lane}: pytest exit code {result.returncode}", flush=True)
    data = output / ".coverage"
    counts = junit_counts(junit) if junit.is_file() else None
    manifest = {
        "schema": SCHEMA,
        "lane": lane,
        "revision": commit,
        "configuration_sha256": _sha256(config),
        "python": sys.version,
        "command": command,
        "pytest_exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "tests": counts,
        "coverage_sha256": _sha256(data) if data.is_file() else None,
        "network_marked_tests_included": False,
        "scope": "offline tests, including slow and perf_full; first-party Python roots",
    }
    _write_json(output / "manifest.json", manifest)
    return result.returncode


def validate_parts(parts: Path, expected_revision: str, config_sha: str) -> list[Path]:
    """Require every controller result; worker fragments cannot stand in for one."""
    from coverage import CoverageData

    inputs = []
    found = {path.name for path in parts.iterdir() if path.is_dir()}
    if found != set(LANES):
        raise CoverageGateError(
            f"coverage lanes differ: missing={sorted(set(LANES) - found)}, "
            f"unexpected={sorted(found - set(LANES))}"
        )
    for lane in LANES:
        directory = parts / lane
        path = directory / ".coverage"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected = {
            "schema": SCHEMA,
            "lane": lane,
            "revision": expected_revision,
            "configuration_sha256": config_sha,
        }
        if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
            raise CoverageGateError(f"{lane}: manifest identity/configuration mismatch")
        if _sha256(directory / "coverage.ini") != config_sha:
            raise CoverageGateError(f"{lane}: uploaded configuration digest mismatch")
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise CoverageGateError(f"{lane}: controller coverage data is missing or invalid")
        if manifest.get("coverage_sha256") != _sha256(path):
            raise CoverageGateError(f"{lane}: coverage data digest mismatch")
        data = CoverageData(basename=str(path))
        try:
            data.read()
            if not data.has_arcs() or not data.measured_files():
                raise CoverageGateError(f"{lane}: no branch-coverage measurements")
        finally:
            data.close()
        inputs.append(path)
    return inputs


def tracked_sources(root: Path) -> set[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", *SOURCE_ROOTS], cwd=root)
    paths = {os.fsdecode(path) for path in raw.split(b"\0") if path and path.endswith(b".py")}
    if not paths:
        raise CoverageGateError("no tracked Python source files were found")
    return paths


def summarize(parts: Path, report: dict[str, Any], expected_sources: set[str]) -> dict[str, Any]:
    """Never turn a failed/all-skipped lane or absent source into full success."""
    missing = sorted(expected_sources - set(report["files"]))
    lanes = {}
    for lane in LANES:
        manifest = json.loads((parts / lane / "manifest.json").read_text(encoding="utf-8"))
        # Read actual JUnit, rather than trusting the duplicated manifest counts.
        counts = junit_counts(parts / lane / "junit.xml")
        code = manifest.get("pytest_exit_code")
        passed = (
            type(code) is int
            and code == 0
            and counts["passed"] > 0
            and counts["failed"] == 0
            and counts["errors"] == 0
        )
        lanes[lane] = {"passed": passed, "pytest_exit_code": code, "tests": counts}
    return {
        "schema": SCHEMA,
        "all_lanes_present": True,
        "all_executed_tests_passed": all(lane["passed"] for lane in lanes.values()),
        "unrepresented_source_files": missing,
        "tracked_source_files": len(expected_sources),
        "coverage": report["totals"],
        "lanes": lanes,
        "has_skipped_tests": any(lane["tests"]["skipped"] for lane in lanes.values()),
        "network_marked_tests_included": False,
        "security_audit_claim": False,
    }


def combine(root: Path, parts: Path, output: Path) -> int:
    from coverage import Coverage
    from coverage.exceptions import CoverageException

    output.mkdir(parents=True, exist_ok=False)
    config = output / "coverage.ini"
    config.write_text(coverage_config(root), encoding="utf-8")
    summary_path = output / "summary.json"
    try:
        commit = revision(root)
        inputs = validate_parts(parts, commit, _sha256(config))
        # Configuration discovery must not depend on cd'ing into an artifact
        # directory, and all relative source names must resolve from the checkout.
        previous = Path.cwd()
        cov = None
        try:
            os.chdir(root)
            cov = Coverage(config_file=str(config), data_file=str(output / ".coverage"))
            cov.combine(data_paths=[str(path) for path in inputs], strict=True, keep=True)
            # Explicit zero-hit entries keep never-imported tracked modules in
            # the denominator even if a producer omitted discovery metadata.
            cov.get_data().touch_files(sorted(tracked_sources(root)))
            cov.save()
            cov.json_report(outfile=str(output / "coverage.json"))
            cov.xml_report(outfile=str(output / "coverage.xml"))
            cov.html_report(directory=str(output / "html"))
            report = json.loads((output / "coverage.json").read_text(encoding="utf-8"))
        finally:
            if cov is not None:
                cov.get_data().close()
            os.chdir(previous)
        summary = summarize(parts, report, tracked_sources(root))
        # The CLI applies coverage.py's exact precision/100%-floor semantics.
        threshold = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                f"--rcfile={config}",
                f"--data-file={output / '.coverage'}",
            ],
            cwd=root,
            check=False,
        ).returncode
        summary["revision"] = commit
        summary["coverage_gate_passed"] = threshold == 0
        summary["passed"] = (
            summary["all_executed_tests_passed"]
            and not summary["unrepresented_source_files"]
            and threshold == 0
        )
        _write_json(summary_path, summary)
        return 0 if summary["passed"] else 1
    except (OSError, ValueError, ET.ParseError, CoverageException) as error:
        _write_json(summary_path, {"schema": SCHEMA, "passed": False, "error": str(error)})
        raise


def run_all(root: Path, output: Path) -> int:
    """Run every lane, retaining other lanes' results when one fails.

    Sequential execution avoids oversubscribing xdist workers and keeps the
    optional native build/environment preparation outside measurement.
    """
    output.mkdir(parents=True, exist_ok=False)
    parts = output / "parts"
    parts.mkdir()
    for lane in LANES:
        try:
            run_lane(root, parts / lane, lane)
        except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as error:
            directory = parts / lane
            directory.mkdir(exist_ok=True)
            _write_json(directory / "runner-error.json", {"error": str(error), "lane": lane})
            print(f"{lane}: {error}", file=sys.stderr)
    # Missing evidence is an aggregate failure, never permission to combine a
    # smaller subset. Individual lane errors are retained for diagnosis.
    return combine(root, parts, output / "combined")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="operation", required=True)
    run = commands.add_parser("run")
    run.add_argument("--lane", choices=LANES, required=True)
    run.add_argument("--output", type=Path, required=True)
    all_lanes = commands.add_parser("all")
    all_lanes.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("combine")
    merge.add_argument("--parts", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    if args.operation == "all":
        return run_all(root, output)
    if args.operation == "run":
        return run_lane(root, output, args.lane)
    return combine(root, args.parts.resolve(), output)


if __name__ == "__main__":
    raise SystemExit(main())
