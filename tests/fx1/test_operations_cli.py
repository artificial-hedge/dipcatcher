"""Offline CLI-boundary regressions using synthetic inputs and temporary files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from fx1.cli import app

RUNNER = CliRunner()
BUDGET = 2_000_000
EXECUTE = ["harness", "execute-operation", "features.simple_returns"]
VALID = '{"prices": [1, 2]}'


def assert_usage_error(result: Result) -> None:
    assert result.exit_code == 2, result.exception
    assert "Invalid value" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("case", ["missing", "directory", "invalid_utf8", "malformed"])
def test_arguments_file_errors_are_usage_errors(tmp_path: Path, case: str) -> None:
    source = tmp_path / "arguments.json"
    if case == "directory":
        source.mkdir()
    elif case == "invalid_utf8":
        source.write_bytes(b"\xff")
    elif case == "malformed":
        source.write_text("{", encoding="utf-8")
    result = RUNNER.invoke(app, [*EXECUTE, "--arguments-file", str(source)])
    assert_usage_error(result)
    assert "cannot read arguments" in result.output


@pytest.mark.parametrize(
    "option,value",
    [("--kind", "unknown"), ("--offset", "-1"), ("--limit", "0"), ("--limit", "101")],
)
def test_invalid_discovery_options_are_usage_errors(option: str, value: str) -> None:
    assert_usage_error(RUNNER.invoke(app, ["harness", "operations", option, value]))


@pytest.mark.parametrize("from_file", [False, True])
def test_raw_argument_budget_precedes_json_parsing(tmp_path: Path, from_file: bool) -> None:
    # Tiny decoded input, but one byte over budget due to whitespace.
    raw = VALID + " " * (BUDGET + 1 - len(VALID))
    if from_file:
        source = tmp_path / "arguments.json"
        source.write_text(raw, encoding="utf-8")
        option = ["--arguments-file", str(source)]
    else:
        option = ["--arguments", raw]
    result = RUNNER.invoke(app, [*EXECUTE, *option])
    assert_usage_error(result)
    assert "2000000-byte" in result.output


@pytest.mark.parametrize("from_file", [False, True])
def test_valid_arguments_at_raw_byte_boundary(tmp_path: Path, from_file: bool) -> None:
    raw = VALID + " " * (BUDGET - len(VALID))
    if from_file:
        source = tmp_path / "arguments.json"
        source.write_text(raw, encoding="utf-8")
        option = ["--arguments-file", str(source)]
    else:
        option = ["--arguments", raw]
    result = RUNNER.invoke(app, [*EXECUTE, *option])
    assert result.exit_code == 0, result.exception
    assert json.loads(result.output)["result"]["returns"] == [None, 1.0]


def test_symlinked_host_workspace_root_is_resolved(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "data.csv").write_text("value\n2\n", encoding="utf-8")
    link = tmp_path / "workspace-link"
    try:
        link.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    result = RUNNER.invoke(
        app,
        [
            "harness",
            "execute-operation",
            "plugins.read_csv",
            "--arguments",
            '{"path":"data.csv"}',
            "--workspace-root",
            str(link),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.output)["result"]["rows"] == [{"value": "2"}]


def test_argument_file_read_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    class CheckedStream(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            assert size == BUDGET + 1
            return super().read(size)

    def open_arguments(path: Path, mode: str) -> CheckedStream:
        assert mode == "rb"
        return CheckedStream(b" " * (BUDGET + 10))

    monkeypatch.setattr(Path, "open", open_arguments)
    result = RUNNER.invoke(app, [*EXECUTE, "--arguments-file", "synthetic.json"])
    assert_usage_error(result)
    assert "2000000-byte" in result.output


def test_unreadable_argument_file_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_open(path: Path, mode: str) -> None:
        raise PermissionError("synthetic permission failure")

    monkeypatch.setattr(Path, "open", deny_open)
    result = RUNNER.invoke(app, [*EXECUTE, "--arguments-file", "synthetic.json"])
    assert_usage_error(result)
    assert "synthetic permission failure" in result.output


@pytest.mark.parametrize("from_file", [False, True])
def test_argument_budget_counts_utf8_bytes(tmp_path: Path, from_file: bool) -> None:
    raw = json.dumps({"prices": [1, 2], "extra": "é" * (BUDGET // 2)}, ensure_ascii=False)
    assert len(raw) < BUDGET < len(raw.encode("utf-8"))
    if from_file:
        source = tmp_path / "arguments.json"
        source.write_text(raw, encoding="utf-8")
        option = ["--arguments-file", str(source)]
    else:
        option = ["--arguments", raw]
    result = RUNNER.invoke(app, [*EXECUTE, *option])
    assert_usage_error(result)
    assert "2000000-byte" in result.output


@pytest.mark.parametrize("raw", ["{", "\udcff"], ids=["malformed", "surrogate"])
def test_invalid_inline_json_is_usage_error(raw: str) -> None:
    result = RUNNER.invoke(app, [*EXECUTE, "--arguments", raw])
    assert_usage_error(result)
    assert "cannot read arguments" in result.output


@pytest.mark.parametrize("kind", ["feature", "skill", "plugin"])
@pytest.mark.parametrize("limit", [1, 100])
def test_valid_discovery_options_preserve_pagination(kind: str, limit: int) -> None:
    result = RUNNER.invoke(
        app, ["harness", "operations", "--kind", kind, "--offset", "1", "--limit", str(limit)]
    )
    assert result.exit_code == 0, result.exception
    page = json.loads(result.output)
    assert page["offset"] == 1
    assert page["limit"] == limit
    assert 0 < len(page["results"]) <= limit
    assert all(row["kind"] == kind for row in page["results"])


def test_relative_host_workspace_root_is_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "data.csv").write_text("value\n2\n", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "harness",
            "execute-operation",
            "plugins.read_csv",
            "--arguments",
            '{"path":"data.csv"}',
            "--workspace-root",
            "workspace",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert json.loads(result.output)["result"]["rows"] == [{"value": "2"}]


def test_workspace_resolution_failure_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_resolve(path: Path) -> Path:
        raise OSError("synthetic resolution failure")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    result = RUNNER.invoke(app, [*EXECUTE, "--arguments", VALID, "--workspace-root", "workspace"])
    assert_usage_error(result)
    assert "cannot resolve workspace root" in result.output


@pytest.mark.parametrize("options", [[], ["--arguments", VALID, "--arguments-file", "unused.json"]])
def test_exactly_one_argument_source_required(options: list[str]) -> None:
    result = RUNNER.invoke(app, [*EXECUTE, *options])
    assert_usage_error(result)
    assert "Provide exactly one" in result.output
