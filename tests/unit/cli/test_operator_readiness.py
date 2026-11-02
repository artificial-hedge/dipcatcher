"""Fail-closed operator CLI and diagnostic regressions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from fx1.cli import app
from fx1.cli_errors import user_error_message
from fx1.doctor import collect_status
from fx1.selftest import run_selftest

RUNNER = CliRunner()


def _assert_clean_usage_error(result: Result, message: str) -> None:
    assert result.exit_code == 2, result.exception
    assert message in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["harness", "describe-operation", "bogus"], "unregistered operation"),
        (
            ["harness", "execute-operation", "bogus", "--arguments", "{}"],
            "unregistered operation",
        ),
        (["sources", "probe", "bogus"], "unknown datasource"),
        (["sources", "describe", "bogus"], "unknown datasource"),
        (["sources", "fetch", "bogus", "--api", "x"], "unknown datasource"),
        (["corpus", "ingest-source", "bogus", "--api", "x"], "unknown datasource"),
    ],
)
def test_unknown_registry_names_are_clean_usage_errors(args: list[str], message: str) -> None:
    _assert_clean_usage_error(RUNNER.invoke(app, args), message)


@pytest.mark.parametrize("command", [["sources", "fetch"], ["corpus", "ingest-source"]])
@pytest.mark.parametrize("raw", ["[]", "null", '"text"'])
def test_source_params_must_be_json_objects(command: list[str], raw: str, tmp_path: Path) -> None:
    args = [*command, "wind", "--api", "synthetic", "--params-json", raw]
    if command[0] == "corpus":
        args.extend(
            [
                "--out",
                str(tmp_path / "corpus.jsonl"),
                "--ledger-path",
                str(tmp_path / "ledger.jsonl"),
            ]
        )
    _assert_clean_usage_error(RUNNER.invoke(app, args), "--params-json")


def test_deep_source_json_is_a_clean_usage_error() -> None:
    raw = '{"x":' * 20_000 + "null" + "}" * 20_000
    result = RUNNER.invoke(app, ["sources", "fetch", "wind", "--api", "x", "--params-json", raw])
    _assert_clean_usage_error(result, "too deeply nested")


def test_operation_data_errors_are_clean_usage_errors(tmp_path: Path) -> None:
    (tmp_path / "bad.csv").write_text("a,b\n1,2,3\n", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "harness",
            "execute-operation",
            "plugins.read_csv",
            "--arguments",
            '{"path":"bad.csv"}',
            "--workspace-root",
            str(tmp_path),
        ],
    )
    _assert_clean_usage_error(result, "cannot execute operation")


def test_operation_file_errors_are_clean_usage_errors(tmp_path: Path) -> None:
    result = RUNNER.invoke(
        app,
        [
            "harness",
            "execute-operation",
            "plugins.read_csv",
            "--arguments",
            '{"path":"missing.csv"}',
            "--workspace-root",
            str(tmp_path),
        ],
    )
    _assert_clean_usage_error(result, "cannot execute operation")


def test_doctor_reports_invalid_utf8_as_unreadable(tmp_path: Path) -> None:
    data = tmp_path / "data" / "fx1"
    data.mkdir(parents=True)
    (data / "corpus.jsonl").write_bytes(b"\xff")
    status = collect_status(tmp_path)
    assert status["data_fx1_corpus"] == "unreadable"
    assert status["data_fx1_eval"] == "missing"


def test_doctor_reports_receipt_scan_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "receipts").mkdir()

    def fail_rglob(path: Path, pattern: str) -> object:
        raise OSError("synthetic scan failure")

    monkeypatch.setattr(Path, "rglob", fail_rglob)
    assert collect_status(tmp_path)["receipts"] == "unreadable"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (OSError("device offline"), "I/O error: device offline"),
        (RecursionError(), "input is too deeply nested"),
    ],
)
def test_common_input_failures_have_operator_messages(error: Exception, message: str) -> None:
    assert user_error_message(error) == message


def test_selftest_bind_failure_returns_failed_report(monkeypatch: pytest.MonkeyPatch) -> None:
    class DenyServer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("synthetic bind failure")

    monkeypatch.setattr("fx1.selftest.ThreadingHTTPServer", DenyServer)
    report = run_selftest()
    assert report.as_dict() == {
        "mode": "local",
        "ok": False,
        "checks": [{"name": "boot", "ok": False, "detail": "OSError: synthetic bind failure"}],
    }


def test_selftest_runtime_failure_restores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "synthetic-preserved-value")

    def fail_create_app(**kwargs: object) -> object:
        raise OSError("synthetic app failure")

    monkeypatch.setattr("fx1.serve.api.create_app", fail_create_app)
    report = run_selftest()
    assert not report.ok
    assert report.checks[-1].name == "local_runtime"
    assert "synthetic app failure" in report.checks[-1].detail
    assert os.environ["MOONSHOT_API_KEY"] == "synthetic-preserved-value"


def test_remote_selftest_setup_failure_returns_failed_report() -> None:
    report = run_selftest(remote="not-a-url")
    assert not report.ok
    assert report.checks[0].name == "client_setup"


@pytest.mark.parametrize("args", [["--timeout", "0"], ["--state-dir", "unused"]])
def test_remote_selftest_rejects_invalid_option_combinations(args: list[str]) -> None:
    result = RUNNER.invoke(app, ["harness", "selftest", "--remote", "http://localhost", *args])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
