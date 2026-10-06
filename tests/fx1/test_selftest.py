"""Selftest contract — the golden-path smoke must itself be green, honest
about partial coverage, and reachable from the CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from fx1.cli import app
from fx1.selftest import SelftestReport, run_selftest


def test_local_selftest_all_green(tmp_path) -> None:
    """Boots a stub engine + the production app on loopback and walks the
    whole golden path, including a real process restart on the journal."""
    report = run_selftest(state_dir=str(tmp_path / "state"))
    failed = [c.name for c in report.checks if not c.ok]
    assert not failed, failed
    names = {c.name for c in report.checks}
    assert "restart_recovers_job" in names
    assert "inprocess_parity" in names


def test_local_selftest_without_state_dir_skips_restart() -> None:
    report = run_selftest()
    failed = [c.name for c in report.checks if not c.ok]
    assert not failed, failed
    names = {c.name for c in report.checks}
    assert "restart_recovers_job" not in names


def test_report_shape_is_stable() -> None:
    report = SelftestReport(mode="local")
    report.note("probe", True, "d")
    out = report.as_dict()
    assert out["mode"] == "local"
    assert out["ok"] is True
    assert out["checks"] == [{"name": "probe", "ok": True, "detail": "d"}]


def test_report_not_ok_propagates() -> None:
    report = SelftestReport(mode="local")
    report.note("probe", False)
    assert report.as_dict()["ok"] is False


_ENV_KEYS = (
    "FX1_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
    "MOONSHOT_API_KEY",
)


def test_selftest_restores_environment(tmp_path, monkeypatch) -> None:
    """A local run must leave os.environ byte-identical — the BYOK opt-in and
    the MOONSHOT_API_KEY pop are probe scaffolding, not caller state."""
    import os

    monkeypatch.setenv("MOONSHOT_API_KEY", "caller-moonshot")
    monkeypatch.delenv("FX1_BYOK_ALLOW_PRIVATE_NETWORKS", raising=False)
    before = {k: os.environ.get(k) for k in _ENV_KEYS}
    report = run_selftest()
    assert report.ok
    assert {k: os.environ.get(k) for k in _ENV_KEYS} == before


def test_selftest_restores_environment_on_failure(monkeypatch) -> None:
    """The env restore lives in ``finally`` — a crashed boot must not strand
    the stubbed BYOK endpoint or eat the caller's MOONSHOT_API_KEY."""
    import os

    import fx1.serve.api as api_mod

    monkeypatch.setenv("MOONSHOT_API_KEY", "caller-moonshot")
    monkeypatch.delenv("FX1_BYOK_ALLOW_PRIVATE_NETWORKS", raising=False)
    before = {k: os.environ.get(k) for k in _ENV_KEYS}

    def _boom(**kwargs):  # noqa: ANN001
        raise RuntimeError("boot exploded")

    monkeypatch.setattr(api_mod, "create_app", _boom)
    with pytest.raises(RuntimeError, match="boot exploded"):
        run_selftest()
    assert {k: os.environ.get(k) for k in _ENV_KEYS} == before


def test_cli_selftest_local(tmp_path) -> None:
    cr = CliRunner().invoke(app, ["harness", "selftest", "--state-dir", str(tmp_path / "s")])
    assert cr.exit_code == 0, cr.output
    payload = json.loads(cr.output)
    assert payload["ok"] is True
    assert payload["mode"] == "local"


def test_cli_selftest_remote_unreachable_reports_failure() -> None:
    cr = CliRunner().invoke(
        app,
        [
            "harness",
            "selftest",
            "--remote",
            "http://127.0.0.1:1",
            "--timeout",
            "2",
        ],
    )
    assert cr.exit_code == 2
    payload = json.loads(cr.output)
    assert payload["ok"] is False
    assert payload["mode"] == "remote"
