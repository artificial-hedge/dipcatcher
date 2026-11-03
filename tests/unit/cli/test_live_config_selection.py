from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner, Result

from quant_fund.cli.main import app

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
RUNNER = CliRunner()


def _pretrade() -> dict[str, Any]:
    raw = yaml.safe_load(PRETRADE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return {str(key): value for key, value in raw.items()}


def _gateway(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "venue": "loopback",
        "starting_cash": 100_000.0,
        "calendar": "us_rth",
        "pretrade_config": str(PRETRADE),
        "state_dir": str(tmp_path / "gateway-state"),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _assert_config_error(result: Result, message: str, *paths: Path) -> None:
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert "Traceback" not in result.output
    for path in paths:
        assert not path.exists()


@pytest.mark.parametrize(
    "missing_fields",
    [("starting_cash",), ("pretrade_config",), ("starting_cash", "pretrade_config")],
)
def test_incomplete_explicit_gateway_never_falls_back_to_desk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_fields: tuple[str, ...]
) -> None:
    payload = _gateway(tmp_path)
    for field in missing_fields:
        del payload[field]
    path = tmp_path / "live.yaml"
    _write(path, payload)
    root = tmp_path / "desk"
    monkeypatch.setenv("DIP_LIVE_ROOT", str(root))
    state = tmp_path / "gateway-state"

    result = RUNNER.invoke(app, ["live", "status", "--config", str(path)])

    assert result.exit_code == 1
    body = json.loads(result.stdout)
    assert body["ok"] is False
    assert body["research_evidence"] is False
    assert missing_fields[0] in body["error"]
    assert not root.exists()
    assert not state.exists()


def test_valid_explicit_gateway_still_selects_gateway(tmp_path: Path) -> None:
    path = tmp_path / "live.yaml"
    _write(path, _gateway(tmp_path))

    result = RUNNER.invoke(app, ["live", "status", "--config", str(path)])

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["ok"] is True
    assert body["venue"] == "loopback"
    assert body["research_evidence"] is False


def test_valid_explicit_pretrade_with_root_still_selects_desk(tmp_path: Path) -> None:
    path = tmp_path / "pretrade.yaml"
    _write(path, _pretrade())
    root = tmp_path / "desk"

    result = RUNNER.invoke(
        app,
        ["live", "status", "--config", str(path), "--root", str(root)],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["schema"] == "live_desk_report.v1"
    assert body["data_label"] == "SHADOW"
    assert body["funded"] is False


def test_explicit_gateway_rejects_desk_root_before_state_creation(tmp_path: Path) -> None:
    path = tmp_path / "live.yaml"
    payload = _gateway(tmp_path)
    _write(path, payload)
    root = tmp_path / "desk"

    result = RUNNER.invoke(
        app,
        ["live", "status", "--config", str(path), "--root", str(root)],
    )

    _assert_config_error(
        result,
        "desk and gateway options cannot be combined",
        root,
        tmp_path / "gateway-state",
    )


def test_explicit_gateway_rejects_desk_venue_before_state_creation(tmp_path: Path) -> None:
    path = tmp_path / "live.yaml"
    _write(path, _gateway(tmp_path))
    state = tmp_path / "gateway-state"

    result = RUNNER.invoke(
        app,
        ["live", "arm", "--config", str(path), "--venue", "loopback"],
    )

    _assert_config_error(result, "desk and gateway options cannot be combined", state)


def test_explicit_pretrade_rejects_gateway_state_dir(tmp_path: Path) -> None:
    path = tmp_path / "pretrade.yaml"
    _write(path, _pretrade())
    state = tmp_path / "gateway-state"

    result = RUNNER.invoke(
        app,
        ["live", "status", "--config", str(path), "--state-dir", str(state)],
    )

    _assert_config_error(result, "gateway options require a gateway config", state)


def test_mixed_desk_and_gateway_flags_fail_before_state_creation(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    state = tmp_path / "gateway-state"
    commands = (
        ["live", "status", "--root", str(root), "--state-dir", str(state)],
        [
            "live",
            "submit",
            "--root",
            str(root),
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--mark",
            "100",
            "--limit",
            "100",
        ],
        [
            "live",
            "kill",
            "--root",
            str(root),
            "--reason",
            "halt",
            "--at",
            "2024-01-03T15:00:00-05:00",
        ],
        ["live", "cancel", "--root", str(root), "--client-order-id", "dc-1"],
    )

    for command in commands:
        result = RUNNER.invoke(app, command)
        _assert_config_error(
            result,
            "desk and gateway options cannot be combined",
            root,
            state,
        )


def test_explicit_desk_selection_allows_the_shared_actor_flag(tmp_path: Path) -> None:
    path = tmp_path / "pretrade.yaml"
    _write(path, _pretrade())
    root = tmp_path / "desk"

    result = RUNNER.invoke(
        app,
        [
            "live",
            "kill",
            "--config",
            str(path),
            "--root",
            str(root),
            "--actor",
            "ops",
            "--reason",
            "halt",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["schema"] == "live_desk_report.v1"
    assert body["status"] == "halted"


@pytest.mark.parametrize("explicit_pretrade", [False, True])
def test_desk_is_the_default_without_gateway_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_pretrade: bool
) -> None:
    root = tmp_path / "desk"
    monkeypatch.setenv("DIP_LIVE_ROOT", str(root))
    monkeypatch.chdir(ROOT)
    command = ["live", "status"]
    if explicit_pretrade:
        path = tmp_path / "pretrade.yaml"
        _write(path, _pretrade())
        command.extend(["--config", str(path)])

    result = RUNNER.invoke(app, command)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema"] == "live_desk_report.v1"


def test_state_dir_selects_the_default_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    state = tmp_path / "gateway-state"

    result = RUNNER.invoke(app, ["live", "status", "--state-dir", str(state)])

    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["ok"] is True
    assert body["venue"] == "loopback"
    assert (state / "hmac.key").is_file()


def test_bare_actor_requires_an_explicit_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "desk"
    monkeypatch.setenv("DIP_LIVE_ROOT", str(root))
    monkeypatch.chdir(tmp_path)

    result = RUNNER.invoke(app, ["live", "kill", "--actor", "ops", "--reason", "halt"])

    _assert_config_error(
        result,
        "--actor requires an explicit desk or gateway selection",
        root,
        tmp_path / "data",
    )


def test_mixed_config_schemas_do_not_echo_values_or_create_state(tmp_path: Path) -> None:
    marker = "synthetic-schema-selection-sensitive-value"
    payload = _pretrade()
    payload["starting_cash"] = marker
    path = tmp_path / "ambiguous.yaml"
    _write(path, payload)
    root = tmp_path / "desk"

    result = RUNNER.invoke(
        app,
        ["live", "status", "--config", str(path), "--root", str(root)],
    )

    _assert_config_error(result, "config mixes desk and gateway schemas", root)
    assert marker not in result.output


def test_malformed_explicit_config_does_not_echo_contents(tmp_path: Path) -> None:
    marker = "synthetic-malformed-selection-sensitive-value"
    path = tmp_path / "live.yaml"
    path.write_text(f'venue: "{marker}\n', encoding="utf-8")
    root = tmp_path / "desk"

    result = RUNNER.invoke(
        app,
        ["live", "status", "--config", str(path), "--root", str(root)],
    )

    _assert_config_error(result, "config contains invalid YAML", root)
    assert marker not in result.output


def test_missing_explicit_config_fails_before_state_creation(tmp_path: Path) -> None:
    root = tmp_path / "desk"

    result = RUNNER.invoke(
        app,
        [
            "live",
            "status",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--root",
            str(root),
        ],
    )

    _assert_config_error(result, "config must be an existing file", root)
