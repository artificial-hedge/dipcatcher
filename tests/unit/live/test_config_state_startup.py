from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.live.brokers import LoopbackBroker
from quant_fund.live.config import LiveConfig
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
RUNNER = CliRunner()
TEST_KEY = b"0123456789abcdef0123456789abcdef"


def _invalid_pretrade(path: Path) -> None:
    path.write_text("schema_version: 1\n", encoding="utf-8")


def _open_gateway(state: Path, pretrade: Path) -> object:
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "loopback",
            "starting_cash": 100_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(pretrade),
            "state_dir": str(state),
        }
    )
    return open_session(
        config,
        pretrade,
        state_dir=state,
        now_ns=lambda: 0,
        broker=LoopbackBroker(cash=100_000.0),
    )


def test_desk_invalid_config_does_not_create_state_or_key(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    config = tmp_path / "invalid.yaml"
    _invalid_pretrade(config)

    result = RUNNER.invoke(
        app,
        ["live", "status", "--root", str(root), "--config", str(config)],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
    assert not root.exists()


def test_desk_invalid_config_preserves_an_existing_key(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    root.mkdir()
    key_path = root / "hmac.key"
    key_path.write_bytes(TEST_KEY)
    config = tmp_path / "invalid.yaml"
    _invalid_pretrade(config)

    result = RUNNER.invoke(
        app,
        ["live", "status", "--root", str(root), "--config", str(config)],
    )

    assert result.exit_code == 1
    assert key_path.read_bytes() == TEST_KEY
    assert not (root / "state.json").exists()


def test_desk_valid_config_reuses_its_persisted_key(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    command = ["live", "status", "--root", str(root), "--config", str(PRETRADE)]

    first = RUNNER.invoke(app, command)
    key_path = root / "hmac.key"
    first_key = key_path.read_bytes()
    second = RUNNER.invoke(app, command)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert key_path.read_bytes() == first_key


def test_gateway_invalid_pretrade_does_not_create_state_or_key(tmp_path: Path) -> None:
    state = tmp_path / "gateway-state"
    invalid = tmp_path / "invalid.yaml"
    _invalid_pretrade(invalid)

    with pytest.raises(ValueError):
        _open_gateway(state, invalid)

    assert not state.exists()


def test_gateway_invalid_pretrade_preserves_an_existing_key(tmp_path: Path) -> None:
    state = tmp_path / "gateway-state"
    state.mkdir()
    key_path = state / "hmac.key"
    key_path.write_bytes(TEST_KEY)
    invalid = tmp_path / "invalid.yaml"
    _invalid_pretrade(invalid)

    with pytest.raises(ValueError):
        _open_gateway(state, invalid)

    assert key_path.read_bytes() == TEST_KEY
    assert not (state / "state.json").exists()


def test_gateway_valid_pretrade_reuses_its_persisted_key(tmp_path: Path) -> None:
    state = tmp_path / "gateway-state"

    _open_gateway(state, PRETRADE)
    key_path = state / "hmac.key"
    first_key = key_path.read_bytes()
    _open_gateway(state, PRETRADE)

    assert key_path.read_bytes() == first_key
