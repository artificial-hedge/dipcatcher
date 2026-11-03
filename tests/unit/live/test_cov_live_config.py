"""Startup config validation for the armed gateway: refuse bad YAML, keys, values.

The live config is the only file that can name a venue, so startup must fail
closed on anything it does not fully understand: non-mapping YAML, unknown
keys, an unsupported schema version, non-positive or non-finite sizing, and
an unrecognised calendar. Pre-trade resolution must also stay inside the
config tree it was given.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quant_fund.live.config import (
    LiveConfig,
    load_gateway_pretrade,
    load_live_config,
    resolve_pretrade_path,
)

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
HMAC_KEY = b"unit-test-hmac-key-32-bytes-long"
BASE: dict[str, object] = {
    "schema_version": 1,
    "venue": "loopback",
    "starting_cash": 100_000.0,
    "pretrade_config": str(PRETRADE),
}


def _write(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_live_config_accepts_a_minimal_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "live.yaml", BASE)
    config = load_live_config(path)
    assert config.venue == "loopback"
    assert config.calendar == "us_rth"
    assert config.cash_tolerance == 1e-6


def test_non_mapping_yaml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_live_config(path)


def test_empty_yaml_is_refused_not_defaulted(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_live_config(path)


@pytest.mark.parametrize(
    "override",
    [
        {"extra_venue": "alpaca-live"},
        {"schema_version": 2},
        {"schema_version": "1"},
        {"venue": "nasdaq"},
        {"calendar": "24x7"},
    ],
)
def test_unknown_keys_versions_and_venues_are_refused(
    tmp_path: Path, override: dict[str, object]
) -> None:
    path = _write(tmp_path / "live.yaml", {**BASE, **override})
    with pytest.raises(ValidationError):
        load_live_config(path)


@pytest.mark.parametrize(
    "starting_cash",
    [0.0, -1.0, float("inf"), float("nan"), "not-a-number"],
)
def test_starting_cash_must_be_positive_and_finite(tmp_path: Path, starting_cash: object) -> None:
    path = _write(tmp_path / "live.yaml", {**BASE, "starting_cash": starting_cash})
    with pytest.raises(ValidationError):
        load_live_config(path)


def test_negative_tolerances_are_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "live.yaml",
        {**BASE, "cash_tolerance": -1.0, "position_tolerance": -1.0},
    )
    with pytest.raises(ValidationError):
        load_live_config(path)


def test_absolute_pretrade_path_is_used_verbatim(tmp_path: Path) -> None:
    config = LiveConfig.model_validate({**BASE, "pretrade_config": str(PRETRADE)})
    assert resolve_pretrade_path(config, tmp_path / "live.yaml") == PRETRADE


def test_relative_pretrade_path_resolves_beside_the_config(tmp_path: Path) -> None:
    local = tmp_path / "limits.yaml"
    _write(local, {})
    config = LiveConfig.model_validate({**BASE, "pretrade_config": "limits.yaml"})
    assert resolve_pretrade_path(config, tmp_path / "live.yaml") == local


def test_relative_pretrade_path_prefers_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    in_cwd = _write(cwd / "limits.yaml", {})
    _write(config_dir / "limits.yaml", {})
    config = LiveConfig.model_validate({**BASE, "pretrade_config": "limits.yaml"})
    monkeypatch.chdir(cwd)
    assert resolve_pretrade_path(config, config_dir / "live.yaml") == in_cwd


def test_gateway_pretrade_keeps_the_file_session_by_default(tmp_path: Path) -> None:
    config = load_live_config(_write(tmp_path / "live.yaml", BASE))
    loaded, digest, signature = load_gateway_pretrade(
        config, tmp_path / "live.yaml", hmac_key=HMAC_KEY
    )
    from quant_fund.pretrade.config import load_pretrade_config

    reference, ref_digest, ref_signature = load_pretrade_config(PRETRADE, hmac_key=HMAC_KEY)
    assert loaded.session == reference.session
    assert (digest, signature) == (ref_digest, ref_signature)


def test_continuous_calendar_replaces_the_session_and_resigns(tmp_path: Path) -> None:
    config = load_live_config(_write(tmp_path / "live.yaml", {**BASE, "calendar": "continuous"}))
    loaded, digest, signature = load_gateway_pretrade(
        config, tmp_path / "live.yaml", hmac_key=HMAC_KEY
    )
    assert loaded.session.timezone == "UTC"
    assert loaded.session.open_minute == 0
    assert loaded.session.close_minute == 24 * 60
    assert loaded.session.weekends_open is True
    from quant_fund.pretrade.config import sign_config

    assert (digest, signature) == sign_config(loaded, HMAC_KEY)


def test_missing_pretrade_file_fails_before_arming(tmp_path: Path) -> None:
    config = load_live_config(
        _write(tmp_path / "live.yaml", {**BASE, "pretrade_config": "absent.yaml"})
    )
    with pytest.raises(FileNotFoundError):
        load_gateway_pretrade(config, tmp_path / "live.yaml", hmac_key=HMAC_KEY)


def test_short_hmac_key_is_refused(tmp_path: Path) -> None:
    config = load_live_config(_write(tmp_path / "live.yaml", BASE))
    with pytest.raises(ValueError, match="hmac_key must be at least 16 bytes"):
        load_gateway_pretrade(config, tmp_path / "live.yaml", hmac_key=b"short")
