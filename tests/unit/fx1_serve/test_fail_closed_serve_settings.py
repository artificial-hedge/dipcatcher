"""Fail-closed serve settings: timeouts and a present weights manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.modelcard import EvalDelta, ModelCard
from fx1.serve.backends import (
    LOCAL_START_TIMEOUT_S_ENV,
    LOCAL_TIMEOUT_S_ENV,
    HostedK3Backend,
    LocalFx1Backend,
    OpenAICompatBackend,
    _env_float,
)


def _card(path: Path) -> None:
    card = ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b5942241..f0e1d2c3",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    )
    card.save(path / "modelcard.json")


def _quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FX1_SIGNING_KEY",
        "FX1_LOCAL_SERVE_URL",
        "FX1_LOCAL_SERVE_CMD",
        "FX1_CHECKPOINT_DIR",
        LOCAL_TIMEOUT_S_ENV,
        LOCAL_START_TIMEOUT_S_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_env_float_uses_default_for_blank_and_keeps_a_finite_override() -> None:
    assert _env_float("FX1_UNIT_TIMEOUT_UNSET", None, 120.0) == 120.0
    assert _env_float("FX1_UNIT_TIMEOUT_UNSET", "", 120.0) == 120.0
    assert _env_float("FX1_UNIT_TIMEOUT_UNSET", 15.0, 120.0) == 15.0


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "0", "-1", "nope", "  ", True])
def test_env_float_rejects_non_finite_and_non_positive(raw: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _env_float("FX1_LOCAL_TIMEOUT_S", raw, 120.0)


def test_hosted_and_byok_timeouts_reject_non_finite() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        HostedK3Backend(api_key="k", timeout_s=float("nan"))
    with pytest.raises(ValueError, match="positive finite"):
        HostedK3Backend(api_key="k", timeout_s=float("inf"))
    with pytest.raises(ValueError, match="positive finite"):
        OpenAICompatBackend(
            base_url="https://example.com/v1",
            api_key="k",
            model="m",
            timeout_s=0,
        )


def test_local_timeout_env_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet_env(monkeypatch)
    _card(tmp_path)
    monkeypatch.setenv(LOCAL_TIMEOUT_S_ENV, "nan")
    with pytest.raises(ValueError, match="FX1_LOCAL_TIMEOUT_S"):
        LocalFx1Backend(tmp_path)
    monkeypatch.setenv(LOCAL_TIMEOUT_S_ENV, "30")
    monkeypatch.setenv(LOCAL_START_TIMEOUT_S_ENV, "inf")
    with pytest.raises(ValueError, match="FX1_LOCAL_START_TIMEOUT_S"):
        LocalFx1Backend(tmp_path)


def test_absent_manifest_still_constructs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet_env(monkeypatch)
    _card(tmp_path)
    backend = LocalFx1Backend(tmp_path)
    assert backend._expected_weights_sha is None


def test_manifest_without_pin_stays_unpinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _quiet_env(monkeypatch)
    _card(tmp_path)
    (tmp_path / "weights.manifest.json").write_text("{}\n", encoding="utf-8")
    backend = LocalFx1Backend(tmp_path)
    assert backend._expected_weights_sha is None


def test_present_manifest_must_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _quiet_env(monkeypatch)
    _card(tmp_path)
    manifest = tmp_path / "weights.manifest.json"
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not JSON"):
        LocalFx1Backend(tmp_path)
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        LocalFx1Backend(tmp_path)
    manifest.write_bytes(b"\xff")
    with pytest.raises(RuntimeError, match="malformed"):
        LocalFx1Backend(tmp_path)


def test_manifest_pin_must_be_one_hex_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _quiet_env(monkeypatch)
    _card(tmp_path)
    manifest = tmp_path / "weights.manifest.json"
    manifest.write_text('{"weights_sha256": "nope"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="64-character hex"):
        LocalFx1Backend(tmp_path)
    pinned = "ab" * 32
    manifest.write_text(
        '{"weights_sha256": "00", "weights_sha256": "' + pinned + '"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate key"):
        LocalFx1Backend(tmp_path)
    manifest.write_text('{"weights_sha256": "' + pinned + '"}\n', encoding="utf-8")
    backend = LocalFx1Backend(tmp_path)
    assert backend._expected_weights_sha == pinned
