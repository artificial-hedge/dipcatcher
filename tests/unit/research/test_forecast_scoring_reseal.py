"""Resealed forecast_scoring.v1 edits fail verify-receipt where the body can say so."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.forecast_scoring import score_forecasts
from quant_fund.schemas.receipt_v2_leaf import seal_receipt, verify_receipt_file

_TASK = "volatility"


def _ts(index: int) -> str:
    return f"2020-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00+00:00"


def _write_csv(path: Path, forecast: float) -> None:
    rows = [f"{_ts(index)},1.0,{forecast}" for index in range(30)]
    path.write_text(
        "timestamp,realized,forecast_var\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def _sealed(tmp_path: Path, forecast: float = 2.0) -> dict[str, object]:
    data = tmp_path / "v.csv"
    _write_csv(data, forecast)
    outcome = score_forecasts(data, task=_TASK, out_dir=tmp_path / "out")
    body = json.loads(outcome.receipt_path.read_text(encoding="utf-8"))
    assert isinstance(body, dict)
    return body


def _errors(body: dict[str, object], tmp_path: Path, name: str) -> list[str]:
    path = tmp_path / name
    path.write_text(json.dumps(seal_receipt(body)), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is False
    return list(result["errors"])


def test_valid_round_trip_is_deterministic(tmp_path: Path) -> None:
    data = tmp_path / "v.csv"
    _write_csv(data, 1.4)
    first = score_forecasts(data, task=_TASK, out_dir=tmp_path / "a")
    second = score_forecasts(data, task=_TASK, out_dir=tmp_path / "b")
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    verified = verify_receipt_file(first.receipt_path)
    assert verified["valid"] is True, verified["errors"]
    assert "kind_fingerprint_mismatch" not in verified["errors"]


def test_resealed_dataset_hash_fails(tmp_path: Path) -> None:
    body = _sealed(tmp_path)
    digest = str(body["dataset_hash"])
    body["dataset_hash"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    errors = _errors(body, tmp_path, "hash.json")
    assert "dataset_hash_mismatch" in errors


def test_resealed_outer_kind_fails(tmp_path: Path) -> None:
    body = _sealed(tmp_path)
    body["kind"] = "forged.v1"
    errors = _errors(body, tmp_path, "kind.json")
    assert "kind_fingerprint_mismatch" in errors
    assert "kind_not_forecast_scoring" in errors


def test_resealed_inner_kind_fails(tmp_path: Path) -> None:
    body = _sealed(tmp_path)
    payload = body["payload"]
    assert isinstance(payload, dict)
    payload["kind"] = "forged.v1"
    errors = _errors(body, tmp_path, "inner-kind.json")
    assert "payload_kind_mismatch" in errors


def test_resealed_honesty_flags_fail(tmp_path: Path) -> None:
    research = _sealed(tmp_path)
    payload = research["payload"]
    assert isinstance(payload, dict)
    payload["research_only"] = False
    assert "research_only_not_true" in _errors(research, tmp_path, "research.json")
    live = _sealed(tmp_path)
    inner = live["payload"]
    assert isinstance(inner, dict)
    inner["live_pnl_claim"] = True
    joined = " ".join(_errors(live, tmp_path, "live.json"))
    assert "live_pnl_claim" in joined


def test_resealed_limitations_fail(tmp_path: Path) -> None:
    body = _sealed(tmp_path)
    payload = body["payload"]
    assert isinstance(payload, dict)
    notes = payload["limitations"]
    assert isinstance(notes, list) and notes
    notes[0] = str(notes[0]) + " tampered"
    assert "limitations_mismatch" in _errors(body, tmp_path, "limits.json")


def test_resealed_params_fail(tmp_path: Path) -> None:
    body = _sealed(tmp_path)
    payload = body["payload"]
    assert isinstance(payload, dict)
    params = payload["params"]
    assert isinstance(params, dict)
    params["seed"] = int(params["seed"]) + 1
    assert "params_hash_mismatch" in _errors(body, tmp_path, "params.json")


def test_resealed_primary_score_fails(tmp_path: Path) -> None:
    body = _sealed(tmp_path)
    payload = body["payload"]
    assert isinstance(payload, dict)
    scores = payload["scores"]
    assert isinstance(scores, dict)
    scores["qlike"] = float(scores["qlike"]) + 3.0
    assert "primary_score_mismatch" in _errors(body, tmp_path, "qlike.json")


def test_resealed_non_primary_score_is_not_recomputed(tmp_path: Path) -> None:
    """``mse`` is not the mean stored in the block table. The source rows are absent."""
    body = _sealed(tmp_path)
    payload = body["payload"]
    assert isinstance(payload, dict)
    scores = payload["scores"]
    assert isinstance(scores, dict)
    scores["mse"] = float(scores["mse"]) + 4.0
    path = tmp_path / "mse.json"
    path.write_text(json.dumps(seal_receipt(body)), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]


@pytest.mark.parametrize("forecast", [1.0, 2.5, 8.0])
def test_primary_reconstruction_accepts_real_receipts(tmp_path: Path, forecast: float) -> None:
    body = _sealed(tmp_path, forecast)
    path = tmp_path / "round.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    result = verify_receipt_file(path)
    assert result["valid"] is True, result["errors"]
    assert body["kind"] == "forecast_scoring.v1"
