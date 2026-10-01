"""HTTP boundary and replay correctness, using explicitly synthetic code/tape."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from fx1.serve.strategy_api import create_app
from fx1.strategy import GeneratedCode, StrategySpec, verify_strategy_receipt


def request_payload():
    spec = StrategySpec(
        "fixture", "Fixture", "Fixed synthetic probabilities; no semantic judge.", {}
    )
    start = datetime(2020, 1, 1, tzinfo=UTC)
    code = "def strategy(history): return {'probability_up': 0.7, 'target_weight': 0.5}"
    generated = GeneratedCode(
        spec.id,
        spec.sha256,
        code,
        hashlib.sha256(code.encode()).hexdigest(),
        start.isoformat(),
        "SYNTHETIC_http_fixture_not_model_inference",
        0.0,
    )
    clocks = [(start + timedelta(days=i)).isoformat() for i in range(4)]
    return {
        "spec": spec.payload(),
        "generated": asdict(generated),
        "bars": [
            {"event_time": clock, "available_time": clock, "close": close}
            for clock, close in zip(clocks, [100, 101, 100, 102], strict=True)
        ],
        "decision_times": clocks[:-1],
        "data_source": "SYNTHETIC_http_tape",
        "synthetic": True,
        "cost_bps": 2.0,
    }


def test_http_receipt_matches_independent_scores_and_replays_from_saved_json(tmp_path):
    result = TestClient(create_app()).post("/v1/strategy/replay", json=request_payload())
    assert result.status_code == 200, result.text
    receipt = result.json()
    assert receipt["replay"]["brier_score"] == pytest.approx((0.09 + 0.49 + 0.09) / 3)
    assert receipt["replay"]["transaction_cost_bps"] == 1.0
    assert receipt["synthetic"] and not receipt["market_evidence"]
    assert not receipt["specification_judge_passed"]
    assert receipt["evaluation"]["specification_correct"] is None
    assert not receipt["generation_identity_independently_verified"]
    saved = tmp_path / "receipt.json"
    saved.write_text(json.dumps(receipt))
    verification = verify_strategy_receipt(saved)
    assert verification["replay_reproduced"] and verification["implementation_matches_current"]
    assert result.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("corruption", ["bool_synthetic", "naive_clock", "wrong_spec", "extra"])
def test_invalid_payload_rejects_without_semantic_pass(corruption):
    payload = request_payload()
    if corruption == "bool_synthetic":
        payload["synthetic"] = "true"
    elif corruption == "naive_clock":
        payload["decision_times"][0] = "2020-01-01T00:00:00"
    elif corruption == "wrong_spec":
        payload["spec"]["description"] = "Unbound changed instruction."
    else:
        payload["broker"] = "forbidden"
    result = TestClient(create_app()).post("/v1/strategy/replay", json=payload)
    assert result.status_code == 422


def test_hostile_code_rejects_without_python_execution(tmp_path):
    marker = tmp_path / "escaped"
    payload = request_payload()
    code = f"open({str(marker)!r}, 'w').write('escape')"
    payload["generated"].update(
        code_text=code, code_sha256=hashlib.sha256(code.encode()).hexdigest()
    )
    result = TestClient(create_app()).post("/v1/strategy/replay", json=payload)
    assert result.status_code == 422
    assert not marker.exists()


def test_api_capabilities_do_not_claim_generation_or_judgment():
    result = TestClient(create_app()).get("/v1/strategy/capabilities").json()
    assert result["replay"]
    assert not any(result[name] for name in ("generation", "semantic_judge", "trained_checkpoint"))


def test_key_protects_replay_and_documentation(monkeypatch):
    monkeypatch.setenv("FX1_API_KEY", "fixture-only-not-secret")
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 401
    assert client.post("/v1/strategy/replay", json=request_payload()).status_code == 401
    assert (
        client.get(
            "/v1/strategy/capabilities", headers={"X-API-Key": "fixture-only-not-secret"}
        ).status_code
        == 200
    )


def test_non_ascii_api_key_is_rejected_without_server_error(monkeypatch):
    monkeypatch.setenv("FX1_API_KEY", "fixture-only-not-secret")
    client = TestClient(create_app())
    response = client.get("/v1/strategy/capabilities", headers={b"x-api-key": b"\xff"})
    assert response.status_code == 401
    assert (
        client.get(
            "/v1/strategy/capabilities", headers={"X-API-Key": "fixture-only-not-secret"}
        ).status_code
        == 200
    )


def test_remote_clients_refused_without_key(monkeypatch):
    monkeypatch.delenv("FX1_API_KEY", raising=False)
    client = TestClient(create_app(), client=("203.0.113.9", 1234))
    assert client.get("/v1/strategy/capabilities").status_code == 403


def test_oversized_body_rejected_before_interpreter():
    response = TestClient(create_app()).post(
        "/v1/strategy/replay", content=b"x" * 65537, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


def test_oversized_stream_cannot_bypass_length_limit():
    response = TestClient(create_app()).post(
        "/v1/strategy/replay",
        content=iter([b"x" * 32768, b"x" * 32769]),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
