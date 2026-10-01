"""Learned-router HTTP pilot with finalized synthetic logs and inherited auth."""

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from quant_fund.api.app import app
from quant_fund.execution.ml_router import FillFeatures, MLFillRouter


def request_payload():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    features = asdict(FillFeatures(-1.0, 2.0, 10.0, 100.0, 10.0, 30.0))
    cutoff = start + timedelta(minutes=40)
    return {
        "observations": [
            {
                "order_id": str(i),
                "venue": "A",
                "decision_time": (start + timedelta(minutes=i)).isoformat(),
                "feature_available_time": (start + timedelta(minutes=i)).isoformat(),
                "outcome_available_time": (start + timedelta(minutes=i, seconds=1)).isoformat(),
                "features": features,
                "filled": i % 2 == 0,
                "toxicity_bps": float(i % 3) if i % 2 == 0 else None,
                "source": "SYNTHETIC_http_log",
                "synthetic": True,
            }
            for i in range(40)
        ],
        "proposals": [
            {
                "venue": "A",
                "quote_available_time": cutoff.isoformat(),
                "feature_available_time": cutoff.isoformat(),
                "price": 100.0,
                "capacity": 10,
                "fee_bps": 0.0,
                "crossing_cost_bps": 1.0,
                "features": features,
            }
        ],
        "training_cutoff": cutoff.isoformat(),
        "decision_time": cutoff.isoformat(),
        "quantity": 5,
        "side": "buy",
        "opportunity_cost_bps": 10.0,
    }


def test_api_fits_real_model_and_returns_bounded_offline_plan():
    response = TestClient(app).post("/v1/blueprint/route", json=request_payload())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["plan"]["children"][0]["quantity"] == 5
    assert 0 < result["plan"]["children"][0]["p_fill"] < 1
    assert result["synthetic"] and result["research_only"]
    assert not result["live_pnl_claim"] and not result["market_evidence"]
    assert len(result["model_sha256"]) == 64
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("change", ["future_quote", "bool_quantity", "bool_fill", "unknown"])
def test_api_refuses_clock_type_and_schema_corruptions(change):
    payload = request_payload()
    if change == "future_quote":
        payload["proposals"][0]["quote_available_time"] = "2021-01-01T00:00:00+00:00"
    elif change == "bool_quantity":
        payload["quantity"] = True
    elif change == "bool_fill":
        payload["observations"][0]["filled"] = "true"
    else:
        payload["submit"] = True
    response = TestClient(app).post("/v1/blueprint/route", json=payload)
    assert response.status_code == 422


def test_router_api_inherits_key_protection(monkeypatch):
    monkeypatch.setenv("QUANT_API_KEY", "fixture-only-not-secret")
    client = TestClient(app)
    assert client.get("/v1/blueprint/capabilities").status_code == 401
    assert (
        client.get(
            "/v1/blueprint/capabilities", headers={"X-API-Key": "fixture-only-not-secret"}
        ).json()["route_submission"]
        is False
    )


def test_non_ascii_api_key_is_rejected_without_server_error(monkeypatch):
    monkeypatch.setenv("QUANT_API_KEY", "fixture-only-not-secret")
    client = TestClient(app)
    response = client.get("/v1/blueprint/capabilities", headers={b"x-api-key": b"\xff"})
    assert response.status_code == 401
    assert response.headers["Cache-Control"] == "no-store"
    assert (
        client.get(
            "/v1/blueprint/capabilities", headers={"X-API-Key": "fixture-only-not-secret"}
        ).status_code
        == 200
    )


@pytest.mark.parametrize("declared_length", [None, "1"])
def test_oversized_stream_returns_413_before_router_training(monkeypatch, declared_length):
    monkeypatch.delenv("QUANT_API_KEY", raising=False)

    def forbid_training(*args, **kwargs):
        raise AssertionError("oversized request reached model training")

    monkeypatch.setattr(MLFillRouter, "update_models", forbid_training)
    body = json.dumps(request_payload()).encode()
    body += b" " * (65537 - len(body))
    headers = {"content-type": "application/json"}
    if declared_length is not None:
        headers["content-length"] = declared_length
    response = TestClient(app).post(
        "/v1/blueprint/route", content=iter([body[:32768], body[32768:]]), headers=headers
    )
    assert response.status_code == 413
    assert response.headers["Cache-Control"] == "no-store"
