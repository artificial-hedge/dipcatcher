"""Versioned offline learned-router API, protected by the harness app middleware.

Requests carry observations rather than filesystem paths. Each fit is bounded
and ephemeral: the returned plan is never submitted to a venue and it is not
persisted or promoted as market evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from quant_fund.execution.ml_router import (
    ChildOrderProposal,
    FillFeatures,
    MLFillRouter,
    OrderObservation,
)

router = APIRouter(prefix="/v1/blueprint", tags=["blueprint research"])
_GRAPH_RUNS = Path(__file__).resolve().parents[3] / "data/metadata/blueprint_graph"
_FEDERATED_RUNS = Path(__file__).resolve().parents[3] / "data/metadata/blueprint_federated"


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return (Path(__file__).parent / "static/blueprint.html").read_text(encoding="utf-8")


@router.get("/graph/{run_id}")
def graph_evidence(run_id: str) -> dict[str, Any]:
    from quant_fund.research.blueprint_graph_evidence import read_graph_evidence

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id):
        raise HTTPException(status_code=422, detail="invalid graph run id")
    root = _GRAPH_RUNS.resolve()
    run = (root / run_id).resolve()
    if run.parent != root:
        raise HTTPException(status_code=422, detail="graph run escapes artifact root")
    try:
        return read_graph_evidence(run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="graph run artifact missing") from exc
    except (ValueError, KeyError, TypeError, OSError, BadZipFile, EOFError) as exc:
        raise HTTPException(status_code=422, detail="graph evidence verification failed") from exc


@router.get("/federated/{run_id}")
def federated_evidence(run_id: str) -> dict[str, Any]:
    from quant_fund.research.blueprint_federated_evidence import read_federated_evidence

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", run_id):
        raise HTTPException(status_code=422, detail="invalid federated run id")
    root = _FEDERATED_RUNS.resolve()
    run = (root / run_id).resolve()
    if run.parent != root:
        raise HTTPException(status_code=422, detail="federated run escapes artifact root")
    try:
        return read_federated_evidence(run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="federated run artifact missing") from exc
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        raise HTTPException(
            status_code=422, detail="federated evidence verification failed"
        ) from exc


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FeaturesInput(_Input):
    price_distance_bps: StrictFloat
    spread_bps: StrictFloat
    queue_ahead: StrictFloat
    displayed_depth: StrictFloat
    volatility_bps: StrictFloat
    seconds_to_deadline: StrictFloat


class ObservationInput(_Input):
    order_id: StrictStr = Field(min_length=1, max_length=128)
    venue: StrictStr = Field(min_length=1, max_length=64)
    decision_time: datetime
    feature_available_time: datetime
    outcome_available_time: datetime
    features: FeaturesInput
    filled: StrictBool
    toxicity_bps: StrictFloat | None
    source: StrictStr = Field(min_length=1, max_length=256)
    synthetic: StrictBool


class ProposalInput(_Input):
    venue: StrictStr = Field(min_length=1, max_length=64)
    quote_available_time: datetime
    feature_available_time: datetime
    price: StrictFloat = Field(gt=0)
    capacity: StrictInt = Field(ge=0)
    fee_bps: StrictFloat
    crossing_cost_bps: StrictFloat
    features: FeaturesInput


class RouteInput(_Input):
    observations: list[ObservationInput] = Field(min_length=20, max_length=512)
    proposals: list[ProposalInput] = Field(min_length=1, max_length=32)
    training_cutoff: datetime
    decision_time: datetime
    quantity: StrictInt = Field(ge=1, le=1_000_000)
    side: Literal["buy", "sell"]
    opportunity_cost_bps: StrictFloat = Field(ge=0)
    limit_price: StrictFloat | None = Field(default=None, gt=0)
    seed: StrictInt = Field(default=7, ge=0, le=2**32 - 1)


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "schema_version": "blueprint_api_v1",
        "router": "regularized_logistic_fill_ridge_gaussian_toxicity",
        "route_submission": False,
        "max_training_rows": 512,
        "max_proposals": 32,
        "research_only": True,
        "live_pnl_claim": False,
        "market_evidence": False,
    }


@router.post("/route")
def route(payload: RouteInput) -> dict[str, Any]:
    try:
        observations = [
            OrderObservation(
                **{key: value for key, value in row.model_dump().items() if key != "features"},
                features=FillFeatures(**row.features.model_dump()),
            )
            for row in payload.observations
        ]
        proposals = [
            ChildOrderProposal(
                **{key: value for key, value in row.model_dump().items() if key != "features"},
                features=FillFeatures(**row.features.model_dump()),
            )
            for row in payload.proposals
        ]
        model = MLFillRouter(seed=payload.seed)
        training = model.update_models(observations, asof=payload.training_cutoff)
        plan = model.route_order(
            proposals,
            quantity=payload.quantity,
            side=payload.side,
            decision_time=payload.decision_time,
            opportunity_cost_bps=payload.opportunity_cost_bps,
            limit_price=payload.limit_price,
        )
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result: dict[str, Any] = {
        "schema_version": "blueprint_route_api_result_v1",
        "input_sha256": hashlib.sha256(
            json.dumps(
                payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "model_sha256": model.model_sha256,
        "training": training,
        "plan": asdict(plan),
        "synthetic": model.synthetic_training,
        "research_only": True,
        "live_pnl_claim": False,
        "market_evidence": False,
        "limitations": [
            "frozen_linear_per_unit_objective",
            "no_size_dependent_fill_or_market_impact",
            "no_execution_counterfactual_or_venue_submission",
            "ephemeral_fit_not_persisted_checkpoint",
        ],
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return result
