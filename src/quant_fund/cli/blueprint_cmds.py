"""Offline PDF-blueprint capability commands with explicit evidence labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import typer

blueprint_app = typer.Typer(
    help="Executive blueprint research pilots; no venue connections or orders."
)


@blueprint_app.command("graph-evidence")
def graph_evidence(
    run: Path = typer.Option(..., help="Saved graph experiment directory."),
) -> None:
    """Check saved graph artifacts and recompute proper scores without retraining."""
    from quant_fund.research.blueprint_graph_evidence import read_graph_evidence

    typer.echo(json.dumps(read_graph_evidence(run), sort_keys=True, allow_nan=False))


@blueprint_app.command("execution-verify")
def execution_verify(
    receipt: Path = typer.Option(..., help="Immutable synthetic parent-order receipt."),
) -> None:
    """Replay the saved policy actions and independently check execution ledgers."""
    from quant_fund.execution.parent_order_rl import verify_execution_receipt

    verified = verify_execution_receipt(receipt)
    payload = verified["payload"]
    typer.echo(
        json.dumps(
            {
                "receipt_sha256": verified["receipt_sha256"],
                "model_sha256": payload["model_sha256"],
                "synthetic": payload["synthetic"],
                "research_only": True,
                "market_evidence": False,
                "live_pnl_claim": False,
                "promote": False,
                "sota_claim": False,
                "verification": {
                    "saved_policy_actions_replayed": True,
                    "independent_execution_ledgers": True,
                    "training_repeated": False,
                    "historical_market_replay": False,
                },
                "diagnostics": payload["diagnostics"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _qubo_summary(report):
    payload = report.payload()
    return {
        "receipt_sha256": report.receipt_sha256,
        "deterministic_sha256": report.deterministic_sha256,
        "problem_sha256": report.model.problem.problem_sha256,
        "diagnostics": payload["diagnostics"],
        "trial_statuses": [trial.status for trial in report.trials],
        "synthetic": report.model.problem.synthetic,
        "research_only": True,
        "market_evidence": False,
        "quantum_advantage": False,
    }


@blueprint_app.command("qubo")
def qubo(
    inputs: Path = typer.Option(..., help="Timed forecast/covariance selection JSON."),
    out: Path = typer.Option(..., help="New immutable classical solver receipt."),
    seeds: str = typer.Option("11,23,47,83", help="Comma-separated unique integer seeds."),
    sweeps: int = typer.Option(100, min=1, max=1000),
    mode: Literal["qubo_bit_flip", "cardinality_swap"] = typer.Option("qubo_bit_flip"),
    initial_temperature: float = typer.Option(10.0, min=1e-12, max=1e15),
    final_temperature: float = typer.Option(0.001, min=1e-12, max=1e15),
) -> None:
    """Anneal a classical QUBO and retain failed trials and matched constraints."""
    from quant_fund.portfolio.qubo_selection import (
        AnnealConfig,
        CovarianceEvidence,
        ForecastEvidence,
        SelectionProblem,
        save_receipt,
        solve_selection,
    )

    if out.exists():
        raise FileExistsError(f"QUBO receipt already exists: {out}")
    data = _json(inputs)
    if set(data) != {
        "forecast",
        "covariance",
        "decision_time",
        "cardinality",
        "risk_aversion",
        "budget",
    }:
        raise ValueError("invalid QUBO selection input schema")
    forecast, covariance = dict(data["forecast"]), dict(data["covariance"])
    for evidence in (forecast, covariance):
        evidence["assets"] = tuple(evidence["assets"])
        seconds = evidence.pop("horizon_seconds")
        if type(seconds) not in (int, float):
            raise ValueError("horizon_seconds must be numeric")
        evidence["horizon"] = timedelta(seconds=seconds)
        for name in ("asof", "available_time"):
            evidence[name] = datetime.fromisoformat(evidence[name])
    forecast["expected_returns"] = tuple(forecast["expected_returns"])
    covariance["values"] = tuple(tuple(row) for row in covariance["values"])
    problem = SelectionProblem(
        ForecastEvidence(**forecast),
        CovarianceEvidence(**covariance),
        datetime.fromisoformat(data["decision_time"]),
        data["cardinality"],
        data["risk_aversion"],
        data["budget"],
    )
    report = solve_selection(
        problem,
        AnnealConfig(
            seeds=tuple(int(value) for value in seeds.split(",")),
            sweeps=sweeps,
            mode=mode,
            initial_temperature=initial_temperature,
            final_temperature=final_temperature,
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_receipt(report, out)
    typer.echo(json.dumps(_qubo_summary(report), sort_keys=True, allow_nan=False))


@blueprint_app.command("qubo-verify")
def qubo_verify(
    receipt: Path = typer.Option(..., help="Saved classical solver receipt."),
) -> None:
    """Replay annealing, greedy and exhaustive-small results; no device claims."""
    from quant_fund.portfolio.qubo_selection import restore_receipt

    report = restore_receipt(receipt)
    typer.echo(
        json.dumps(
            {
                **_qubo_summary(report),
                "verification": {
                    "solver_behavior_replayed": True,
                    "elapsed_time_independently_verified": False,
                    "source_authenticity_independently_verified": False,
                },
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _bounded_bytes(path: Path, budget: int) -> bytes:
    if path.stat().st_size > budget:
        raise ValueError("blueprint file exceeds the input resource budget")
    with path.open("rb") as stream:
        raw = stream.read(budget + 1)
    if len(raw) > budget:
        raise ValueError("blueprint file exceeds the input resource budget")
    return raw


def _json(path: Path) -> dict:
    payload = json.loads(_bounded_bytes(path, 2_000_000))
    if not isinstance(payload, dict):
        raise ValueError("blueprint input must be a JSON object")
    return payload


@blueprint_app.command("federated")
def federated(
    inputs: Path = typer.Option(
        ..., help="Timed client/holdout/config JSON; all clients share this process."
    ),
    out: Path = typer.Option(..., help="New immutable offline comparison directory."),
) -> None:
    """Train real local SGD/FedAvg and retain unequal-budget controls; no privacy claim."""
    import math

    from quant_fund.models import federated_market
    from quant_fund.models.federated_market import (
        BinaryExample,
        BinaryTask,
        ClientConfig,
        FederatedClient,
        FederatedServer,
        compare_models,
        train_centralized,
    )
    from quant_fund.research.blueprint_federated_evidence import (
        _federated_examples,
        _federated_hash,
        _federated_json,
    )

    if out.exists():
        raise FileExistsError(f"federated run already exists: {out}")
    raw = _bounded_bytes(inputs, 2_000_000)
    data = json.loads(raw)
    keys = {
        "task",
        "clients",
        "holdout",
        "initial_asof",
        "training_asof",
        "aggregation_asof",
        "evaluation_asof",
        "client_config",
        "centralized_config",
        "rounds",
    }
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError("invalid federated input schema")
    task_data = dict(data["task"])
    task_data["feature_names"] = tuple(task_data["feature_names"])
    task = BinaryTask(**task_data)
    if len(task.feature_names) > 32:
        raise ValueError("CLI supports at most 32 features")
    if type(data["rounds"]) is not int or not 1 <= data["rounds"] <= 20:
        raise ValueError("one to twenty rounds required")
    if not isinstance(data["clients"], list) or not 1 <= len(data["clients"]) <= 8:
        raise ValueError("one to eight explicit clients required")
    client_config = ClientConfig(**data["client_config"])
    central_config = ClientConfig(**data["centralized_config"])
    if max(client_config.epochs, central_config.epochs) > 20:
        raise ValueError("CLI optimizer epochs exceed twenty")
    clients = []
    pooled: list[BinaryExample] = []
    client_sizes = []
    for value in data["clients"]:
        if set(value) != {"client_id", "examples"}:
            raise ValueError("invalid client input schema")
        rows = _federated_examples(value["examples"])
        clients.append(FederatedClient(value["client_id"], task, rows))
        pooled.extend(rows)
        client_sizes.append(len(rows))
    if len(pooled) > 1024 or len({row.record_id for row in pooled}) != len(pooled):
        raise ValueError("client prefixes overlap or exceed the 1024-row budget")
    holdout = _federated_examples(data["holdout"])
    steps = data["rounds"] * client_config.epochs * sum(
        math.ceil(size / client_config.batch_size) for size in client_sizes
    ) + central_config.epochs * math.ceil(len(pooled) / central_config.batch_size)
    if steps > 50_000:
        raise ValueError("CLI optimization work exceeds 50000 steps")
    clocks = {
        name: datetime.fromisoformat(data[name])
        for name in (
            "initial_asof",
            "training_asof",
            "aggregation_asof",
            "evaluation_asof",
        )
    }
    server = FederatedServer(task, initial_as_of=clocks["initial_asof"])
    initial = server.freeze()
    for index in range(data["rounds"]):
        aggregation = clocks["aggregation_asof"] + timedelta(seconds=index)
        spec = server.begin_round(
            tuple(client.client_id for client in clients),
            training_as_of=clocks["training_asof"],
            aggregation_as_of=aggregation,
            config=client_config,
        )
        local = tuple(
            client.train(spec, server.freeze(), completed_at=aggregation) for client in clients
        )
        server.aggregate(tuple(result.update for result in local), as_of=aggregation)
    central = train_centralized(
        task,
        pooled,
        initial,
        config=central_config,
        training_as_of=clocks["training_asof"],
        completed_at=clocks["aggregation_asof"],
    )
    comparison = compare_models(
        server.freeze(), local, central, holdout, as_of=clocks["evaluation_asof"]
    )
    models = {
        "global": asdict(server.freeze()),
        "centralized": asdict(central.model),
        **{f"local:{result.update.client_id}": asdict(result.model) for result in local},
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs.json").write_bytes(raw)
    (out / "model_source.py").write_bytes(Path(federated_market.__file__).read_bytes())
    server.save_json(out / "server.json")
    artifacts = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in (
            "inputs.json",
            "model_source.py",
            "server.json",
        )
    }
    body = {
        "schema_version": "blueprint_federated_run_v1",
        "artifacts": artifacts,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "model_sha256": server.freeze().model_sha256,
        "models": models,
        "comparison": comparison,
        "training_ids": sorted(
            result.record_id
            for result in pooled
            if result.split == "train" and result.target_available_time <= clocks["training_asof"]
        ),
        "synthetic": any(score["synthetic"] for score in comparison["scores"].values()),
        "research_only": True,
        "market_evidence": False,
        "privacy_guarantee": False,
        "process_isolation": False,
        "source_authenticity_verified": False,
    }
    body = json.loads(_federated_json(body))
    receipt = {"body": body, "receipt_sha256": _federated_hash(body)}
    with (out / "receipt.json").open("x") as stream:
        stream.write(_federated_json(receipt) + "\n")
    typer.echo(
        _federated_json(
            {
                "receipt_sha256": receipt["receipt_sha256"],
                "comparison": comparison,
                "synthetic": body["synthetic"],
                "research_only": True,
                "privacy_guarantee": False,
                "process_isolation": False,
                "market_evidence": False,
            }
        )
    )


@blueprint_app.command("federated-verify")
def federated_verify(
    run: Path = typer.Option(..., help="Saved offline federated run directory."),
) -> None:
    """Replay aggregation and every saved model's heldout scores, without training."""
    from quant_fund.research.blueprint_federated_evidence import (
        _federated_json,
        read_federated_evidence,
    )

    typer.echo(_federated_json(read_federated_evidence(run)))


def _anomaly_patterns(payload: list):
    from quant_fund.models.trade_anomaly import TradePattern

    return [
        TradePattern(
            **{
                **row,
                **{
                    key: datetime.fromisoformat(row[key])
                    for key in ("event_time", "available_time", "decision_time")
                },
            }
        )
        for row in payload
    ]


def _anomaly_annotations(payload: list):
    from quant_fund.models.trade_anomaly import PatternAnnotation

    return [
        PatternAnnotation(
            **{**row, "available_time": datetime.fromisoformat(row["available_time"])}
        )
        for row in payload
    ]


@blueprint_app.command("anomaly")
def anomaly(
    inputs: Path = typer.Option(..., help="Timed train/calibration/evaluation JSON."),
    out: Path = typer.Option(..., help="New immutable research run directory."),
    trees: int = typer.Option(64, min=1, max=256),
    samples: int = typer.Option(256, min=2, max=256),
    seed: int = typer.Option(7, min=0, max=2**32 - 1),
) -> None:
    """Fit actual isolation trees and score later supplied suspicious annotations."""
    from quant_fund.models.trade_anomaly import TradeAnomalyModel

    if out.exists():
        raise FileExistsError(f"anomaly output already exists: {out}")
    raw = _bounded_bytes(inputs, 2_000_000)
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {
        "training",
        "fit_asof",
        "calibration",
        "calibration_annotations",
        "calibration_asof",
        "evaluation",
        "evaluation_annotations",
        "evaluation_asof",
    }:
        raise ValueError("invalid anomaly experiment schema")
    model = TradeAnomalyModel(trees=trees, samples=samples, seed=seed).fit(
        _anomaly_patterns(data["training"]), asof=datetime.fromisoformat(data["fit_asof"])
    )
    model.calibrate(
        _anomaly_patterns(data["calibration"]),
        _anomaly_annotations(data["calibration_annotations"]),
        asof=datetime.fromisoformat(data["calibration_asof"]),
    )
    result = model.evaluate(
        _anomaly_patterns(data["evaluation"]),
        _anomaly_annotations(data["evaluation_annotations"]),
        asof=datetime.fromisoformat(data["evaluation_asof"]),
    )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "inputs.json").open("xb") as stream:
        stream.write(raw)
    model.save(out / "model.json")
    source = Path(__file__).resolve().parents[1] / "models" / "trade_anomaly.py"
    with (out / "model_source.py").open("xb") as stream:
        stream.write(source.read_bytes())
    receipt = {
        "schema_version": "blueprint_trade_anomaly_v1",
        "model_sha256": model.model_sha256,
        "artifacts": {
            "inputs.json": hashlib.sha256(raw).hexdigest(),
            "model.json": hashlib.sha256((out / "model.json").read_bytes()).hexdigest(),
            "model_source.py": hashlib.sha256((out / "model_source.py").read_bytes()).hexdigest(),
        },
        "evaluation": result,
        "research_only": True,
        "market_evidence": False,
        "live_pnl_claim": False,
        "promote": False,
        "sota_claim": False,
        "limitations": [
            "annotation_independence_and_data_rights_unverified",
            "insider_misconduct_and_informed_intent_unknown",
            "caller_computed_features_not_raw_tape_ingestion",
            "offline_research_not_operational_surveillance",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with (out / "receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    typer.echo(json.dumps(receipt, sort_keys=True, allow_nan=False))


@blueprint_app.command("anomaly-verify")
def anomaly_verify(
    run: Path = typer.Option(..., help="Saved anomaly run directory."),
) -> None:
    """Reload safe trees and reproduce held-out probabilities and proper scores."""
    from quant_fund.models.trade_anomaly import TradeAnomalyModel

    root = run.resolve()
    for name, budget in (
        ("receipt.json", 2_000_000),
        ("inputs.json", 2_000_000),
        ("model.json", 16_000_000),
    ):
        path = (root / name).resolve()
        if path.parent != root or path.stat().st_size > budget:
            raise ValueError("anomaly artifact path or resource budget invalid")
    receipt = _json(root / "receipt.json")
    identity = receipt.pop("receipt_sha256", None)
    if (
        identity
        != hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    ):
        raise ValueError("anomaly receipt hash mismatch")
    if receipt.get("schema_version") != "blueprint_trade_anomaly_v1" or set(
        receipt["artifacts"]
    ) not in ({"inputs.json", "model.json"}, {"inputs.json", "model.json", "model_source.py"}):
        raise ValueError("invalid anomaly receipt schema")
    for key, expected in {
        "research_only": True,
        "market_evidence": False,
        "live_pnl_claim": False,
        "promote": False,
        "sota_claim": False,
    }.items():
        if receipt.get(key) is not expected:
            raise ValueError("anomaly receipt honesty flags invalid")
    for name, expected in receipt["artifacts"].items():
        path = (root / name).resolve()
        if path.parent != root:
            raise ValueError("anomaly artifact path invalid")
        budget = 16_000_000 if name == "model.json" else 2_000_000
        if hashlib.sha256(_bounded_bytes(path, budget)).hexdigest() != expected:
            raise ValueError("anomaly artifact hash mismatch")
    model = TradeAnomalyModel.load(root / "model.json")
    if model.model_sha256 != receipt["model_sha256"]:
        raise ValueError("anomaly model identity mismatch")
    data = _json(root / "inputs.json")
    payload = model.snapshot()["payload"]
    source_path = (root / "model_source.py").resolve()
    source_verified = False
    if source_path.exists():
        if source_path.parent != root or source_path.stat().st_size > 2_000_000:
            raise ValueError("anomaly source snapshot path or resource budget invalid")
        source_verified = (
            hashlib.sha256(_bounded_bytes(source_path, 2_000_000)).hexdigest()
            == payload["implementation_sha256"]
        )
        if not source_verified:
            raise ValueError("anomaly source snapshot hash mismatch")
    training = _anomaly_patterns(data["training"])
    calibration = _anomaly_patterns(data["calibration"])
    calibration_annotations = _anomaly_annotations(data["calibration_annotations"])

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    if (
        digest([row.record() for row in training]) != payload["training_sha256"]
        or [row.pattern_id for row in training] != payload["training_ids"]
        or datetime.fromisoformat(data["fit_asof"]) != datetime.fromisoformat(payload["fit_asof"])
        or digest(
            {
                "patterns": [row.record() for row in calibration],
                "annotations": [a.record() for a in calibration_annotations],
            }
        )
        != payload["calibration"]["sha256"]
        or datetime.fromisoformat(data["calibration_asof"])
        != datetime.fromisoformat(payload["calibration"]["asof"])
    ):
        raise ValueError("anomaly training or calibration input binding mismatch")
    reproduced = model.evaluate(
        _anomaly_patterns(data["evaluation"]),
        _anomaly_annotations(data["evaluation_annotations"]),
        asof=datetime.fromisoformat(data["evaluation_asof"]),
    )
    if reproduced != receipt["evaluation"]:
        raise ValueError("anomaly scores did not reproduce")
    typer.echo(
        json.dumps(
            {
                "receipt_sha256": identity,
                "evaluation": reproduced,
                "verification": {
                    "saved_forecasts_and_scores_recomputed": True,
                    "training_and_calibration_input_hashes": True,
                    "measured_source_snapshot_hash": source_verified,
                    "implementation_matches_current": payload["implementation_sha256"]
                    == hashlib.sha256(
                        (
                            Path(__file__).resolve().parents[1] / "models" / "trade_anomaly.py"
                        ).read_bytes()
                    ).hexdigest(),
                    "training_repeated": False,
                    "source_availability_independently_verified": False,
                },
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _logs(path: Path):
    from quant_fund.execution.ml_router import FillFeatures, OrderObservation

    payload = _json(path)
    return [
        OrderObservation(
            order_id=row["order_id"],
            venue=row["venue"],
            decision_time=datetime.fromisoformat(row["decision_time"]),
            feature_available_time=datetime.fromisoformat(row["feature_available_time"]),
            outcome_available_time=datetime.fromisoformat(row["outcome_available_time"]),
            features=FillFeatures(**row["features"]),
            filled=row["filled"],
            toxicity_bps=row["toxicity_bps"],
            source=row["source"],
            synthetic=row["synthetic"],
        )
        for row in payload["observations"]
    ]


@blueprint_app.command("route")
def route(
    logs: Path = typer.Option(..., help="Finalized past OrderObservation JSON."),
    proposals: Path = typer.Option(..., help="ChildOrderProposal JSON candidates."),
    train_asof: str = typer.Option(..., help="Timezone-aware frozen training cutoff."),
    decision_time: str = typer.Option(..., help="Timezone-aware routing decision."),
    quantity: int = typer.Option(..., min=1, help="Parent quantity in integer units."),
    side: str = typer.Option(..., help="buy | sell"),
    opportunity_cost_bps: float = typer.Option(..., min=0.0),
    out: Path = typer.Option(..., help="New immutable research route receipt."),
    limit_price: float | None = typer.Option(None, min=0.0),
    evaluation_logs: Path | None = typer.Option(
        None, help="Optional later finalized holdout orders."
    ),
    evaluation_asof: str | None = typer.Option(None, help="Required with evaluation_logs."),
    seed: int = typer.Option(7),
) -> None:
    """Fit real fill/toxicity models and allocate an offline capacity-constrained plan."""
    from quant_fund.execution.ml_router import ChildOrderProposal, FillFeatures, MLFillRouter

    if out.exists():
        raise FileExistsError(f"route receipt already exists: {out}")
    if (evaluation_logs is None) != (evaluation_asof is None):
        raise ValueError("evaluation_logs and evaluation_asof must be supplied together")
    model = MLFillRouter(seed=seed)
    training = model.update_models(_logs(logs), asof=datetime.fromisoformat(train_asof))
    candidates = [
        ChildOrderProposal(
            venue=row["venue"],
            quote_available_time=datetime.fromisoformat(row["quote_available_time"]),
            feature_available_time=datetime.fromisoformat(row["feature_available_time"]),
            price=row["price"],
            capacity=row["capacity"],
            fee_bps=row["fee_bps"],
            crossing_cost_bps=row["crossing_cost_bps"],
            features=FillFeatures(**row["features"]),
        )
        for row in _json(proposals)["proposals"]
    ]
    plan = model.route_order(
        candidates,
        quantity=quantity,
        side=side,
        decision_time=datetime.fromisoformat(decision_time),
        opportunity_cost_bps=opportunity_cost_bps,
        limit_price=limit_price,
    )
    evaluation = None
    if evaluation_logs is not None and evaluation_asof is not None:
        evaluation = model.evaluate(
            _logs(evaluation_logs), asof=datetime.fromisoformat(evaluation_asof)
        )
    paths = [logs, proposals] + ([evaluation_logs] if evaluation_logs is not None else [])
    receipt = {
        "schema_version": "blueprint_ml_router_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "training": training,
        "plan": asdict(plan),
        "evaluation": evaluation,
        "config": {
            "seed": seed,
            "quantity": quantity,
            "side": side,
            "opportunity_cost_bps": opportunity_cost_bps,
            "limit_price": limit_price,
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths
        ],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_implementation_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "execution" / "ml_router.py").read_bytes()
        ).hexdigest(),
        "synthetic": model.synthetic_training,
        "research_only": True,
        "live_pnl_claim": False,
        "market_evidence": False,
        "claim": "research_baseline",
        "limitations": [
            "frozen_per_unit_linear_objective",
            "no_size_dependent_fill_or_market_impact",
            "no_venue_submission",
            "no_counterfactual_execution_comparison",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    typer.echo(
        json.dumps(
            {
                "out": str(out),
                "receipt_sha256": receipt["receipt_sha256"],
                "model_sha256": model.model_sha256,
                "synthetic": model.synthetic_training,
                "evaluation_measured": evaluation is not None,
                "research_only": True,
                "live_pnl_claim": False,
            }
        )
    )
