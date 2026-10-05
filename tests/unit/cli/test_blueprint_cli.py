"""End-to-end offline routing with synthetic data and immutable labeled output."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.execution.ml_router import ChildOrderProposal, FillFeatures, OrderObservation
from quant_fund.models.trade_anomaly import PatternAnnotation, TradePattern


def test_router_cli_fits_routes_and_retains_synthetic_identity(tmp_path):
    start = datetime(2020, 1, 1, tzinfo=UTC)
    rows = []
    features = FillFeatures(-1, 2, 10, 100, 10, 30)
    for i in range(40):
        clock = start + timedelta(minutes=i)
        filled = i % 2 == 0
        rows.append(
            asdict(
                OrderObservation(
                    str(i),
                    "A",
                    clock,
                    clock,
                    clock + timedelta(seconds=1),
                    features,
                    filled,
                    float(i % 3) if filled else None,
                    "SYNTHETIC_cli",
                    True,
                )
            )
        )
    logs = tmp_path / "orders.json"
    logs.write_text(json.dumps({"observations": rows}, default=lambda value: value.isoformat()))
    cutoff = start + timedelta(minutes=40)
    proposal = ChildOrderProposal("A", cutoff, cutoff, 100, 10, 0, 1, features)
    candidates = tmp_path / "proposals.json"
    candidates.write_text(
        json.dumps({"proposals": [asdict(proposal)]}, default=lambda value: value.isoformat())
    )
    out = tmp_path / "route.json"
    command = [
        "blueprint",
        "route",
        "--logs",
        str(logs),
        "--proposals",
        str(candidates),
        "--train-asof",
        cutoff.isoformat(),
        "--decision-time",
        cutoff.isoformat(),
        "--quantity",
        "5",
        "--side",
        "buy",
        "--opportunity-cost-bps",
        "10",
        "--out",
        str(out),
    ]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.exception
    payload = json.loads(out.read_text())
    assert payload["synthetic"] and payload["evaluation"] is None
    assert payload["plan"]["children"][0]["quantity"] == 5
    assert not payload["live_pnl_claim"] and not payload["market_evidence"]
    assert CliRunner().invoke(app, command).exit_code != 0


def test_anomaly_cli_saved_model_score_replay_and_forged_score_rejection(tmp_path):
    start = datetime(2020, 1, 1, tzinfo=UTC)

    def rows(first):
        patterns, annotations = [], []
        for i in range(first, first + 40):
            clock = start + timedelta(minutes=i)
            features = (float(2 + i % 2), 0.01, float(i % 3), float(i % 4), 5.0, 0.1)
            patterns.append(
                TradePattern(str(i), clock, clock, clock, features, "SYNTHETIC_cli", True).record()
            )
            annotations.append(
                PatternAnnotation(
                    str(i), bool(i % 2), clock, "SYNTHETIC_annotations", "predeclared fixture", True
                ).record()
            )
        return patterns, annotations

    train, _ = rows(0)
    calibration, calibration_annotations = rows(41)
    evaluation, evaluation_annotations = rows(82)
    inputs = tmp_path / "input.json"
    inputs.write_text(
        json.dumps(
            {
                "training": train,
                "fit_asof": (start + timedelta(minutes=40)).isoformat(),
                "calibration": calibration,
                "calibration_annotations": calibration_annotations,
                "calibration_asof": (start + timedelta(minutes=81)).isoformat(),
                "evaluation": evaluation,
                "evaluation_annotations": evaluation_annotations,
                "evaluation_asof": (start + timedelta(minutes=123)).isoformat(),
            }
        )
    )
    run = tmp_path / "run"
    command = [
        "blueprint",
        "anomaly",
        "--inputs",
        str(inputs),
        "--out",
        str(run),
        "--trees",
        "4",
        "--samples",
        "32",
    ]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.exception
    verify = ["blueprint", "anomaly-verify", "--run", str(run)]
    verified = CliRunner().invoke(app, verify)
    assert verified.exit_code == 0, verified.exception
    replayed = json.loads(verified.stdout)
    assert replayed["verification"]["saved_forecasts_and_scores_recomputed"]
    assert replayed["verification"]["training_and_calibration_input_hashes"]
    assert not replayed["verification"]["training_repeated"]
    assert replayed["evaluation"]["synthetic"]
    assert CliRunner().invoke(app, command).exit_code != 0
    receipt_path = run / "receipt.json"
    original_receipt = receipt_path.read_bytes()
    saved_inputs = run / "inputs.json"
    original_inputs = saved_inputs.read_bytes()
    changed_inputs = json.loads(original_inputs)
    changed_inputs["training"][0]["features"][0] += 1
    saved_inputs.write_text(json.dumps(changed_inputs))
    changed_receipt = json.loads(original_receipt)
    changed_receipt.pop("receipt_sha256")
    changed_receipt["artifacts"]["inputs.json"] = hashlib.sha256(
        saved_inputs.read_bytes()
    ).hexdigest()
    changed_receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(changed_receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(changed_receipt))
    refused_binding = CliRunner().invoke(app, verify)
    assert refused_binding.exit_code != 0
    assert "input binding mismatch" in str(refused_binding.exception)
    saved_inputs.write_bytes(original_inputs)
    receipt_path.write_bytes(original_receipt)
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("receipt_sha256")
    receipt["evaluation"]["brier"] = -1
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    refused = CliRunner().invoke(app, verify)
    assert refused.exit_code != 0
    assert "did not reproduce" in str(refused.exception)


def test_qubo_cli_runs_and_replays_classical_solver_receipt(tmp_path):
    clock = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    evidence = {
        "assets": ["A", "B", "C"],
        "horizon_seconds": 86400,
        "asof": clock,
        "available_time": clock,
        "source_id": "SYNTHETIC_cli",
        "source_sha256": hashlib.sha256(b"SYNTHETIC_numeric_problem").hexdigest(),
        "synthetic": True,
    }
    data = {
        "forecast": {**evidence, "expected_returns": [0.01, 0.03, -0.01]},
        "covariance": {
            **evidence,
            "values": [[0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]],
        },
        "decision_time": clock,
        "cardinality": 1,
        "risk_aversion": 2.0,
        "budget": 0.8,
    }
    inputs = tmp_path / "problem.json"
    inputs.write_text(json.dumps(data))
    receipt = tmp_path / "qubo.json"
    command = [
        "blueprint",
        "qubo",
        "--inputs",
        str(inputs),
        "--out",
        str(receipt),
        "--seeds",
        "3,19",
        "--sweeps",
        "10",
    ]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.exception
    summary = json.loads(result.stdout)
    assert summary["synthetic"] and not summary["quantum_advantage"]
    assert summary["diagnostics"]["exact_optimality_proven"]
    assert not summary["diagnostics"]["objective_is_proper_score"]
    assert CliRunner().invoke(app, command).exit_code != 0
    verified = CliRunner().invoke(app, ["blueprint", "qubo-verify", "--receipt", str(receipt)])
    assert verified.exit_code == 0, verified.exception
    assert json.loads(verified.stdout)["verification"]["solver_behavior_replayed"]
    assert json.loads(verified.stdout)["deterministic_sha256"] == summary["deterministic_sha256"]


def _federated_input():
    from quant_fund.models.federated_market import BinaryExample, BinaryTask

    start = datetime(2020, 1, 1, tzinfo=UTC)
    task = BinaryTask(
        "cli_synthetic_event", ("x",), "event", "label predicts supplied synthetic event"
    )

    def rows(prefix, count, first, split):
        values = []
        for i in range(count):
            clock = start + timedelta(hours=first + i)
            row = BinaryExample(
                f"{prefix}:{i}",
                "ASSET",
                task.task_sha256,
                task.feature_names,
                (float((i % 4) - 1.5),),
                int(i % 4 >= 2),
                clock,
                clock,
                clock + timedelta(minutes=1),
                clock + timedelta(minutes=5),
                clock + timedelta(minutes=8),
                "SYNTHETIC_cli",
                "a" * 64,
                split,
                True,
            )
            record = asdict(row)
            record.pop("schema_version")
            values.append(record)
        return values

    task_record = asdict(task)
    for name in ("schema_version", "task_sha256"):
        task_record.pop(name)
    body = {
        "task": task_record,
        "clients": [
            {"client_id": "alpha", "examples": rows("a", 16, 0, "train")},
            {"client_id": "beta", "examples": rows("b", 24, 0, "train")},
        ],
        "holdout": rows("holdout", 32, 120, "holdout"),
        "initial_asof": start - timedelta(hours=1),
        "training_asof": start + timedelta(hours=30),
        "aggregation_asof": start + timedelta(hours=31),
        "evaluation_asof": start + timedelta(hours=200),
        "client_config": {"epochs": 2, "batch_size": 8, "seed": 7},
        "centralized_config": {"epochs": 4, "batch_size": 8, "seed": 7},
        "rounds": 3,
    }
    return json.loads(json.dumps(body, default=lambda value: value.isoformat()))


def test_federated_cli_trains_replays_and_retains_privacy_and_budget_limits(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quant_fund.api import blueprint
    from quant_fund.api.app import app as api_app

    data = _federated_input()
    inputs = tmp_path / "federated.json"
    inputs.write_text(json.dumps(data))
    run = tmp_path / "federated-run"
    command = ["blueprint", "federated", "--inputs", str(inputs), "--out", str(run)]
    trained = CliRunner().invoke(app, command)
    assert trained.exit_code == 0, trained.exception
    result = json.loads(trained.stdout)
    assert result["synthetic"] and not result["privacy_guarantee"]
    assert not result["process_isolation"] and not result["market_evidence"]
    assert not result["comparison"]["compute_budgets_equal"]
    assert set(result["comparison"]["scores"]) == {
        "global",
        "centralized",
        "local:alpha",
        "local:beta",
    }
    assert result["comparison"]["budgets"]["global_cumulative_optimizer_steps"] == 30
    assert result["comparison"]["budgets"]["centralized_optimizer_steps"] == 20
    assert (run / "inputs.json").read_bytes() == inputs.read_bytes()
    assert CliRunner().invoke(app, command).exit_code != 0
    verified = CliRunner().invoke(app, ["blueprint", "federated-verify", "--run", str(run)])
    assert verified.exit_code == 0, verified.exception
    replay = json.loads(verified.stdout)
    assert replay["receipt_sha256"] == result["receipt_sha256"]
    assert replay["aggregation_replayed"] and replay["heldout_scores_recomputed"]
    assert not replay["training_repeated"] and not replay["client_updates_authenticated"]
    assert replay["scores"] == result["comparison"]["scores"]
    monkeypatch.setattr(blueprint, "_FEDERATED_RUNS", tmp_path)
    client = TestClient(api_app)
    response = client.get("/v1/blueprint/federated/federated-run")
    assert response.status_code == 200, response.text
    assert response.json()["scores"] == replay["scores"]
    assert not response.json()["privacy_guarantee"]
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/v1/blueprint/federated/missing").status_code == 404
    outside = tmp_path / "outside"
    outside.mkdir()
    roots = tmp_path / "roots"
    roots.mkdir()
    (roots / "escaped").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(blueprint, "_FEDERATED_RUNS", roots)
    assert client.get("/v1/blueprint/federated/escaped").status_code == 422


def test_federated_cli_rejects_rehashed_scores_and_changed_training_labels(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from quant_fund.api import blueprint
    from quant_fund.api.app import app as api_app
    from quant_fund.research.blueprint_federated_evidence import _federated_hash

    inputs = tmp_path / "federated.json"
    inputs.write_text(json.dumps(_federated_input()))
    run = tmp_path / "run"
    trained = CliRunner().invoke(
        app, ["blueprint", "federated", "--inputs", str(inputs), "--out", str(run)]
    )
    assert trained.exit_code == 0, trained.exception
    receipt_file = run / "receipt.json"
    original = json.loads(receipt_file.read_text())
    verify = ["blueprint", "federated-verify", "--run", str(run)]
    forged = json.loads(json.dumps(original))
    comparison = forged["body"]["comparison"]
    comparison["scores"]["global"]["brier"] = 0.0
    comparison["scores"]["global"]["receipt_sha256"] = _federated_hash(
        {k: v for k, v in comparison["scores"]["global"].items() if k != "receipt_sha256"}
    )
    comparison["receipt_sha256"] = _federated_hash(
        {k: v for k, v in comparison.items() if k != "receipt_sha256"}
    )
    forged["receipt_sha256"] = _federated_hash(forged["body"])
    receipt_file.write_text(json.dumps(forged))
    rejected = CliRunner().invoke(app, verify)
    assert rejected.exit_code != 0
    assert "heldout score mismatch" in str(rejected.exception)
    monkeypatch.setattr(blueprint, "_FEDERATED_RUNS", tmp_path)
    assert TestClient(api_app).get("/v1/blueprint/federated/run").status_code == 422
    receipt_file.write_text(json.dumps(original))
    changed = json.loads((run / "inputs.json").read_text())
    changed["clients"][0]["examples"][0]["label"] = (
        1 - changed["clients"][0]["examples"][0]["label"]
    )
    (run / "inputs.json").write_text(json.dumps(changed))
    digest = hashlib.sha256((run / "inputs.json").read_bytes()).hexdigest()
    original["body"]["artifacts"]["inputs.json"] = digest
    original["body"]["input_sha256"] = digest
    original["receipt_sha256"] = _federated_hash(original["body"])
    receipt_file.write_text(json.dumps(original))
    rejected = CliRunner().invoke(app, verify)
    assert rejected.exit_code != 0
    assert "update/input source binding mismatch" in str(rejected.exception)
