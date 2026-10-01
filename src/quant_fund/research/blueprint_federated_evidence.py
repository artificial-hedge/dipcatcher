"""Bounded saved FedAvg evidence reader; aggregation/score replay, not privacy proof.

Client gradients are not reexecuted. Source commitments and supplied data
rights/labels are unverified. All participants were in one Python process.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from quant_fund.models.federated_market import BinaryExample


def _bounded_bytes(path: Path, budget: int) -> bytes:
    if path.stat().st_size > budget:
        raise ValueError("federated artifact exceeds resource budget")
    with path.open("rb") as stream:
        raw = stream.read(budget + 1)
    if len(raw) > budget:
        raise ValueError("federated artifact exceeds resource budget")
    return raw


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_bounded_bytes(path, 2_000_000))
    if not isinstance(value, dict):
        raise ValueError("federated input must be a JSON object")
    return value


def _federated_examples(values: list[dict[str, Any]]) -> tuple[BinaryExample, ...]:
    if not isinstance(values, list) or not 4 <= len(values) <= 1024:
        raise ValueError("four to 1024 explicit examples required")
    clocks = (
        "event_time",
        "feature_available_time",
        "decision_time",
        "target_event_time",
        "target_available_time",
    )
    rows = []
    for value in values:
        row = dict(value)
        row["feature_names"] = tuple(row["feature_names"])
        row["features"] = tuple(row["features"])
        for name in clocks:
            row[name] = datetime.fromisoformat(row[name])
        rows.append(BinaryExample(**row))
    return tuple(rows)


def _federated_json(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, datetime):
            return item.isoformat()
        raise TypeError("unsupported federated artifact value")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode)


def _federated_hash(value: Any) -> str:
    return hashlib.sha256(_federated_json(value).encode()).hexdigest()


def read_federated_evidence(run: Path) -> dict[str, Any]:
    from quant_fund.models import federated_market
    from quant_fund.models.federated_market import (
        BinaryTask,
        ClientConfig,
        FederatedModel,
        FederatedServer,
        evaluate_binary,
    )

    root = run.resolve()
    receipt = json.loads(_bounded_bytes(root / "receipt.json", 16_000_000))
    body = receipt["body"]
    if (
        body["schema_version"] != "blueprint_federated_run_v1"
        or _federated_hash(body) != receipt["receipt_sha256"]
    ):
        raise ValueError("federated receipt hash/schema mismatch")
    if set(body["artifacts"]) != {"inputs.json", "model_source.py", "server.json"}:
        raise ValueError("federated artifact contract mismatch")
    budgets = {"inputs.json": 2_000_000, "model_source.py": 2_000_000, "server.json": 16_000_000}
    for name, expected in body["artifacts"].items():
        target = (root / name).resolve()
        if (
            target.parent != root
            or hashlib.sha256(_bounded_bytes(target, budgets[name])).hexdigest() != expected
        ):
            raise ValueError("federated artifact path/hash mismatch")
    if (
        body["artifacts"]["model_source.py"]
        != hashlib.sha256(Path(federated_market.__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("federated measured implementation differs from current code")
    if body["input_sha256"] != body["artifacts"]["inputs.json"]:
        raise ValueError("federated input identity mismatch")
    data = _json(root / "inputs.json")
    holdout = _federated_examples(data["holdout"])
    training = tuple(
        row for client in data["clients"] for row in _federated_examples(client["examples"])
    )
    cutoff = datetime.fromisoformat(data["training_asof"])
    ids = sorted(
        row.record_id
        for row in training
        if row.split == "train" and row.target_available_time <= cutoff
    )
    if ids != body["training_ids"]:
        raise ValueError("federated training identity mismatch")
    server = FederatedServer.load_json(root / "server.json")
    task_data = dict(data["task"])
    task_data["feature_names"] = tuple(task_data["feature_names"])
    task = BinaryTask(**task_data)
    config = ClientConfig(**data["client_config"])
    if len(server.receipts) != data["rounds"] or task != server.freeze().task:
        raise ValueError("federated task/round count mismatch")
    client_rows = {
        client["client_id"]: _federated_examples(client["examples"]) for client in data["clients"]
    }
    if len(client_rows) != len(data["clients"]):
        raise ValueError("federated duplicate client identities")
    for index, saved in enumerate(server.receipts):
        if (
            saved.spec.training_as_of != cutoff
            or saved.spec.client_config != config
            or saved.spec.client_ids != tuple(client_rows)
            or saved.spec.aggregation_as_of
            != datetime.fromisoformat(data["aggregation_asof"]) + timedelta(seconds=index)
        ):
            raise ValueError("federated input/round protocol mismatch")
        for update in saved.updates:
            rows = sorted(
                (
                    row
                    for row in client_rows[update.client_id]
                    if row.split == "train" and row.target_available_time <= cutoff
                ),
                key=lambda row: (row.decision_time, row.record_id),
            )
            expected = _federated_hash([task.task_sha256, [asdict(row) for row in rows]])
            if (
                update.training_source_sha256 != expected
                or update.n_examples != len(rows)
                or update.training_synthetic != any(row.synthetic for row in rows)
                or update.max_feature_availability
                != max(row.feature_available_time for row in rows)
                or update.max_target_availability != max(row.target_available_time for row in rows)
            ):
                raise ValueError("federated update/input source binding mismatch")
    if (
        server.freeze().model_sha256 != body["model_sha256"]
        or body["models"]["global"]["model_sha256"] != body["model_sha256"]
    ):
        raise ValueError("federated model identity mismatch")
    comparison = body["comparison"]
    unhashed = {key: value for key, value in comparison.items() if key != "receipt_sha256"}
    if _federated_hash(unhashed) != comparison["receipt_sha256"]:
        raise ValueError("federated comparison hash mismatch")
    for name, payload in body["models"].items():
        model = FederatedModel.from_payload(payload)
        if name.startswith("local:"):
            updates = {
                f"local:{update.client_id}": update for update in server.receipts[-1].updates
            }
            update = updates[name]
            if (
                model.coefficients != update.coefficients
                or model.intercept != update.intercept
                or model.training_cutoff != update.training_cutoff
                or model.available_as_of != update.completed_at
            ):
                raise ValueError("federated saved local model/update mismatch")
        score = evaluate_binary(
            model,
            holdout,
            as_of=datetime.fromisoformat(data["evaluation_asof"]),
            forbidden_training_ids=ids,
        )
        if score != comparison["scores"][name]:
            raise ValueError("federated heldout score mismatch")
    expected_models = {"global", "centralized", *{f"local:{name}" for name in client_rows}}
    if set(body["models"]) != expected_models:
        raise ValueError("federated comparison model set mismatch")
    budgets = comparison["budgets"]
    if budgets[
        "global_cumulative_optimizer_steps"
    ] != server.freeze().cumulative_optimizer_steps or budgets[
        "local_last_round_optimizer_steps"
    ] != {update.client_id: update.optimizer_steps for update in server.receipts[-1].updates}:
        raise ValueError("federated optimizer budget mismatch")
    if set(body["models"]) != set(comparison["scores"]) or body["synthetic"] != any(
        score["synthetic"] for score in comparison["scores"].values()
    ):
        raise ValueError("federated model/evidence identity mismatch")
    for record in (body, comparison):
        if record["research_only"] is not True or any(
            record[name] is not False
            for name in ("market_evidence", "privacy_guarantee", "process_isolation")
        ):
            raise ValueError("federated honesty claim mismatch")
    if body["source_authenticity_verified"] is not False:
        raise ValueError("supplied client source authenticity is unverified")
    return {
        "receipt_sha256": receipt["receipt_sha256"],
        "model_sha256": body["model_sha256"],
        "scores": comparison["scores"],
        "budgets": comparison["budgets"],
        "compute_budgets_equal": comparison["compute_budgets_equal"],
        "limitations": comparison["limitations"],
        "synthetic": body["synthetic"],
        "aggregation_replayed": True,
        "heldout_scores_recomputed": True,
        "training_repeated": False,
        "client_updates_authenticated": False,
        "privacy_guarantee": False,
        "process_isolation": False,
        "market_evidence": False,
    }
