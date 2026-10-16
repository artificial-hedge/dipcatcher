"""SYNTHETIC correctness tests for offline public-eval artifact checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from fx1.eval.evidence_contract import (
    MAX_BUNDLE_BYTES,
    EvalEvidenceError,
    HeldoutBankManifest,
    audit_signed_case_consistency,
)
from fx1.modelcard import EvalDelta, ModelCard
from fx1.serve.release_gate import _independent_public_eval_verified


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _synthetic_case_fixture() -> tuple[dict[str, Any], Ed25519PrivateKey]:
    evaluator = Ed25519PrivateKey.from_private_bytes(b"\x51" * 32)
    release = Ed25519PrivateKey.from_private_bytes(b"\x52" * 32)
    base_pin = _sha(b"synthetic base weights")
    candidate_pin = _sha(b"synthetic candidate weights")
    config_pin = _sha(b"synthetic inference config")
    tokenizer_pin = _sha(b"synthetic tokenizer")
    tasks = [
        {
            "name": f"synthetic-honesty-{i}",
            "kind": "honesty",
            "messages": [{"role": "user", "content": f"Synthetic honesty question {i}"}],
            "required_tokens": ["safe"],
        }
        for i in range(10)
    ]
    tasks.extend(
        {
            "name": f"synthetic-domain-{i}",
            "kind": "domain",
            "messages": [{"role": "user", "content": f"Synthetic domain question {i}"}],
            "required_tokens": ["answer"],
        }
        for i in range(30)
    )
    tasks.extend(
        {
            "name": f"synthetic-general-{i}",
            "kind": "general",
            "messages": [{"role": "user", "content": f"Synthetic general question {i}"}],
            "required_tokens": ["answer"],
            "enforce_honesty": False,
        }
        for i in range(30)
    )
    bank_bytes = _json(
        {"schema": "fx1_heldout_bank.v1", "bank_id": "synthetic-test-bank", "tasks": tasks}
    )
    start = {
        "schema": "fx1_eval_run_start.v1",
        "run_id": "synthetic-test-run",
        "model_id": "fx1",
        "challenge_nonce_hex": "a" * 64,
        "bank_sha256": _sha(bank_bytes),
        "base_weights_sha256": base_pin,
        "candidate_weights_sha256": candidate_pin,
        "inference_config_sha256": config_pin,
        "tokenizer_sha256": tokenizer_pin,
        "evaluator_public_key_hex": evaluator.public_key().public_bytes_raw().hex(),
    }
    start_bytes = _json(start)
    cases = []
    for task in tasks:
        kind = task["kind"]
        cases.append(
            {
                "task": task["name"],
                "base_response": "wrong" if kind == "domain" else "answer safe",
                "candidate_response": "answer safe",
            }
        )
    bundle = {
        "schema": "fx1_eval_cases.v1",
        "data_label": "SYNTHETIC",
        "run_id": start["run_id"],
        "model_id": "fx1",
        "run_start_sha256": _sha(start_bytes),
        "bank_sha256": _sha(bank_bytes),
        "base_weights_sha256": base_pin,
        "candidate_weights_sha256": candidate_pin,
        "inference_config_sha256": config_pin,
        "tokenizer_sha256": tokenizer_pin,
        "cases": cases,
    }
    bundle_bytes = _json(bundle)
    card = ModelCard(
        version="fx-1.v0.1",
        base_checkpoint_sha256=base_pin,
        corpus_sha256="b" * 64,
        corpus_receipt_range="synthetic-test-only",
        training_manifest_sha256="c" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.0,
            domain_pass_rate_candidate=1.0,
            general_pass_rate_base=1.0,
            general_pass_rate_candidate=1.0,
            honesty_gate_candidate=True,
        ),
        public_model_id="fx1",
        fixture_only=True,
    )
    args = {
        "bank_bytes": bank_bytes,
        "expected_bank_sha256": _sha(bank_bytes),
        "run_start_bytes": start_bytes,
        "expected_start_sha256": _sha(start_bytes),
        "bundle_bytes": bundle_bytes,
        "bundle_signature_hex": evaluator.sign(bundle_bytes).hex(),
        "evaluator_key_hex": evaluator.public_key().public_bytes_raw().hex(),
        "release_key_hex": release.public_key().public_bytes_raw().hex(),
        "expected_candidate_sha256": candidate_pin,
        "expected_inference_sha256": config_pin,
        "expected_tokenizer_sha256": tokenizer_pin,
        "card": card,
    }
    return args, evaluator


def _resign(args: dict[str, Any], evaluator: Ed25519PrivateKey, bundle: dict[str, Any]) -> None:
    args["bundle_bytes"] = _json(bundle)
    args["bundle_signature_hex"] = evaluator.sign(args["bundle_bytes"]).hex()


def test_synthetic_signed_case_bundle_only_proves_consistency(tmp_path: Path) -> None:
    args, _ = _synthetic_case_fixture()
    audit = audit_signed_case_consistency(**args)
    assert audit.claimed_data_label == "SYNTHETIC"
    assert audit.computed_delta == args["card"].eval_delta
    assert audit.domain_comparison.significant_improvement
    assert not _independent_public_eval_verified(tmp_path)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("signature", "signature does not cover"),
        ("release_key", "must differ from release signer"),
        ("bank_pin", "held-out bank differs"),
        ("run_start_pin", "bank, run, alias, or evaluator binding differs"),
        ("candidate_pin", "candidate, inference, or tokenizer pin differs"),
        ("case_order", "exact ordered bank"),
        ("card_delta", "differ from signed model card"),
        ("honesty_general", "differ from signed model card"),
        ("label", "invalid SignedCaseBundle"),
    ],
)
def test_synthetic_case_bundle_refusal(change: str, error: str) -> None:
    args, evaluator = _synthetic_case_fixture()
    bundle = json.loads(args["bundle_bytes"])
    if change == "signature":
        args["bundle_signature_hex"] = "0" * 128
    elif change == "release_key":
        args["release_key_hex"] = args["evaluator_key_hex"].upper()
    elif change == "bank_pin":
        args["expected_bank_sha256"] = "0" * 64
    elif change == "run_start_pin":
        start = json.loads(args["run_start_bytes"])
        start["challenge_nonce_hex"] = "b" * 64
        args["run_start_bytes"] = _json(start)
        args["expected_start_sha256"] = _sha(args["run_start_bytes"])
    elif change == "candidate_pin":
        bundle["candidate_weights_sha256"] = "0" * 64
        _resign(args, evaluator, bundle)
    elif change == "case_order":
        bundle["cases"][0], bundle["cases"][1] = bundle["cases"][1], bundle["cases"][0]
        _resign(args, evaluator, bundle)
    elif change == "card_delta":
        args["card"] = args["card"].model_copy(
            update={
                "eval_delta": args["card"].eval_delta.model_copy(
                    update={"domain_pass_rate_base": 0.1}
                )
            }
        )
    elif change == "honesty_general":
        bundle["cases"][-1]["candidate_response"] = "answer Sharpe: 2.4"
        _resign(args, evaluator, bundle)
    else:
        bundle["data_label"] = "UNLABELED"
        _resign(args, evaluator, bundle)
    with pytest.raises(EvalEvidenceError, match=error):
        audit_signed_case_consistency(**args)


def test_synthetic_case_bundle_rejects_duplicate_fields_and_large_input() -> None:
    args, evaluator = _synthetic_case_fixture()
    args["bundle_bytes"] = b'{"schema":"fx1_eval_cases.v1",' + args["bundle_bytes"][1:]
    args["bundle_signature_hex"] = evaluator.sign(args["bundle_bytes"]).hex()
    with pytest.raises(EvalEvidenceError, match="duplicate JSON field"):
        audit_signed_case_consistency(**args)
    args["bundle_bytes"] = b"x" * (MAX_BUNDLE_BYTES + 1)
    with pytest.raises(EvalEvidenceError, match="oversized evaluation artifact"):
        audit_signed_case_consistency(**args)


def test_synthetic_toy_bank_cannot_meet_structural_floor() -> None:
    args, _ = _synthetic_case_fixture()
    bank = json.loads(args["bank_bytes"])
    bank["tasks"] = bank["tasks"][:3]
    args["bank_bytes"] = _json(bank)
    args["expected_bank_sha256"] = _sha(args["bank_bytes"])
    with pytest.raises(EvalEvidenceError, match="invalid HeldoutBankManifest"):
        audit_signed_case_consistency(**args)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("vacuous", "scoring criterion"),
        ("system_prompt", "user/assistant messages"),
        ("disabled_honesty", "must enforce"),
    ],
)
def test_synthetic_bank_rejects_weak_task_definitions(change: str, error: str) -> None:
    args, _ = _synthetic_case_fixture()
    bank = json.loads(args["bank_bytes"])
    if change == "vacuous":
        bank["tasks"][0]["required_tokens"] = []
    elif change == "system_prompt":
        bank["tasks"][0]["messages"][0]["role"] = "system"
    else:
        bank["tasks"][0]["enforce_honesty"] = False
    with pytest.raises(ValidationError, match=error):
        HeldoutBankManifest.model_validate(bank)
