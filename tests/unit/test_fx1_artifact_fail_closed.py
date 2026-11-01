"""Fail-closed artifact and benchmark validation regressions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fx1.harness_bench import run_bench
from fx1.modelcard import EvalDelta, ModelCard
from fx1.sdk import Fx1Harness
from fx1.serve.finetune import FTJobOutcome

_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'


def test_finetune_missing_result_artifact_fails_job(tmp_path: Path) -> None:
    def missing_artifact(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts={"missing.json": spec.work_dir / "missing.json"},
        )

    sdk = Fx1Harness(ft_runner=missing_artifact, ft_dir=tmp_path)
    job = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    assert job.status == "failed"
    assert job.result_files == []
    assert job.error is not None
    assert "cannot read fine-tune result artifact 'missing.json'" in job.error.message


def _card(**overrides: Any) -> ModelCard:
    values: dict[str, Any] = {
        "version": "fx-1.v0.1",
        "corpus_sha256": "a" * 64,
        "corpus_receipt_range": "b..c",
        "training_manifest_sha256": "b" * 64,
        "eval_delta": EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.6,
            general_pass_rate_base=0.7,
            general_pass_rate_candidate=0.7,
            honesty_gate_candidate=True,
        ),
    }
    values.update(overrides)
    return ModelCard(**values)


@pytest.mark.parametrize(
    "field", ["base_checkpoint_sha256", "corpus_sha256", "training_manifest_sha256"]
)
def test_modelcard_rejects_nonhex_provenance_hashes(field: str) -> None:
    with pytest.raises(ValidationError, match="pattern"):
        _card(**{field: "g" * 64})


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf")])
def test_bench_rejects_nonfinite_timeouts_before_calls(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be finite"):
        run_bench(object(), timeout_s=timeout_s)
