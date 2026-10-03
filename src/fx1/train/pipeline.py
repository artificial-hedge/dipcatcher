"""The fx-1 staged training pipeline — industry-grade orchestration.

Stages, each gated and receipted:
  data -> quality -> eval_base -> train -> eval_candidate -> card

The trainer itself is injectable: in this repo the default raises with setup
instructions (torch/trl and a cluster are required); tests inject fakes. A
stage that cannot produce its evidence stops the pipeline — there is no
"skip" flag, mirroring the harness's fail-closed gates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fx1.data.quality import dedup_and_filter, frozen_split
from fx1.eval.bank import DEFAULT_BANK
from fx1.eval.compare import compare_runs
from fx1.eval.suite import run_suite
from fx1.train.config import TrainConfig
from fx1.train.receipts import issue_receipt, loads_receipt
from quant_fund.utils.atomicio import atomic_write_text


class Stage(StrEnum):
    DATA = "data"
    QUALITY = "quality"
    EVAL_BASE = "eval_base"
    TRAIN = "train"
    EVAL_CANDIDATE = "eval_candidate"
    CARD = "card"


STAGE_ORDER = list(Stage)

# Trainer signature: (train_jsonl, val_jsonl, config, work_dir) -> checkpoint dir
TrainerFn = Callable[[Path, Path, TrainConfig, Path], Path]
# Model function for eval stages: messages -> response
ModelFn = Callable[[list[dict[str, str]]], str]


def default_trainer(
    train_jsonl: Path, val_jsonl: Path, config: TrainConfig, work_dir: Path
) -> Path:
    raise NotImplementedError(
        "GPU training requires torch/trl/peft and a cluster per "
        "docs/FX1_TRAINING.md. Inject a trainer or run the generated cluster "
        "spec (fx1.train.cluster) on your scheduler."
    )


class PipelineState(BaseModel):
    stage: Stage = Stage.DATA
    artifacts: dict[str, str] = {}
    metrics: dict[str, float] = {}


class Pipeline:
    """Runs fx-1 training stages with hard gates between them."""

    def __init__(
        self,
        config: TrainConfig,
        work_dir: str | Path,
        *,
        trainer: TrainerFn = default_trainer,
    ) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = trainer
        self.state = PipelineState()

    def _advance(self, stage: Stage, **artifacts: str) -> None:
        expected = STAGE_ORDER[STAGE_ORDER.index(self.state.stage)]
        if stage != expected:
            raise RuntimeError(f"pipeline gate violation: expected stage {expected}, got {stage}")
        self.state.artifacts.update(artifacts)
        nxt = STAGE_ORDER.index(stage) + 1
        if nxt < len(STAGE_ORDER):
            self.state.stage = STAGE_ORDER[nxt]

    def run_quality_gate(self, eval_prompts: list[str] | None = None) -> dict[str, Any]:
        """DATA + QUALITY: load corpus, dedup/decontaminate, frozen split."""
        corpus_path = Path(self.config.corpus_jsonl)
        examples = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # The full eval surface — canonical bank, red-team, rephrased and
        # masked twins — is always screened; caller-supplied prompts only
        # widen the target, never narrow it.
        from fx1.eval import eval_prompt_surface

        prompts = eval_prompt_surface() + list(eval_prompts or [])
        kept, report = dedup_and_filter(examples, eval_prompts=prompts)
        if report.kept == 0:
            raise RuntimeError("quality gate: corpus empty after filtering")
        manifest = frozen_split(kept, self.work_dir / "corpus")
        self.state.metrics["corpus_kept"] = float(report.kept)
        self.state.metrics["val_count"] = float(manifest.val_count)
        self._advance(Stage.DATA, corpus=str(corpus_path))
        self._advance(
            Stage.QUALITY,
            train=f"{self.work_dir}/corpus.train.jsonl",
            val=f"{self.work_dir}/corpus.val.jsonl",
            split_manifest=f"{self.work_dir}/corpus.split.json",
            quality_report=str(self._write("quality_report.json", report.model_dump())),
        )
        return report.model_dump()

    def run_eval_base(self, base_model_fn: ModelFn) -> Path:
        """EVAL_BASE: recorded base-model results are mandatory pre-training."""
        summary = run_suite(base_model_fn, list(DEFAULT_BANK))
        if not summary["honesty_gate_passed"]:
            raise RuntimeError(
                "base model fails the honesty gate; fix the base or the "
                "system contract before training"
            )
        out = self._write("eval_base.json", summary)
        self._advance(Stage.EVAL_BASE, eval_base=str(out))
        return out

    def run_training(self, seed: int = 17) -> Path:
        """TRAIN: emit the immutable receipt, then invoke the trainer."""
        receipt = issue_receipt(
            run_name=self.config.run_name,
            repo_root=Path.cwd(),
            config_path=self._write("train_config.json", self.config.model_dump(mode="json")),
            corpus_path=self.config.corpus_jsonl,
            split_manifest_path=self.state.artifacts["split_manifest"],
            eval_base_path=self.state.artifacts["eval_base"],
            seed=seed,
            out_path=self.work_dir / "training_receipt.json",
        )
        checkpoint = self.trainer(
            Path(self.state.artifacts["train"]),
            Path(self.state.artifacts["val"]),
            self.config,
            self.work_dir,
        )
        self._advance(
            Stage.TRAIN,
            training_receipt=str(self.work_dir / "training_receipt.json"),
            checkpoint=str(checkpoint),
        )
        self.state.metrics["receipt_dirty"] = float(receipt.dirty_worktree)
        return checkpoint

    def run_eval_candidate(self, candidate_fn: ModelFn) -> dict[str, Any]:
        """EVAL_CANDIDATE: statistical comparison against the recorded base.

        The base line is re-verified against the hash the training receipt
        pinned at EVAL_BASE time: a stale or tampered eval_base.json cannot
        silently re-baseline the ship gate. Results are paired by task
        name, not position, and both summaries must pin the same eval-bank
        hash.
        """
        base_path = Path(self.state.artifacts["eval_base"])
        receipt_path = self.state.artifacts.get("training_receipt")
        if receipt_path:
            receipt = loads_receipt(receipt_path)
            actual = hashlib.sha256(base_path.read_bytes()).hexdigest()
            if actual != receipt.eval_base_sha256:
                raise RuntimeError(
                    "eval_base.json no longer matches the hash pinned in the "
                    "training receipt; re-run the base eval rather than "
                    "re-baselining the ship gate on tampered or stale results"
                )
        base_summary = json.loads(base_path.read_text(encoding="utf-8"))
        cand_summary = run_suite(candidate_fn, list(DEFAULT_BANK))
        cand_out = self._write("eval_candidate.json", cand_summary)
        if not cand_summary["honesty_gate_passed"]:
            raise RuntimeError(
                "candidate fails the honesty gate; the comparison must not "
                "certify a model that violates the contract"
            )
        if not isinstance(base_summary, dict) or not isinstance(base_summary.get("results"), list):
            raise RuntimeError("eval_base.json is not a suite summary")
        base_bank = base_summary.get("eval_bank_sha256")
        if not base_bank or base_bank != cand_summary.get("eval_bank_sha256"):
            raise RuntimeError(
                "base and candidate eval summaries pin different eval banks; "
                "comparing them would certify nothing"
            )
        base_by_name: dict[str, Any] = {}
        for r in base_summary["results"]:
            name = r.get("task")
            if not isinstance(name, str) or not name:
                raise RuntimeError("eval_base.json contains a result without a task name")
            base_by_name[name] = r
        cand_by_name = {str(r.get("task")): r for r in cand_summary.results}
        comparison = self._compare_by_kind(base_by_name, cand_by_name, kind="domain")
        if not comparison:
            raise RuntimeError(
                "no domain tasks in the eval bank — there is no ship gate "
                "to certify a candidate against"
            )
        general = self._compare_by_kind(base_by_name, cand_by_name, kind="general")
        if general:
            comparison["general"] = general
        comp_out = self._write("comparison.json", comparison)
        self._advance(
            Stage.EVAL_CANDIDATE,
            eval_candidate=str(cand_out),
            comparison=str(comp_out),
        )
        return comparison

    @staticmethod
    def _compare_by_kind(
        base_by_name: dict[str, Any], cand_by_name: dict[str, Any], *, kind: str
    ) -> dict[str, Any]:
        """Pair same-kind results by task name and compare them."""
        base_tasks = {n for n, r in base_by_name.items() if r.get("kind") == kind}
        cand_tasks = {n for n, r in cand_by_name.items() if r.get("kind") == kind}
        if base_tasks != cand_tasks:
            raise RuntimeError(
                f"base and candidate evals cover different {kind} tasks "
                f"(only base: {sorted(base_tasks - cand_tasks)}, only "
                f"candidate: {sorted(cand_tasks - base_tasks)})"
            )
        if not base_tasks:
            return {}
        names = sorted(base_tasks)
        comparison = compare_runs(
            [bool(base_by_name[n]["passed"]) for n in names],
            [bool(cand_by_name[n]["passed"]) for n in names],
        )
        return dict(comparison.model_dump())

    def _write(self, name: str, payload: dict[str, Any]) -> Path:
        out = self.work_dir / name
        atomic_write_text(out, json.dumps(payload, indent=2))
        return out
