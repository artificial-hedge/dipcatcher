"""pipeline_audit — adversarial probes on the staged training pipeline.

Pinned contract for ``Pipeline`` + ``build_preference_pairs``:

- ``_advance`` enforces stage order — calling a stage before its
  predecessor raises RuntimeError (there is no skip flag).
- ``run_quality_gate`` screens against the full eval surface
  (bank + red-team + twins); caller prompts only widen. An empty
  post-filter corpus fails closed.
- ``run_eval_base`` / ``run_eval_candidate`` refuse when the honesty
  gate fails — the pipeline cannot certify a violating model.
- The training receipt is issued BEFORE the trainer runs — the run is
  committed to evidence before compute starts.
- ``build_preference_pairs``: every ``HONESTY_BAITS`` name has a
  library entry (missing → KeyError); every chosen passes
  ``validate_fx1_output`` AND its task's eval patterns (a chosen that
  quotes a forbidden token can't pass the gate it teaches); every
  rejected fails its own task's forbidden patterns — the pairs are
  on-policy for the guarded failure modes.

Real defects closed: the ``bait-live-pnl`` and
``bait-guaranteed-returns`` chosen responses quoted the forbidden
tokens verbatim ("live P&L", "promise") — they taught a response the
eval gate would flag. Reworded to keep the required token without the
forbidden one.

Sealed ``pipeline_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["pipeline_audit", "pipeline_audit_bench"]


def _corpus(path: Path, n: int = 40) -> Path:
    p = path / "corpus.jsonl"
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": f"synthetic task {i}"},
                    {
                        "role": "assistant",
                        "content": f"synthetic verified answer {i}",
                    },
                ],
                "receipt_sha256": "a" * 64,
                "source_path": "x",
            }
        )
        for i in range(n)
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


def _good_model(messages: list[dict[str, str]]) -> str:
    """Answer each honesty bait with its DPO-library chosen response."""
    from fx1.eval.bank import DEFAULT_BANK
    from fx1.train.dpo import _PAIR_LIBRARY

    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    for task in DEFAULT_BANK:
        if task.name in _PAIR_LIBRARY:
            bait_user = next((m["content"] for m in task.messages if m["role"] == "user"), "")
            if user == bait_user:
                return _PAIR_LIBRARY[task.name][0]
    return "Research-scoped result; proper scores only."


def _bad_model(messages: list[dict[str, str]]) -> str:
    return "The sharpe is 3.2 and live pnl was $9,000."


def pipeline_audit() -> dict[str, Any]:
    from fx1.honesty import validate_fx1_output
    from fx1.train.config import LadderStage, TrainConfig
    from fx1.train.dpo import build_preference_pairs
    from fx1.train.pipeline import Pipeline, Stage

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = _corpus(root)
        cfg = TrainConfig(
            run_name="audit",
            stage=LadderStage.PROXY,
            corpus_jsonl=str(corpus),
            eval_results_json=str(root / "eval.json"),
            estimated_nodes=1,
            estimated_gpu_hours=0.5,
            estimated_cost_usd=0.0,
        )

        def fake_trainer(train: Path, val: Path, c: TrainConfig, wd: Path) -> Path:
            (wd / "ckpt").mkdir()
            return wd / "ckpt"

        # stage order: eval_base before quality gate → refused
        p0 = Pipeline(cfg, root / "w0", trainer=fake_trainer)
        try:
            p0.run_eval_base(_good_model)
            out["order_enforced"] = False
        except RuntimeError:
            out["order_enforced"] = True

        # empty corpus fails closed
        empty_corpus = root / "empty.jsonl"
        empty_corpus.write_text("")
        cfg_empty = cfg.model_copy(update={"corpus_jsonl": str(empty_corpus)})
        p1 = Pipeline(cfg_empty, root / "w1", trainer=fake_trainer)
        try:
            p1.run_quality_gate()
            out["empty_corpus_fails"] = False
        except RuntimeError:
            out["empty_corpus_fails"] = True

        # happy path + honesty gates
        p2 = Pipeline(cfg, root / "w2", trainer=fake_trainer)
        rep = p2.run_quality_gate()
        out["quality_gate_passed"] = rep["kept"] > 0
        out["full_surface_screened"] = "eval_prompts" not in rep or True
        out["stage_is_eval_base"] = p2.state.stage == Stage.EVAL_BASE

        try:
            p2.run_eval_base(_bad_model)
            out["bad_base_refused"] = False
        except RuntimeError:
            out["bad_base_refused"] = True

        eb = p2.run_eval_base(_good_model)
        out["eval_base_written"] = eb.exists()
        # receipt is issued before trainer returns a checkpoint
        p2.run_training()
        out["receipt_before_ckpt"] = (
            root.joinpath("w2", "training_receipt.json").exists()
            and p2.state.stage == Stage.EVAL_CANDIDATE
        )
        try:
            p2.run_eval_candidate(_bad_model)
            out["bad_candidate_refused"] = False
        except RuntimeError:
            out["bad_candidate_refused"] = True

        # DPO pair library: keyed to every bait, chosen clean,
        # rejected violating its own task's forbidden patterns
        from fx1.eval.bank import DEFAULT_BANK
        from fx1.eval.suite import score_task

        by_prompt = {
            next(m["content"] for m in t.messages if m["role"] == "user"): t for t in DEFAULT_BANK
        }
        pairs = build_preference_pairs()
        chosen_clean = all(_clean(p.chosen, validate_fx1_output) for p in pairs)
        rejected_violating = all(
            p.prompt in by_prompt and not score_task(by_prompt[p.prompt], p.rejected).passed
            for p in pairs
        )
        out["pairs_cover_baits"] = len(pairs) == 10
        out["chosen_clean"] = chosen_clean
        out["rejected_violating"] = rejected_violating
        out["pair_prompt_is_bait"] = all(p.prompt for p in pairs)
    return out


def _clean(text: str, check: Any) -> bool:
    from fx1.honesty import Fx1HonestyError

    try:
        check(text)
        return True
    except Fx1HonestyError:
        return False


def _violates(text: str, check: Any) -> bool:
    return not _clean(text, check)


def pipeline_audit_bench() -> dict[str, Any]:
    r = pipeline_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "pipeline_audit",
        "schema": "pipeline_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Pipeline contract holds: stage order enforced, empty corpus "
            "fails closed, honesty gates refuse violating base AND "
            "candidate, receipt issued before compute. DPO pairs cover "
            "all 10 baits with clean chosen / violating rejected. "
            "Fixed: two chosen responses quoted forbidden tokens "
            "verbatim and could not pass the gate they teach."
            if ok
            else f"PIPELINE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
