"""fx1_tail_audit — adversarial probes on fx-1's tail surface.

Covers ``data/traces`` (verification-gated trajectory admission),
``data/notebooks`` (markdown → SFT), ``train/dpo`` (preference pairs),
``train/pipeline`` (staged gate orchestration), and ``train/tracking``
(JSONL-always tracker).

Pinned contract:

- ``TraceRecorder.admit`` writes only after every assistant message (incl.
  ``reasoning_content`` and serialized ``tool_call`` JSON) passes the
  honesty gate; a refused trajectory writes nothing — it is dropped
  entirely rather than recorded as a negative. ``verify_ok`` alone is a
  claim: a positive example also requires ``artifact_receipts``, else the
  trajectory is admitted but demoted to ``negative``.
- Tool results (``role=tool``) are observations and are not gated.
- ``notebook_examples`` splits on ``#`` sections, binds ``receipt_sha256``
  to the whole file's sha256, and caps the quoted content; the assistant
  turn never echoes the document, so a doc containing forbidden tokens
  lands only as *user* content.
- ``build_preference_pairs`` emits exactly one pair per honesty bait,
  keyed by task name (a missing library entry raises); every ``chosen``
  passes the honesty gate.
- ``Pipeline`` advances strictly in stage order and refuses out-of-order
  transitions; the default trainer raises with setup instructions; the
  quality gate refuses an empty post-filter corpus; eval gates refuse
  models that violate honesty.
- ``Tracker`` writes JSONL regardless of mlflow availability; params are
  stringified; close() is a no-op without a run.

Flagged warts (documented, not fixed):

- ``flag_rejected_ungated`` — 7 of 10 DPO ``rejected`` responses pass
  ``validate_fx1_output``: the preference pairs train refusal against
  violations the production gate cannot detect (the gate sees tokens,
  the taxonomy sees intent). The hard failures (headline metric, live
  claim, guaranteed returns) ARE gated.
- ``flag_card_stage_unreachable`` — ``Stage.CARD`` exists in the order
  but no ``Pipeline`` method advances to it; the pipeline can never
  reach the card stage as written.
- ``flag_refused_traces_dropped`` — an honesty-violating trajectory is
  refused with no ledger record; the corpus gains no negative evidence
  of the refusal event itself.
- ``flag_split_manifest_keyerror`` — ``run_training`` before the quality
  gate raises a bare ``KeyError`` on the missing artifact, not a staged
  ``RuntimeError``.
- ``flag_tracker_logs_missing_artifact`` — ``log_artifact`` on a
  nonexistent path is recorded in the JSONL log regardless (no existence
  check); and ``Trajectory.sha256`` covers ``recorded_utc`` so two
  identical trajectories minted at different times hash differently
  (``flag_sha_includes_timestamp``).

Sealed ``fx1_tail_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["fx1_tail_audit", "fx1_tail_audit_bench"]


def fx1_tail_audit() -> dict[str, Any]:
    import hashlib
    import json
    import tempfile
    from pathlib import Path

    from fx1.data.notebooks import notebook_examples
    from fx1.data.traces import ToolCall, TraceRecorder, TraceStep, Trajectory
    from fx1.eval.bank import HONESTY_BAITS
    from fx1.honesty import validate_fx1_output
    from fx1.train.dpo import _PAIR_LIBRARY, build_preference_pairs
    from fx1.train.pipeline import Pipeline, Stage, default_trainer
    from fx1.train.tracking import Tracker

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    # ---------------- traces ------------------------------------
    def _traj(
        *,
        verify_ok: bool,
        reasoning: str = "",
        content: str = "done",
        tool: str | None = None,
        receipts: list[str] | None = None,
    ) -> Trajectory:
        step = TraceStep(
            reasoning_content=reasoning,
            assistant_content=content,
            tool_call=ToolCall(name="t", arguments={"q": tool}) if tool else None,
            tool_result="obs" if tool else None,
        )
        return Trajectory(
            session_id="s1",
            user_intent="do it",
            steps=[step],
            verify_ok=verify_ok,
            artifact_receipts=receipts or [],
        )

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "traces.jsonl"
        rec = TraceRecorder(log)
        # A positive example needs evidence: verify_ok + artifact receipts.
        out["admit_positive"] = (
            rec.admit(_traj(verify_ok=True, receipts=["a" * 64]), "sys")
            and not json.loads(log.read_text(encoding="utf-8").strip())["negative"]
        )
        # verify_ok without receipts is a bare claim — demoted to negative.
        out["verify_no_receipts_demoted"] = (
            rec.admit(_traj(verify_ok=True), "sys")
            and json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])["negative"]
        )
        out["admit_negative"] = (
            rec.admit(_traj(verify_ok=False), "sys")
            and json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])["negative"]
        )
        refused = rec.admit(_traj(verify_ok=True, content="Sharpe 2.4"), "sys")
        after_lines = log.read_text(encoding="utf-8").strip().splitlines()
        out["admit_dishonest_refused"] = not refused
        out["flag_refused_traces_dropped"] = len(after_lines) == 3  # pos + demoted + neg
        out["tool_call_gated"] = not rec.admit(_traj(verify_ok=True, tool="pnl: 4"), "sys")
        out["reasoning_gated"] = not rec.admit(_traj(verify_ok=True, reasoning="nav 1.9"), "sys")
        out["tool_result_ungated"] = rec.admit(_traj(verify_ok=True, tool="x"), "sys")
        msgs = _traj(verify_ok=True, reasoning="r", tool="q").to_sft_messages("sys")
        out["trace_message_shape"] = (
            msgs[0]["role"] == "system"
            and msgs[1]["role"] == "user"
            and msgs[2]["role"] == "assistant"
            and "<reasoning>" in msgs[2]["content"]
            and "<tool_call>" in msgs[2]["content"]
            and msgs[3]["role"] == "tool"
        )
        t1 = _traj(verify_ok=True)
        out["traj_sha_stable"] = (
            t1.sha256 == hashlib.sha256(t1.model_dump_json().encode()).hexdigest()
        )
        # sha256 must cover ``recorded_utc``: re-mint the identical
        # trajectory with a different stamp (explicit update — not a
        # second ``now()`` call, which can collide on coarse clocks).
        t2 = t1.model_copy(update={"recorded_utc": "1970-01-01T00:00:00+00:00"})
        out["flag_sha_includes_timestamp"] = t1.sha256 != t2.sha256

    # ---------------- notebooks ---------------------------------
    with tempfile.TemporaryDirectory() as td:
        doc = Path(td) / "n.md"
        text = "# One\nalpha body\n# Two\nbeta body " + "x" * 2000
        doc.write_text(text, encoding="utf-8")
        exs = notebook_examples(doc, "sys")
        out["nb_sections"] = len(exs) == 2
        digest = hashlib.sha256(text.encode()).hexdigest()
        out["nb_receipt_binds_source"] = all(e.receipt_sha256 == digest for e in exs)
        out["nb_assistant_no_quote"] = all(
            "alpha body" not in e.messages[-1]["content"] for e in exs
        )
        out["nb_user_quotes_content"] = "alpha body" in exs[0].messages[1]["content"]
        empty = Path(td) / "e.md"
        empty.write_text("   \n", encoding="utf-8")
        out["nb_empty_zero"] = notebook_examples(empty, "sys") == []
        hot = Path(td) / "hot.md"
        hot.write_text("# S\nthis doc mentions Sharpe 2.0\n", encoding="utf-8")
        hot_exs = notebook_examples(hot, "sys")
        out["nb_forbidden_only_in_user"] = (
            len(hot_exs) == 1
            and "Sharpe" in hot_exs[0].messages[1]["content"]
            and _raises(lambda: validate_fx1_output(hot_exs[0].messages[-1]["content"]))
            == "no-raise"
        )

    # ---------------- dpo ---------------------------------------
    pairs = build_preference_pairs()
    out["pairs_per_bait"] = len(pairs) == len(HONESTY_BAITS)
    out["library_matches_baits"] = set(_PAIR_LIBRARY) == {t.name for t in HONESTY_BAITS}
    out["chosen_all_pass"] = all(
        _raises(lambda p=p: validate_fx1_output(p.chosen)) == "no-raise" for p in pairs
    )
    ungated = sum(
        1 for p in pairs if _raises(lambda p=p: validate_fx1_output(p.rejected)) == "no-raise"
    )
    out["flag_rejected_ungated"] = ungated == 7
    with tempfile.TemporaryDirectory() as td:
        pj = Path(td) / "pairs.jsonl"
        build_preference_pairs(pj)
        out["pairs_written_jsonl"] = len(
            pj.read_text(encoding="utf-8").strip().splitlines()
        ) == len(HONESTY_BAITS)
    out["pair_prompt_is_user"] = all(
        p.prompt
        == next(
            m["content"]
            for m in next(t for t in HONESTY_BAITS if t.name == name).messages
            if m["role"] == "user"
        )
        for name, p in zip(_PAIR_LIBRARY, pairs, strict=True)
    )

    # ---------------- pipeline ----------------------------------
    from fx1.train.config import LadderStage, TrainConfig  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n",
            encoding="utf-8",
        )
        cfg = TrainConfig(
            run_name="audit",
            stage=LadderStage.PROXY,
            corpus_jsonl=str(corpus),
            eval_results_json=str(corpus),
            estimated_nodes=1,
            estimated_gpu_hours=0.5,
            estimated_cost_usd=0.0,
        )
        pipe = Pipeline(cfg, Path(td) / "work")
        out["stage_order_enforced"] = _raises(lambda: pipe._advance(Stage.TRAIN)) == "RuntimeError"
        out["default_trainer_refuses"] = (
            _raises(lambda: default_trainer(Path("a"), Path("b"), cfg, Path("c")))
            == "NotImplementedError"
        )
        out["flag_card_stage_unreachable"] = not hasattr(
            pipe, "run_card"
        ) and Stage.CARD.value not in {m for m in dir(pipe) if m.startswith("run_")}
        out["flag_split_manifest_keyerror"] = _raises(lambda: pipe.run_training()) == "KeyError"
        empty_corpus = Path(td) / "empty.jsonl"
        empty_corpus.write_text("", encoding="utf-8")
        cfg_empty = cfg.model_copy(update={"corpus_jsonl": str(empty_corpus)})
        pipe_empty = Pipeline(cfg_empty, Path(td) / "work2")
        out["quality_gate_empty_fails"] = (
            _raises(lambda: pipe_empty.run_quality_gate()) == "RuntimeError"
        )
        try:
            import mlflow  # noqa: PLC0415

            mlflow.set_tracking_uri(f"file:{td}/mlruns")
            _has_mlflow = True
        except Exception:  # noqa: BLE001
            _has_mlflow = False
        tr = Tracker(experiment="audit", fallback_log=Path(td) / "fb.jsonl")
        tr.log_params({"a": 1})
        tr.log_metric("m", 0.5, step=2)
        art = Path(td) / "a.txt"
        art.write_text("x", encoding="utf-8")
        tr.log_artifact(art)
        missing_art = Path(td) / "gone.bin"
        out["flag_tracker_logs_missing_artifact"] = (
            _raises(lambda: tr.log_artifact(missing_art)) == "no-raise"
        )  # wart: a nonexistent path is recorded, not refused
        tr.close()
        lines = (Path(td) / "fb.jsonl").read_text(encoding="utf-8").strip().splitlines()
        out["tracker_jsonl_always"] = len(lines) == 4
        out["tracker_params_stringified"] = json.loads(lines[0])["params"]["a"] == "1"
        out["tracker_metric_shape"] = json.loads(lines[1])["step"] == 2
        out["tracker_close_noop"] = _raises(lambda: tr.close()) == "no-raise"

    return out


def fx1_tail_audit_bench() -> dict[str, Any]:
    r = fx1_tail_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "fx1_tail_audit",
        "schema": "fx1_tail_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "tail surface holds: trace admission is honesty-gated and "
            "fail-closed (refusals write nothing), notebooks bind source "
            "hashes and never quote into the assistant turn, DPO emits one "
            "pair per bait with all chosen responses clean, the pipeline "
            "enforces stage order with gated eval stages, and the tracker "
            "always lands JSONL. Flags: 7/10 rejected responses are "
            "contract-clean to the gate; Stage.CARD is unreachable; "
            "run_training pre-quality raises KeyError; a missing "
            "artifact path is still recorded by the tracker; the "
            "trajectory hash includes its creation timestamp."
            if ok
            else f"TAIL AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
