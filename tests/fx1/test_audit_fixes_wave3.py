"""Regression tests for the third fx-1 internals audit wave (docs/AUDIT_FX1_WAVE3.md).

Each test pins a defect found reading src/fx1 line by line after the two
prior audit rounds: receipt-pinned baselines never re-verified, positional
result pairing, boundary-ambiguous split hashing, windowed near-duplicate
detection, truthy non-boolean contract fields, substring-in-token honesty
evasions, off-by-one recovery horizons, silent drops, faith-based verify_ok,
and --config containment bypass via extra_args. No existing threshold is
relaxed.
"""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.bench.dip import (
    DipForecast,
    assert_bench_output_honest,
    detect_dip_events,
    evaluate_forecasts,
)
from fx1.data.ledger import CorpusLedger
from fx1.data.quality import _hash_lines, dedup_and_filter, frozen_split
from fx1.data.traces import TraceRecorder, TraceStep, Trajectory
from fx1.eval.contamination import run_contamination_audit
from fx1.eval.suite import EvalTask, run_suite
from fx1.harness import Harness
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.hypotheses import ResearchTrace, validate_trace_scores
from fx1.modelcard import EvalDelta, ModelCard
from fx1.mrm import compile_dossier

SYSTEM = "system"


def _ex(text: str) -> dict:
    return {
        "messages": [{"role": "user", "content": text}],
        "receipt_sha256": "a" * 64,
        "source_path": "t",
        "negative": False,
    }


# ---------------------------------------------------------------------------
# honesty.py — function-word/reporting-verb phrasing must not bypass the gate
# ---------------------------------------------------------------------------


def test_headline_gate_catches_function_word_phrasing():
    for text in (
        "Sharpe of the strategy is 2.1",
        "pnl for the quarter was $12,000",
        "the fund's nav reached 1.9",
        "calmar ratio for the whole period was 0.8",
        "the strategy's sharpe stood at 2.35 last quarter",
        "expected nav was about 1.9",
    ):
        with pytest.raises(Fx1HonestyError):
            validate_fx1_output(text)


def test_headline_gate_still_allows_discussion():
    for text in (
        "pnl, in the 2024 reporting sense, is not a score",
        "the sharpe family of ratios is excluded by contract",
        "nav at the end of 2024 reporting is not a score",
        "pnl or other measures like 2.0 are discussed but not headlined",
        "the sharpe ratio is a forbidden headline metric here.",
    ):
        assert validate_fx1_output(text) == text


# ---------------------------------------------------------------------------
# eval/__init__ — the full message surface is the decontamination target
# ---------------------------------------------------------------------------


def test_eval_prompt_surface_covers_all_roles():
    import fx1.eval as fx1_eval
    from fx1.eval import DEFAULT_BANK, REDTEAM_TASKS, eval_prompt_surface

    assert "eval_prompt_surface" in fx1_eval.__all__
    surface = set(eval_prompt_surface())
    for task in [*DEFAULT_BANK, *REDTEAM_TASKS]:
        for message in task.messages:
            assert message["content"] in surface


# ---------------------------------------------------------------------------
# data/quality.py — near-dup detection is complete; drop classes counted
# ---------------------------------------------------------------------------


def test_near_duplicate_detected_beyond_old_window():
    filler = [_ex(" ".join(f"tok{i}{j}" for j in range(12))) for i in range(600)]
    # 20-token original (13 shingles); the duplicate appends one token and
    # shares 13/14 shingles -> Jaccard ~0.93, a near-dup the old 500-deep
    # window would have missed at this distance.
    original = _ex(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron pi rho sigma tau upsilon"
    )
    distant_duplicate = _ex(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
        "mu nu xi omicron pi rho sigma tau upsilon phi"
    )
    kept, report = dedup_and_filter([*filler, original, distant_duplicate], eval_prompts=[])
    assert report.near_duplicates_removed == 1
    assert report.kept == 601


def test_split_manifest_hashes_match_written_files(tmp_path: Path):
    examples = [_ex(f"distinct example {i} with enough tokens {i}") for i in range(30)]
    manifest = frozen_split(examples, tmp_path / "corpus", seed=7)
    for name, digest in (("train", manifest.train_sha256), ("val", manifest.val_sha256)):
        written = (tmp_path / f"corpus.{name}.jsonl").read_bytes()
        assert hashlib.sha256(written).hexdigest() == digest


def test_split_line_hashing_is_boundary_sensitive():
    assert _hash_lines(["ab", "c"]) != _hash_lines(["a", "bc"])


def test_empty_and_over_length_examples_are_counted():
    examples = [
        _ex(""),
        _ex("x " * 40_000),
        _ex("normal distinct content here today ok"),
    ]
    kept, report = dedup_and_filter(examples, eval_prompts=[], max_len_chars=1000)
    assert report.empty_removed == 1
    assert report.over_length_removed == 1
    assert report.kept == 1


def test_frozen_split_refuses_a_vacuous_val_split(tmp_path: Path):
    with pytest.raises(ValueError, match="at least 2"):
        frozen_split([_ex("only one")], tmp_path / "corpus")


# ---------------------------------------------------------------------------
# data/receipts.py — the research_only contract is a literal boolean
# ---------------------------------------------------------------------------


def test_truthy_nonboolean_research_only_is_ineligible(tmp_path: Path):
    from fx1.data.receipts import _eligibility

    research_only, live = _eligibility({"research_only": "yes"})
    assert research_only is False
    research_only, _ = _eligibility({"research_only": 1})
    assert research_only is False
    research_only, _ = _eligibility({"research_only": True})
    assert research_only is True


# ---------------------------------------------------------------------------
# data/ledgers.py — any truthy live-claim value is a claim
# ---------------------------------------------------------------------------


def test_live_claim_numeric_and_yes_variants_flagged(tmp_path: Path):
    from fx1.data.ledgers import ledger_examples

    for i, payload in enumerate(
        (
            {"live_pnl_claim": 1, "text": "x"},
            {"livePnlClaim": "yes", "text": "x"},
            {"live-pnl-claim": "recorded", "text": "x"},
        )
    ):
        artifact = tmp_path / f"ledger{i}.json"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        examples = ledger_examples(artifact, SYSTEM)
        assert len(examples) == 1
        assert examples[0].negative is True


# ---------------------------------------------------------------------------
# hypotheses.py — substring evasion + validator wiring
# ---------------------------------------------------------------------------


def test_trace_scores_reject_forbidden_substrings():
    for key in ("crps_realizedpnl", "pinball_navtotal", "brier_drawdownsharpe"):
        with pytest.raises(ValueError, match="forbidden"):
            validate_trace_scores({key: 0.1})
    # A token that is itself an allowed score wins the substring check.
    validate_trace_scores({"sharpness_mean": 0.4, "brier_1m": 0.2})


def test_research_trace_model_enforces_score_contract():
    with pytest.raises(ValidationError):
        ResearchTrace(
            hypothesis="h",
            config_diff="d",
            bench_command="research",
            scores={"pnl_total": 5000.0},
            receipt_sha256="a" * 64,
        )


# ---------------------------------------------------------------------------
# train/run.py — eval gate re-derives violations from every recorded field
# ---------------------------------------------------------------------------


def _manifest_cfg(tmp_path: Path) -> object:
    from fx1.train.config import LadderStage, TrainConfig

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "q"}],
                "receipt_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainConfig(
        run_name="fx-1.v0.1",
        stage=LadderStage.PROXY,
        base_model="Qwen/Qwen3-32B",
        corpus_jsonl=str(corpus),
        eval_results_json=str(tmp_path / "eval.json"),
        estimated_nodes=1,
        estimated_gpu_hours=4.0,
        estimated_cost_usd=50.0,
    )


def test_eval_gate_rejects_summary_level_violations(tmp_path: Path):
    from fx1.train.run import build_training_manifest

    cfg = _manifest_cfg(tmp_path)
    eval_payload = {
        "honesty_gate_passed": True,
        "results": [{"task": "h1", "kind": "honesty", "passed": True, "failures": []}],
        "honesty_violations": ["sneaky_domain_task"],
    }
    (tmp_path / "eval.json").write_text(json.dumps(eval_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="honesty"):
        build_training_manifest(cfg, tmp_path / "m.json")


def test_eval_gate_rejects_result_level_violations(tmp_path: Path):
    from fx1.train.run import build_training_manifest

    cfg = _manifest_cfg(tmp_path)
    eval_payload = {
        "honesty_gate_passed": True,
        "results": [
            {"task": "h1", "kind": "honesty", "passed": True, "failures": []},
            {
                "task": "d1",
                "kind": "domain",
                "passed": True,
                "failures": [],
                "honesty_violations": ["honesty: sharpe headline"],
            },
        ],
    }
    (tmp_path / "eval.json").write_text(json.dumps(eval_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="honesty"):
        build_training_manifest(cfg, tmp_path / "m.json")


# ---------------------------------------------------------------------------
# bench/dip.py — horizon counts bars AFTER the trigger; unmatched = error
# ---------------------------------------------------------------------------


def test_dip_horizon_counts_post_trigger_bars():
    # Recovery exactly at the bars-th bar after the trough-crossing bar.
    closes = [100.0, 102.0, 80.0, 90.0, 90.0, 90.0, 103.0]
    dates = [f"2026-01-0{i + 1}" for i in range(7)]
    events = detect_dip_events(closes, dates, "T", threshold=0.10, horizons_bars={"1m": 4})
    assert len(events) == 1
    assert events[0].recovered["1m"] is True


def test_dip_horizon_unobservable_at_data_end():
    closes = [100.0, 102.0, 80.0, 90.0, 90.0, 90.0, 90.0]
    dates = [f"2026-01-0{i + 1}" for i in range(7)]
    events = detect_dip_events(closes, dates, "T", threshold=0.10, horizons_bars={"1m": 5})
    assert events[0].recovered["1m"] is None


def test_unmatched_forecast_fails_closed():
    closes = [100.0, 102.0, 80.0, 90.0, 103.0, 104.0, 105.0]
    dates = [f"2026-01-0{i + 1}" for i in range(7)]
    events = detect_dip_events(closes, dates, "TEST", threshold=0.10, horizons_bars={"1m": 3})
    with pytest.raises(ValueError, match="outside the frozen"):
        evaluate_forecasts(events, [DipForecast("TEST", "1999-01-01", {"1m": 0.5})])


def test_bench_honesty_substring_evasion_flagged():
    with pytest.raises(ValueError, match="forbidden"):
        assert_bench_output_honest({"brier_realizedpnl": 0.1})
    assert_bench_output_honest({"brier_1m": 0.2, "ece_1m": 0.05})


# ---------------------------------------------------------------------------
# mrm.py — a non-object contamination report is flagged, never trusted
# ---------------------------------------------------------------------------


def _card(tmp_path: Path) -> Path:
    path = tmp_path / "modelcard.json"
    ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b..c",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    ).save(path)
    return path


def test_mrm_nonobject_contamination_report_flags(tmp_path: Path):
    report = tmp_path / "contamination_report.json"
    report.write_text("[1, 2, 3]", encoding="utf-8")
    dossier = compile_dossier(
        modelcard_path=_card(tmp_path),
        artifacts={"contamination_report": report},
        out_path=tmp_path / "d.json",
    )
    assert dossier.contamination_flagged is True
    assert dossier.ship_eligible is False


# ---------------------------------------------------------------------------
# eval/contamination.py — the audit refuses to certify vacuity
# ---------------------------------------------------------------------------


def test_contamination_audit_refuses_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        run_contamination_audit([], ["some prompt"])
    with pytest.raises(ValueError, match="non-empty"):
        run_contamination_audit(["some corpus"], [])


# ---------------------------------------------------------------------------
# data/traces.py — verify_ok without receipts is not a positive example
# ---------------------------------------------------------------------------


def test_verified_claim_without_receipts_demoted(tmp_path: Path):
    traj = Trajectory(
        session_id="s1",
        user_intent="x",
        steps=[TraceStep(assistant_content="clean answer")],
        verify_ok=True,
        artifact_receipts=[],
    )
    rec = TraceRecorder(tmp_path / "t.jsonl")
    assert rec.admit(traj, SYSTEM) is True
    record = json.loads((tmp_path / "t.jsonl").read_text().strip())
    assert record["negative"] is True


# ---------------------------------------------------------------------------
# harness.py — extra_args --config obeys the configs/ containment
# ---------------------------------------------------------------------------


def _fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


def test_extra_args_config_contained(tmp_path: Path, monkeypatch):
    harness = Harness(runner=_fake_runner)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config", "/etc/passwd"])
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("doctor", extra_args=["--config=/etc/passwd"])
    configs = tmp_path / "configs"
    configs.mkdir()
    inside = configs / "ok.yaml"
    inside.write_text("x: 1\n", encoding="utf-8")
    result = harness.run("doctor", extra_args=["--config", "configs/ok.yaml"])
    assert result.ok


# ---------------------------------------------------------------------------
# data/ledger.py — every hash field is pinned to 64 hex chars
# ---------------------------------------------------------------------------


def test_ledger_entry_hash_fields_validated(tmp_path: Path):
    ledger = CorpusLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValidationError):
        ledger.record_example(
            source_sha256="a" * 64,
            transform_sha256="short",
            example_sha256="c" * 64,
        )
    with pytest.raises(ValidationError):
        ledger.record_example(
            source_sha256="a" * 64,
            transform_sha256="b" * 64,
            example_sha256="short",
        )


# ---------------------------------------------------------------------------
# data/notebooks.py — the full chunk is embedded, per the docstring contract
# ---------------------------------------------------------------------------


def test_notebook_embeds_full_chunk(tmp_path: Path):
    from fx1.data.notebooks import notebook_examples

    doc = tmp_path / "note.md"
    section = "# Section\n" + ("word " * 600).strip()  # > 1500 chars, < 4000
    doc.write_text(section, encoding="utf-8")
    examples = notebook_examples(doc, SYSTEM)
    user = examples[0].messages[1]["content"]
    assert user.count("word") == 600


# ---------------------------------------------------------------------------
# eval/suite.py — task authoring bugs are construction-time errors
# ---------------------------------------------------------------------------


def test_eval_task_rejects_blank_required_token():
    with pytest.raises(ValidationError, match="non-blank"):
        EvalTask(
            name="t",
            kind="domain",
            messages=[{"role": "user", "content": "q"}],
            required_tokens=[""],
        )


def test_eval_task_rejects_uncompilable_pattern():
    with pytest.raises(ValidationError, match="compile"):
        EvalTask(
            name="t",
            kind="domain",
            messages=[{"role": "user", "content": "q"}],
            forbidden_patterns=["([unclosed"],
        )


def test_run_suite_rejects_duplicate_task_names():
    task = EvalTask(
        name="dup",
        kind="domain",
        messages=[{"role": "user", "content": "q"}],
    )
    with pytest.raises(ValueError, match="duplicate"):
        run_suite(lambda msgs: "ok", [task, task])


# ---------------------------------------------------------------------------
# serve/backends.py — a null content field is malformed, not "None"
# ---------------------------------------------------------------------------


def test_hosted_backend_rejects_null_content(monkeypatch):
    from fx1.serve.backends import HostedK3Backend

    class _Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": None}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import fx1.serve.backends as _be

    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    monkeypatch.setattr(_be, "_openai_urlopen", lambda *a, **k: _Resp())
    backend = HostedK3Backend()
    with pytest.raises(RuntimeError, match="malformed"):
        backend.complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# data/sources/base.py — as_of must be a real YYYY-MM-DD date
# ---------------------------------------------------------------------------


def test_fetch_request_as_of_must_be_a_date():
    from fx1.data.sources.base import FetchRequest

    FetchRequest(api="x", as_of="2026-09-24")
    for bad in ("banana", "2026-9-4", "2026-02-30", "next tuesday"):
        with pytest.raises(ValidationError):
            FetchRequest(api="x", as_of=bad)


# ---------------------------------------------------------------------------
# train/pipeline.py — eval_base is re-verified against the receipt pin
# ---------------------------------------------------------------------------


def _pipeline(tmp_path: Path):
    from fx1.train.config import LadderStage, TrainConfig
    from fx1.train.pipeline import Pipeline

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps(_ex(f"unique training content {i} tokens here")) for i in range(30))
        + "\n",
        encoding="utf-8",
    )
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps({"honesty_gate_passed": True, "results": []}))
    cfg = TrainConfig(
        run_name="fx-1.v0.1",
        stage=LadderStage.PROXY,
        base_model="Qwen/Qwen3-32B",
        corpus_jsonl=str(corpus),
        eval_results_json=str(eval_json),
        estimated_nodes=1,
        estimated_gpu_hours=4.0,
        estimated_cost_usd=50.0,
    )
    trainer = lambda tr, va, c, wd: wd / "ckpt"  # noqa: E731
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "ckpt").mkdir(exist_ok=True)
    return Pipeline(cfg, tmp_path / "work", trainer=trainer)


def _good_model(msgs: list[dict[str, str]]) -> str:
    return (
        "proper scores: crps pinball pit qlike kupiec vpin kyle "
        "walk-forward cpcv almgren next-open; no live claims; "
        "SYNTHETIC labeled; fail-closed; I cannot guarantee that; 391 data; "
        "backtest evidence, not live; net of cost; survivorship bias noted; "
        "multiple testing controlled; conformal coverage; spread and impact "
        "charged; ifind wind lead CN routing; as_of blocks leakage; "
        "sha256 receipt provenance; dip recovery scored with brier"
    )


def _run_to_candidate(pipe) -> Path:
    pipe.run_quality_gate()
    pipe.run_eval_base(_good_model)
    pipe.run_training(seed=17)
    return Path(pipe.state.artifacts["eval_base"])


def test_eval_candidate_rejects_tampered_base(tmp_path: Path):
    pipe = _pipeline(tmp_path)
    base_path = _run_to_candidate(pipe)
    summary = json.loads(base_path.read_text(encoding="utf-8"))
    summary["results"][0]["passed"] = not summary["results"][0]["passed"]
    base_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash pinned|training receipt"):
        pipe.run_eval_candidate(_good_model)


def test_eval_candidate_rejects_mismatched_task_sets(tmp_path: Path):
    from fx1.train.receipts import loads_receipt

    pipe = _pipeline(tmp_path)
    base_path = _run_to_candidate(pipe)
    # Rewrite eval_base with one domain task dropped, then re-issue the
    # receipt pin so ONLY the task-set check can catch it.
    summary = json.loads(base_path.read_text(encoding="utf-8"))
    domain = next(r for r in summary["results"] if r["kind"] == "domain")
    summary["results"].remove(domain)
    base_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt_path = Path(pipe.state.artifacts["training_receipt"])
    receipt = loads_receipt(receipt_path)
    receipt.eval_base_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different domain tasks"):
        pipe.run_eval_candidate(_good_model)


def test_eval_candidate_rejects_foreign_eval_bank(tmp_path: Path):
    from fx1.train.receipts import loads_receipt

    pipe = _pipeline(tmp_path)
    base_path = _run_to_candidate(pipe)
    summary = json.loads(base_path.read_text(encoding="utf-8"))
    summary["eval_bank_sha256"] = "0" * 64
    base_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt_path = Path(pipe.state.artifacts["training_receipt"])
    receipt = loads_receipt(receipt_path)
    receipt.eval_base_sha256 = hashlib.sha256(base_path.read_bytes()).hexdigest()
    receipt_path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different eval banks"):
        pipe.run_eval_candidate(_good_model)
