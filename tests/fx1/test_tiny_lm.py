"""KATs for the real tiny-LM trainer (fx1.train.tiny_lm).

The trainer is the mechanism the ft surface runs: corpus text → a real
safetensors checkpoint that ``LocalWeightsEngine`` can load and serve.
These tests pin the honest pieces — input floors, determinism, the
cancel/interrupt contract, and the manifest sha pin.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="nn extra not installed")
pytest.importorskip("safetensors.numpy", reason="nn extra not installed")

from fx1.serve.local_engine import LocalWeightsEngine  # noqa: E402
from fx1.train.tiny_lm import (  # noqa: E402
    DEFAULT_CTX,
    DEFAULT_DIM,
    DEFAULT_N_HEADS,
    MAX_FT_STEPS,
    MIN_FT_STEPS,
    VOCAB_SIZE,
    build_tiny_lm,
    chat_jsonl_to_text,
    make_tiny_lm_trainer,
    plan_steps,
    save_weights,
    sha256_file,
    train_tiny_lm,
    val_loss_of,
)

_ROOT = Path(__file__).resolve().parents[2]
_BASE_CKPT = _ROOT / "artifacts" / "fx1_tiny_lm"

_TEXT = "the journal replays every record; non-terminal means failed. " * 20


def test_plan_steps_clamps():
    assert plan_steps(1, 1) == MIN_FT_STEPS
    assert plan_steps(1_000_000, 2) == MAX_FT_STEPS
    mid = plan_steps(DEFAULT_CTX * 200, 1)
    assert MIN_FT_STEPS <= mid <= MAX_FT_STEPS
    # epochs multiply the token budget: 2 epochs ≈ 2x a single pass
    assert plan_steps(DEFAULT_CTX * 200, 2) == min(MAX_FT_STEPS, 2 * mid)


def test_chat_jsonl_to_text_floors(tmp_path: Path):
    good = tmp_path / "good.jsonl"
    body = (
        b'{"messages":[{"role":"user","content":"'
        + b"q" * 400
        + b'"},{"role":"assistant","content":"'
        + b"a" * 400
        + b'"}]}\n'
    )
    good.write_bytes(body)
    text = chat_jsonl_to_text(good)
    assert text.strip()

    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"not json\n")
    with pytest.raises(ValueError):
        chat_jsonl_to_text(bad)
    thin = tmp_path / "thin.jsonl"
    thin.write_bytes(b'{"messages":[{"role":"user","content":"q"}]}\n')
    with pytest.raises(ValueError, match="bytes"):
        chat_jsonl_to_text(thin)


def test_train_tiny_lm_deterministic_same_seed(tmp_path: Path):
    model_a, loss_a, tokens_a, val_a = train_tiny_lm(_TEXT, steps=32, lr=1e-4, seed=7)
    model_b, loss_b, tokens_b, val_b = train_tiny_lm(_TEXT, steps=32, lr=1e-4, seed=7)
    assert tokens_a == tokens_b == 32 * DEFAULT_CTX
    assert loss_a == loss_b and val_a == val_b
    # same seed → byte-identical weights
    sa, _ = save_weights(tmp_path / "a", model_a)
    sb, _ = save_weights(tmp_path / "b", model_b)
    assert sa == sb
    # a different seed trains differently — the seed is real, not cosmetic
    _, loss_c, _, _ = train_tiny_lm(_TEXT, steps=32, lr=1e-4, seed=8)
    assert loss_c != loss_a


def test_train_tiny_lm_interrupt_is_honest():
    calls = {"n": 0}

    def stop_early() -> bool:
        calls["n"] += 1
        return calls["n"] < 5

    with pytest.raises(RuntimeError, match="interrupted"):
        train_tiny_lm(_TEXT, steps=64, lr=1e-4, seed=0, should_continue=stop_early)


def test_trainer_fn_writes_loadable_checkpoint(tmp_path: Path):
    """The TrainerFn the pipeline calls produces a real checkpoint dir: the
    manifest sha pins the weights bytes, the card carries the eval-pending
    sentinel (ship gate fails closed until the job stamps the measured
    eval pair), and LocalWeightsEngine loads it on the measurement path."""
    work = tmp_path / "job"
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(b'{"messages":[{"role":"user","content":"' + b"q" * 600 + b'"}]}\n')
    fn = make_tiny_lm_trainer(
        seed=0,
        fine_tuned_model="ft:fx1:test:abc",
        job_id="ftjob-x",
        corpus_label="corpus.jsonl",
        base_checkpoint=_BASE_CKPT,
        base_weights_sha256=sha256_file(_BASE_CKPT / "weights.safetensors"),
        lr=1e-4,
        val_path=tmp_path / "val-missing.jsonl",
        should_continue=lambda: True,
    )
    from fx1.train.config import LadderStage, TrainConfig

    config = TrainConfig(
        run_name="ft:fx1:test:abc",
        stage=LadderStage.PROXY,
        corpus_jsonl=str(corpus),
        eval_results_json=str(work / "eval_base.json"),
        epochs=1,
        learning_rate=1e-4,
        estimated_nodes=1,
        estimated_gpu_hours=0.1,
        estimated_cost_usd=0.0,
    )
    out = fn(corpus, tmp_path / "val.jsonl", config, work)
    ckpt = Path(out)
    assert ckpt == work / "checkpoint"
    manifest = json.loads((ckpt / "weights.manifest.json").read_text())
    assert manifest["weights_sha256"] == sha256_file(ckpt / "weights.safetensors")
    assert manifest["training"]["fine_tuned_model"] == "ft:fx1:test:abc"
    assert manifest["training"]["trained_tokens"] == MIN_FT_STEPS * DEFAULT_CTX
    card = json.loads((ckpt / "modelcard.json").read_text())
    assert card["version"] == "fx-1.v0.1"
    # eval-pending sentinel: all-zero rates + honesty_gate_candidate=False
    # fail the ship gate closed — the checkpoint is never servable until
    # the job's real measured eval pair is stamped via
    # stamp_measured_eval_delta
    assert card["eval_delta"]["honesty_gate_candidate"] is False
    assert not card["eval_delta"]["domain_pass_rate_candidate"]
    assert not card["eval_delta"]["general_pass_rate_base"]
    assert any(lim.startswith("eval_delta pending") for lim in card["known_limits"])
    with pytest.raises(RuntimeError, match="ship gate"):
        LocalWeightsEngine(ckpt)
    engine = LocalWeightsEngine(ckpt, enforce_ship_gate=False)
    assert engine.complete_messages([{"role": "user", "content": "hi"}], max_tokens=8).text


def _trained_ckpt(tmp_path: Path) -> Path:
    """One real trained checkpoint dir carrying the eval-pending card."""
    work = tmp_path / "job"
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(b'{"messages":[{"role":"user","content":"' + b"q" * 600 + b'"}]}\n')
    fn = make_tiny_lm_trainer(
        seed=0,
        fine_tuned_model="ft:fx1:test:stamp",
        job_id="ftjob-stamp",
        corpus_label="corpus.jsonl",
        base_checkpoint=_BASE_CKPT,
        base_weights_sha256=sha256_file(_BASE_CKPT / "weights.safetensors"),
        lr=1e-4,
        val_path=tmp_path / "val-missing.jsonl",
        should_continue=lambda: True,
    )
    from fx1.train.config import LadderStage, TrainConfig

    config = TrainConfig(
        run_name="ft:fx1:test:stamp",
        stage=LadderStage.PROXY,
        corpus_jsonl=str(corpus),
        eval_results_json=str(work / "eval_base.json"),
        epochs=1,
        learning_rate=1e-4,
        estimated_nodes=1,
        estimated_gpu_hours=0.1,
        estimated_cost_usd=0.0,
    )
    return Path(fn(corpus, tmp_path / "val.jsonl", config, work))


def test_stamp_measured_eval_delta_regresses_and_refuses_restamping(tmp_path: Path):
    """A candidate that measured worse carries an honest negative delta —
    the gate's verdict reflects reality; and stamping over anything that
    is not the pending sentinel refuses, so measured evidence is never
    silently overwritten."""
    from fx1.modelcard import EvalDelta, ModelCard
    from fx1.train.tiny_lm import stamp_measured_eval_delta

    ckpt = _trained_ckpt(tmp_path)
    worse = EvalDelta(
        domain_pass_rate_base=0.5,
        domain_pass_rate_candidate=0.25,
        general_pass_rate_base=0.9,
        general_pass_rate_candidate=0.5,
        honesty_gate_candidate=False,
    )
    stamp_measured_eval_delta(ckpt, worse, evidence_note="eval_delta measured by test harness")
    card = ModelCard.load(ckpt / "modelcard.json")
    assert card.eval_delta.domain_pass_rate_candidate < card.eval_delta.domain_pass_rate_base
    assert card.eval_delta.ship_eligible is False
    assert card.known_limits[-1].startswith("eval_delta measured by test harness")
    with pytest.raises(RuntimeError, match="eval-pending sentinel"):
        stamp_measured_eval_delta(ckpt, worse, evidence_note="restamp attempt")


def test_stamp_measured_eval_delta_eligible_serves(tmp_path: Path):
    """A measured improvement + honesty pass stamps ship-eligible — the
    gate judges the produced card like any other and the engine serves."""
    from fx1.modelcard import EvalDelta, ModelCard
    from fx1.train.tiny_lm import stamp_measured_eval_delta

    ckpt = _trained_ckpt(tmp_path)
    better = EvalDelta(
        domain_pass_rate_base=0.5,
        domain_pass_rate_candidate=0.75,
        general_pass_rate_base=0.9,
        general_pass_rate_candidate=0.95,
        honesty_gate_candidate=True,
    )
    stamp_measured_eval_delta(ckpt, better, evidence_note="eval_delta measured by test harness")
    card = ModelCard.load(ckpt / "modelcard.json")
    assert card.eval_delta.ship_eligible
    engine = LocalWeightsEngine(ckpt)
    assert engine.complete_messages([{"role": "user", "content": "hi"}], max_tokens=8).text


def test_stamp_measured_eval_delta_refuses_fixture_placeholder(tmp_path: Path):
    """The labeled fixture placeholder is existing evidence too — stamping
    over it would hide the placeholder convention, so it refuses."""
    import shutil

    from fx1.modelcard import EvalDelta
    from fx1.train.tiny_lm import stamp_measured_eval_delta

    fixture = tmp_path / "fixture"
    shutil.copytree(_BASE_CKPT, fixture)
    delta = EvalDelta(
        domain_pass_rate_base=0.5,
        domain_pass_rate_candidate=0.75,
        general_pass_rate_base=0.9,
        general_pass_rate_candidate=0.95,
        honesty_gate_candidate=True,
    )
    with pytest.raises(RuntimeError, match="eval-pending sentinel"):
        stamp_measured_eval_delta(fixture, delta, evidence_note="restamp attempt")


def test_val_loss_of_deterministic():
    model = build_tiny_lm(VOCAB_SIZE, DEFAULT_DIM, DEFAULT_CTX, DEFAULT_N_HEADS)
    a = val_loss_of(model, _TEXT, seed=0)
    b = val_loss_of(model, _TEXT, seed=0)
    assert a is not None and a == b and a > 0
    assert val_loss_of(model, "", seed=0) is None
