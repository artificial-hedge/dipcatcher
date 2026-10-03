"""infra_audit — adversarial probes on train config/curriculum/cluster.

Pinned contract:

- ``TrainConfig``: ``FINAL_K3`` requires the K3 base, ≥2 nodes, and
  preserved reasoning content — each refusal raises ValueError;
  epochs/lr/lora bounds enforced by pydantic.
- ``build_curriculum``: levels emit in canonical order
  (CONTRACTS→REFUSAL); within-level order is deterministic under the
  seed AND invariant to input ordering (the sha256 pre-sort canonicalizes
  before the seeded shuffle).
- ``ClusterSpec``: nodes ≥ 2 by physics; mxfp4 requires ZeRO-3; ≥128k
  context requires ≥8 nodes; deepspeed export carries the same seed.

Sealed ``train_infra_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["train_infra_audit", "train_infra_audit_bench"]


def _cfg(**kw: Any) -> Any:
    from fx1.train.config import LadderStage, TrainConfig

    base: dict[str, Any] = {
        "run_name": "x",
        "stage": LadderStage.PROXY,
        "corpus_jsonl": "c.jsonl",
        "eval_results_json": "e.json",
        "estimated_nodes": 1,
        "estimated_gpu_hours": 0.5,
        "estimated_cost_usd": 0.0,
    }
    base.update(kw)
    return TrainConfig.model_validate(base)


def train_infra_audit() -> dict[str, Any]:
    from fx1.train.cluster import ClusterSpec
    from fx1.train.config import LadderStage
    from fx1.train.curriculum import Level, build_curriculum, classify

    out: dict[str, Any] = {}

    # TrainConfig K3 disclosure gates
    k3_bad_model = _raises(
        lambda: _cfg(stage=LadderStage.FINAL_K3, base_model="other/m", estimated_nodes=4)
    )
    k3_few_nodes = _raises(lambda: _cfg(stage=LadderStage.FINAL_K3, estimated_nodes=1))
    k3_no_preserve = _raises(
        lambda: _cfg(
            stage=LadderStage.FINAL_K3,
            estimated_nodes=4,
            preserve_reasoning_content=False,
        )
    )
    out["k3_gates"] = (
        k3_bad_model.endswith("Error")
        and k3_few_nodes.endswith("Error")
        and k3_no_preserve.endswith("Error")
    )
    out["k3_valid_ok"] = (
        _cfg(stage=LadderStage.FINAL_K3, estimated_nodes=4).stage == LadderStage.FINAL_K3
    )
    out["bounds_enforced"] = all(
        _raises(lambda b=b: _cfg(**b)) == "ValidationError"
        for b in (
            {"epochs": 0},
            {"epochs": 11},
            {"learning_rate": 0},
            {"estimated_nodes": 0},
            {"estimated_gpu_hours": 0},
        )
    )

    # curriculum: ordering + determinism + input-order invariance
    contract_ex = {"messages": [{"role": "assistant", "content": "rules"}]}
    interp_ex = {"messages": [{"role": "assistant", "content": "use verify-research"}]}
    loop_ex = {"messages": [{"role": "assistant", "content": "<tool_call>x</tool_call>"}]}
    refusal_ex = {"negative": True, "messages": []}
    corpus = [loop_ex, refusal_ex, contract_ex, interp_ex]
    out["classify_order"] = [
        classify(contract_ex),
        classify(interp_ex),
        classify(loop_ex),
        classify(refusal_ex),
    ] == [Level.CONTRACTS, Level.INTERPRETATION, Level.RESEARCH_LOOP, Level.REFUSAL]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in corpus) + "\n")
        o1, o2 = Path(tmp) / "o1.jsonl", Path(tmp) / "o2.jsonl"
        c1 = build_curriculum(p, o1, seed=7)
        # reversed input order → identical output order (canonical sort)
        p_rev = Path(tmp) / "cr.jsonl"
        p_rev.write_text("\n".join(json.dumps(e) for e in reversed(corpus)) + "\n")
        c2 = build_curriculum(p_rev, o2, seed=7)
        out["curriculum_deterministic"] = o1.read_text() == o2.read_text()
        out["level_counts"] = (
            c1
            == c2
            == {
                "contracts": 1,
                "interpretation": 1,
                "research_loop": 1,
                "refusal": 1,
            }
        )
        ordered = [json.loads(ln) for ln in o1.read_text().splitlines()]
        out["refusal_last"] = ordered[-1].get("negative") is True

    # cluster spec sanity
    out["nodes_min2"] = _raises(lambda: ClusterSpec(run_name="x", nodes=1)) == "ValidationError"
    out["mxfp4_needs_zero3"] = _raises(
        lambda: ClusterSpec(run_name="x", nodes=4, precision="mxfp4", zero_stage=2)
    ).endswith("Error")
    out["long_ctx_needs_8"] = _raises(
        lambda: ClusterSpec(run_name="x", nodes=4, sequence_length=131_072)
    ).endswith("Error")
    spec = ClusterSpec(run_name="x", nodes=4, precision="fp8", seed=7)
    out["world_size"] = spec.world_size == 32
    ds = spec.to_deepspeed_config()
    out["deepspeed_carries_seed"] = ds["seed"] == 7 and ds["zero_optimization"]["stage"] == 3
    return out


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def train_infra_audit_bench() -> dict[str, Any]:
    r = train_infra_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "train_infra_audit",
        "schema": "train_infra_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Train-infra contract holds: K3 disclosure gates refuse "
            "wrong-model/few-node/no-preserve configs; numeric bounds "
            "enforced; curriculum is level-ordered and input-order "
            "invariant under a fixed seed; cluster physics enforced."
            if ok
            else f"TRAIN INFRA AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
