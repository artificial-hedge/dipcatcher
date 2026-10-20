"""Real-registry corpus build must drop verifier-rejected receipts.

``build_corpus`` without an explicit repo root used to skip the quarantine
registry (it imported a module that does not exist). Two of the nine
positive examples, ``basis_carry_dd7705fc0f2f1c25.json`` and
``cost_aware_rerun_20261007.json``, stayed in the output. The assertion
``sha not in emitted`` fails on that behavior. The receipt files stay.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.data.corpus import build_corpus
from fx1.train import LadderStage, TrainConfig, build_training_manifest

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "quality" / "legacy_quarantine.json"

_QUARANTINED_POSITIVES = (
    "basis_carry_dd7705fc0f2f1c25.json",
    "cost_aware_rerun_20261007.json",
    "legacy-unsealed/adaptive_mix_20asset_1d_20260922.json",
    "legacy-unsealed/adaptive_mix_band_search_20asset_1d_20260922.json",
    "legacy-unsealed/basis_pair_candidate_20asset_1d_20260922.json",
    "legacy-unsealed/basis_reversion_screen_20asset_1d_20260922.json",
    "legacy-unsealed/dip_bench_crypto_1d_20260925.json",
    "legacy-unsealed/fast_replay_p42_conformance_20260927.json",
    "legacy-unsealed/incumbent_bench_qlib.json",
)


def test_real_registry_excludes_quarantined_positives(tmp_path: Path) -> None:
    registry = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    files = registry["files"]
    assert len(_QUARANTINED_POSITIVES) == 9
    forbidden = [
        rel for rel in _QUARANTINED_POSITIVES if "forbidden_metric_keys" in files[rel]["errors"]
    ]
    assert len(forbidden) == 4

    out = tmp_path / "corpus.jsonl"
    build_corpus(_REPO / "receipts", out)
    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    emitted = {row["receipt_sha256"] for row in rows}
    paths = {row["source_path"] for row in rows}
    for rel in _QUARANTINED_POSITIVES:
        assert files[rel]["sha256"] not in emitted, rel
        assert not any(path.endswith(rel) for path in paths), rel
    for rel, entry in files.items():
        assert entry["sha256"] not in emitted, rel


def test_training_manifest_refuses_quarantined_source(tmp_path: Path) -> None:
    source = _REPO / "receipts" / _QUARANTINED_POSITIVES[0]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "q"}],
                "receipt_sha256": "a" * 64,
                "source_path": str(source),
                "negative": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ev = tmp_path / "eval.json"
    ev.write_text(
        json.dumps(
            {
                "honesty_gate_passed": True,
                "results": [
                    {
                        "task": "bait-0",
                        "kind": "honesty",
                        "passed": True,
                        "response": "r",
                        "failures": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = TrainConfig(
        run_name="fx-1.v0.1",
        stage=LadderStage.PROXY,
        corpus_jsonl=str(corpus),
        eval_results_json=str(ev),
        estimated_nodes=1,
        estimated_gpu_hours=1.0,
        estimated_cost_usd=1.0,
    )
    with pytest.raises(ValueError, match="not training evidence"):
        build_training_manifest(cfg, tmp_path / "manifest.json")
