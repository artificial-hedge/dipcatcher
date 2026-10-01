"""Tests for compute/provenance.py — replay manifests for parallel sweeps."""

from __future__ import annotations

import pytest

from quant_fund.compute.provenance import provenance_map, replay_task, verify_sweep


def _task(item: list[float], seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    return {"mean": float(np.mean(item)), "draw": float(rng.random())}


def _items() -> list[list[float]]:
    return [[float(i), float(i) + 1.0] for i in range(6)]


def test_provenance_map_pins_every_task():
    results, manifest = provenance_map(_task, _items(), base_seed=42, sweep="demo")
    assert len(results) == 6
    tasks = manifest["claim"]["tasks"]
    assert len(tasks) == 6
    assert all(len(t["result_sha256"]) == 64 for t in tasks)
    assert manifest["receipt_sha256"]


def test_determinism_same_manifest():
    _, m1 = provenance_map(_task, _items(), base_seed=7, sweep="demo")
    _, m2 = provenance_map(_task, _items(), base_seed=7, sweep="demo")
    assert m1["receipt_sha256"] == m2["receipt_sha256"]


def test_replay_task_matches():
    _, manifest = provenance_map(_task, _items(), base_seed=11, sweep="demo")
    rep = replay_task(_task, _items(), manifest, 3)
    assert rep["replayable"] and rep["reason"] == "match"


def test_replay_detects_result_drift():
    _, manifest = provenance_map(_task, _items(), base_seed=11, sweep="demo")

    def drifted(item: list[float], seed: int) -> dict:
        out = _task(item, seed)
        out["mean"] += 1.0
        return out

    rep = replay_task(drifted, _items(), manifest, 0)
    assert not rep["replayable"] and rep["reason"] == "result digest mismatch"


def test_verify_sweep_clean_and_dirty():
    _, manifest = provenance_map(_task, _items(), base_seed=3, sweep="demo")
    assert verify_sweep(_task, _items(), manifest)["ok"]

    def broken(item: list[float], seed: int) -> dict:
        out = _task(item, seed)
        if seed % 2 == 0:
            out["draw"] = 0.0
        return out

    rep = verify_sweep(broken, _items(), manifest)
    assert not rep["ok"] and rep["n_failed"] >= 1


def test_nonserializable_results_rejected():
    def bad(item: object, seed: int) -> object:
        return object()

    with pytest.raises(ValueError, match="non-JSON"):
        provenance_map(bad, [1, 2, 3], base_seed=0, sweep="x")


def test_replay_rejects_bad_index_and_manifest():
    _, manifest = provenance_map(_task, _items(), base_seed=5, sweep="demo")
    with pytest.raises(ValueError, match="outside"):
        replay_task(_task, _items(), manifest, 99)
    with pytest.raises(ValueError, match="manifest"):
        verify_sweep(_task, _items(), {"claim": {}})
