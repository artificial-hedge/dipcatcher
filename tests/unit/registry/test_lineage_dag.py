"""Tests for quant_fund.registry.lineage_dag."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.registry.lineage_dag import (
    collect_receipt_refs,
    lineage_dag,
    lineage_dag_bench,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _receipt(path: Path, kind: str, digest: str | None = None, **extra) -> str:
    payload = {"kind": kind, "schema": f"{kind}.v1", **extra}
    payload["receipt_sha256"] = digest or hash_bytes(canonical_json_bytes(payload))
    # reseal after adding the digest field only if caller planted one
    if digest:
        payload["receipt_sha256"] = digest
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload["receipt_sha256"]


def test_collect_refs_harvests_digests_and_files(tmp_path: Path):
    p = tmp_path / "a.json"
    own = _receipt(
        p,
        "a",
        inputs_sha256="ab" * 32,
        dataset_hash="cd" * 32,
        claim={"inputs": ["data/raw/bars.parquet", "notes"]},
    )
    refs = collect_receipt_refs(p)
    kinds = {r["kind"] for r in refs["refs"]}
    assert refs["digest"] == own
    assert "digest" in kinds and "file" in kinds
    # own seal is never an input
    assert all(r["value"] != own for r in refs["refs"])


def test_dag_receipt_to_receipt_edge(tmp_path: Path):
    a = _receipt(tmp_path / "a.json", "producer")
    b = _receipt(tmp_path / "b.json", "consumer", inputs_sha256=a)
    dag = lineage_dag(tmp_path)
    assert dag["receipt_edges"] == 1
    assert dag["n_receipts"] == 2
    assert dag["cycles"] == []
    # a is consumed; b is a sink
    assert b in dag["sinks"]


def test_dag_flags_cycle(tmp_path: Path):
    a = _receipt(tmp_path / "a.json", "a", digest="aa" * 32, inputs_sha256="bb" * 32)
    _receipt(tmp_path / "b.json", "b", digest="bb" * 32, inputs_sha256=a)
    dag = lineage_dag(tmp_path)
    assert dag["cycles"] or dag["verdict"] != "provenance_complete"


def test_dangling_file_ref(tmp_path: Path):
    _receipt(
        tmp_path / "r.json",
        "r",
        claim={"inputs": ["data/does_not_exist.parquet"]},
    )
    dag = lineage_dag(tmp_path)
    assert dag["dangling_inputs"]
    assert dag["verdict"] == "open"


def test_orphans_and_external(tmp_path: Path):
    _receipt(tmp_path / "x.json", "x")
    _receipt(tmp_path / "y.json", "y", drill={"shard": {"source": "yahoo_eod", "symbol": "NVDA"}})
    dag = lineage_dag(tmp_path)
    assert any(e.startswith("shard:yahoo_eod") for e in dag["external_inputs"])
    assert len(dag["orphan_receipts"]) == 1  # x has no edges


def test_bench_sealed(tmp_path: Path):
    _receipt(tmp_path / "p.json", "producer")
    _receipt(tmp_path / "c.json", "consumer", inputs_sha256="ee" * 32)
    r1 = lineage_dag_bench(tmp_path)
    r2 = lineage_dag_bench(tmp_path)
    assert r1["schema"] == "lineage_dag.v1"
    assert r1["claim"]["verdict"] in {"provenance_complete", "open"}
    # strip git_revision drift before determinism check
    r1.pop("git_revision")
    r2.pop("git_revision")
    assert r1 == r2
