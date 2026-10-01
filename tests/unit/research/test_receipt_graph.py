"""Tests for the provenance citation-graph audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_fund.research.receipt_graph import (
    GRAPH_SCHEMA,
    graph_contract_errors,
    receipt_graph,
    write_graph_receipt,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_file

FIXTURE_KIND = "synthetic_fixture.v1"


def _write(root: Path, name: str, doc: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(doc))
    return path


def _receipt(extra: dict | None = None, *, kind: str = FIXTURE_KIND) -> dict:
    body = {
        "kind": kind,
        "schema": kind,
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "inputs_sha256": "0" * 64,
    }
    if extra:
        body.update(extra)
    return body


def _sealed(extra: dict | None = None, *, kind: str = FIXTURE_KIND) -> dict:
    return seal_receipt(_receipt(extra, kind=kind))


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_dir_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        receipt_graph(tmp_path / "nope")


def test_empty_corpus_is_clean(tmp_path: Path) -> None:
    out = receipt_graph(tmp_path)
    assert out["schema"] == GRAPH_SCHEMA
    assert out["verdict"] == "clean"
    assert out["n_members"] == 0
    assert out["n_edges"] == 0
    assert graph_contract_errors(out) == []


def test_single_self_contained_receipt_is_orphan(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed())
    out = receipt_graph(tmp_path)
    assert out["n_members"] == 1
    # The member's own seal is skipped; inputs_sha256 resolves to nothing.
    assert all(e["target"] is None for e in out["edges"])
    assert out["orphans"] == ["a.json"]
    assert out["verdict"] == "clean"
    assert graph_contract_errors(out) == []


def test_two_receipt_chain_prev_link_resolves(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.json", _sealed())
    b = _sealed(
        {
            "prev_epoch_sha256": json.loads(a.read_text())["receipt_sha256"],
            "prev_epoch_receipt": "a.json",
        }
    )
    _write(tmp_path, "b.json", b)
    out = receipt_graph(tmp_path)
    assert out["verdict"] == "clean"
    digest_edge = next(e for e in out["edges"] if e["field_path"] == "prev_epoch_sha256")
    assert digest_edge["file"] == "b.json"
    assert digest_edge["target"] == "a.json"
    assert digest_edge["target_via"] == "receipt_sha256"
    name_edge = next(e for e in out["edges"] if e["field_path"] == "prev_epoch_receipt")
    assert name_edge["ref"] == "filename"
    assert name_edge["target"] == "a.json"
    assert out["orphans"] == []
    assert out["n_dangling"] == 0
    assert graph_contract_errors(out) == []


def test_forged_dangling_prev_reference_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed())
    _write(tmp_path, "b.json", _sealed({"prev_epoch_sha256": "f" * 64}))
    out = receipt_graph(tmp_path)
    assert out["verdict"] == "dangling"
    assert out["n_dangling"] == 1
    assert out["dangling"][0]["file"] == "b.json"
    assert out["dangling"][0]["field_path"] == "prev_epoch_sha256"
    assert graph_contract_errors(out) == []


def test_non_membership_digest_is_not_dangling(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed({"inputs_sha256": "f" * 64}))
    out = receipt_graph(tmp_path)
    assert out["n_dangling"] == 0
    assert out["verdict"] == "clean"
    edge = next(e for e in out["edges"] if e["field_path"] == "inputs_sha256")
    assert edge["target"] is None


def test_file_bytes_digest_resolves(tmp_path: Path) -> None:
    b = _write(tmp_path, "b.json", _sealed())
    a = _sealed({"files": [{"file": "b.json", "file_sha256": _file_digest(b)}]})
    _write(tmp_path, "a.json", a)
    out = receipt_graph(tmp_path)
    edge = next(e for e in out["edges"] if e["field_path"] == "files[0].file_sha256")
    assert edge["target"] == "b.json"
    assert edge["target_via"] == "file_bytes"


def test_filename_cycle_detected(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed({"prev_epoch_receipt": "b.json"}))
    _write(tmp_path, "b.json", _sealed({"prev_epoch_receipt": "a.json"}))
    out = receipt_graph(tmp_path)
    assert out["verdict"] == "cyclic"
    assert out["n_cycles"] == 1
    assert out["cycles"][0]["members"] == ["a.json", "b.json"]
    assert graph_contract_errors(out) == []


def test_unresolvable_receipt_filename_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed({"prev_epoch_receipt": "ghost.json"}))
    out = receipt_graph(tmp_path)
    assert out["verdict"] == "dangling"
    assert out["n_unresolvable"] == 1
    assert out["unresolvable"][0]["value"] == "ghost.json"


def test_orphan_classification_and_epoch_adjacent_exempt(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed())
    _write(tmp_path, "b.json", _sealed({"prev_epoch_receipt": "a.json"}))
    _write(tmp_path, "c.json", _sealed())
    _write(tmp_path, "meta.json", _sealed(kind="receipt_lattice.v1"))
    out = receipt_graph(tmp_path)
    assert out["orphans"] == ["c.json"]  # meta kind exempt, a/b chained
    assert graph_contract_errors(out) == []


def test_parse_error_downgrades_verdict(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed())
    (tmp_path / "bad.json").write_bytes(b"{not json")
    out = receipt_graph(tmp_path)
    assert out["verdict"] == "partially_unreadable"
    assert out["n_parse_errors"] == 1
    assert "bad.json" in out["members"]  # still an edge target


def test_v2_envelope_member_walks_payload(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed())
    inner = _receipt({"prev_epoch_receipt": "a.json"})
    env = {
        "schema": "receipt.v2",
        "kind": FIXTURE_KIND,
        "payload": inner,
        "code_files": {"fixture.py": "1" * 64},
        "dataset_hash": "2" * 64,
        "params_hash": "3" * 64,
        "environment": {},
        "receipt_sha256": "4" * 64,
    }
    _write(tmp_path, "e.json", env)
    out = receipt_graph(tmp_path)
    edge = next(e for e in out["edges"] if e["field_path"] == "payload.prev_epoch_receipt")
    assert edge["target"] == "a.json"
    assert out["orphans"] == []


def test_written_receipt_verifies_and_contract_clean(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _sealed())
    _write(corpus, "b.json", _sealed({"prev_epoch_receipt": "a.json"}))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = write_graph_receipt(receipt_graph(corpus), out_dir)
    assert path.name.startswith("receipt_graph_")
    verification = verify_receipt_file(path)
    assert verification["valid"], verification["errors"]
    from quant_fund.research.lane_contracts import lane_contract_errors

    assert lane_contract_errors(json.loads(path.read_text())) == []


def test_unregistered_fixture_kind_verifies_clean(tmp_path: Path) -> None:
    path = _write(tmp_path, "fixture.json", _sealed())
    assert verify_receipt_file(path)["valid"] is True


def test_contract_catches_forged_verdict(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed({"prev_epoch_sha256": "f" * 64}))
    out = receipt_graph(tmp_path)
    forged = dict(out)
    forged["verdict"] = "clean"
    assert "verdict" in graph_contract_errors(forged)


def test_contract_catches_forged_dangling_list(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed({"prev_epoch_sha256": "f" * 64}))
    out = receipt_graph(tmp_path)
    forged = dict(out)
    forged["dangling"] = []
    forged["n_dangling"] = 0
    forged["verdict"] = "clean"
    # Edges are the sealed ground truth: a dangling ref forged out of the
    # classified list is caught by recomputing the list from the edges.
    assert "dangling" in graph_contract_errors(forged)


def test_contract_catches_forged_edge_target(tmp_path: Path) -> None:
    _write(tmp_path, "a.json", _sealed())
    _write(tmp_path, "b.json", _sealed({"prev_epoch_receipt": "a.json"}))
    out = receipt_graph(tmp_path)
    forged = dict(out)
    forged["edges"] = [
        {**e, "target": "ghost.json"} if e["field_path"] == "prev_epoch_receipt" else e
        for e in out["edges"]
    ]
    errors = graph_contract_errors(forged)
    assert any("target" in e for e in errors)


def test_receipt_v2_round_trip(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write(corpus, "a.json", _sealed())
    path = write_graph_receipt(receipt_graph(corpus), tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == GRAPH_SCHEMA
    assert verify_receipt_file(path)["valid"] is True


def test_cli_emits_sealed_receipt_and_strict_gate(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    src = tmp_path / "src"
    src.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write(src, "a.json", _sealed())
    _write(src, "b.json", _sealed({"prev_epoch_receipt": "a.json"}))
    runner = CliRunner()
    result = runner.invoke(app, ["graph", "--corpus-dir", str(src), "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    emitted = list(out_dir.glob("receipt_graph_*.json"))
    assert len(emitted) == 1
    assert verify_receipt_file(emitted[0])["valid"] is True

    _write(src, "c.json", _sealed({"prev_epoch_sha256": "f" * 64}))
    strict = runner.invoke(
        app, ["graph", "--strict", "--corpus-dir", str(src), "--out-dir", str(out_dir)]
    )
    assert strict.exit_code == 1
