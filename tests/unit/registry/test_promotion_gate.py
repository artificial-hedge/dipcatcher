"""Tests for registry/promotion_gate.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.registry.promotion_gate import (
    _mentions,
    promotion_audit,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _seal(tmp: Path, name: str, payload: dict) -> Path:
    payload = dict(payload)
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    p = tmp / name
    p.write_text(json.dumps(payload))
    return p


def _rec(**kw):
    base = {
        "kind": "demo",
        "schema": "demo.v1",
        "git_revision": "x",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {},
        "interpretation": "",
    }
    base.update(kw)
    return base


def test_mention_matching_exact_only():
    assert _mentions({"a": {"ar": 1, "var": 2}, "b": ["x", "ar"]}, "ar") == [
        "$.a.ar (key)",
        "$.b[1]",
    ]
    assert _mentions({"heads": ["aardvark"]}, "ar") == []


def test_unverified_when_silent(tmp_path):
    _seal(tmp_path, "r1.json", _rec(claim={"winner": "other_head"}))
    out = promotion_audit("mystery", tmp_path)
    assert out["verdict"] == "unverified"
    assert out["n_mentions"] == 0


def test_synthetic_only(tmp_path):
    _seal(tmp_path, "r1.json", _rec(data_label="SYNTHETIC", claim={"top": "head_a"}))
    out = promotion_audit("head_a", tmp_path)
    assert out["verdict"] == "synthetic_only"


def test_admitted_on_real_evidence(tmp_path):
    _seal(tmp_path, "r1.json", _rec(data_label="SYNTHETIC", claim={"top": "head_a"}))
    _seal(tmp_path, "r2.json", _rec(data_label="MIXED", claim={"losers": ["head_a"]}))
    out = promotion_audit("head_a", tmp_path)
    assert out["verdict"] == "admitted"
    assert out["n_mentions"] == 2


def test_rejected_on_unsealed_evidence(tmp_path):
    # take a REAL committed receipt, inject a name mention into a claim
    # leaf, and leave the stale seal — verification must fail closed
    src = Path("receipts")
    donor = next(iter(sorted(src.glob("*.json"))))
    rec = json.loads(donor.read_text())
    claim = rec.get("claim")
    if isinstance(claim, dict):
        claim["head_x"] = True
    else:
        rec["subject"] = "head_x"
    (tmp_path / "tampered.json").write_text(json.dumps(rec))
    out = promotion_audit("head_x", tmp_path)
    assert out["verdict"] == "rejected"
    assert out["mentions"][0]["verified"] is False


def test_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        promotion_audit("  ", tmp_path)
    with pytest.raises(FileNotFoundError):
        promotion_audit("x", tmp_path / "nope")


def test_audit_receipt_sealed(tmp_path):
    _seal(tmp_path, "r1.json", _rec(claim={"top": "head_a"}))
    a = promotion_audit("head_a", tmp_path)
    b = promotion_audit("head_a", tmp_path)
    assert a["schema"] == "promotion_gate.v1"
    assert a["receipt_sha256"] == b["receipt_sha256"]
