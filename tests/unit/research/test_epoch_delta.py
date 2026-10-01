"""epoch_delta.v1 — delta completeness between two corpus epochs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import epoch_root
from quant_fund.research.epoch_delta import (
    epoch_delta_errors,
    epoch_delta_receipt,
    member_map_sha256,
    verify_epoch_delta,
    write_epoch_delta,
)
from quant_fund.research.epoch_merkle import merkle_root
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _write_epoch(dir_: Path, members: dict[str, str], prev: str | None) -> str:
    """Emit a sealed corpus_epoch receipt file into ``dir_``."""
    member_list = [{"name": n, "sha256": s} for n, s in sorted(members.items())]
    body: dict = {
        "kind": "corpus_epoch.v1",
        "schema": "corpus_epoch.v1",
        "epoch_root_sha256": epoch_root(members),
        "members": member_list,
        "n_members": len(member_list),
        "prev_epoch_receipt": prev,
        "params": {"pattern": "*.json"},
        "data_label": "CORPUS",
    }
    seal = hash_bytes(canonical_json_bytes(body))
    name = f"corpus_epoch_{seal[:16]}.json"
    (dir_ / name).write_text(json.dumps({**body, "receipt_sha256": seal}), encoding="utf-8")
    return name


def _members(base: dict[str, str]) -> dict[str, str]:
    return dict(base)


BASE = {
    "alpha.json": hashlib.sha256(b"a").hexdigest(),
    "beta.json": hashlib.sha256(b"b").hexdigest(),
    "gamma.json": hashlib.sha256(b"g").hexdigest(),
    "delta.json": hashlib.sha256(b"d").hexdigest(),
}


@pytest.fixture
def two_epochs(tmp_path: Path) -> tuple[Path, str, str]:
    """Genesis epoch + a successor with add/remove/change transitions."""
    prev = _write_epoch(tmp_path, BASE, None)
    nxt_members = _members(BASE)
    nxt_members.pop("gamma.json")  # removed
    nxt_members["epsilon.json"] = hashlib.sha256(b"e").hexdigest()  # added
    nxt_members["delta.json"] = hashlib.sha256(b"d2").hexdigest()  # changed
    nxt = _write_epoch(tmp_path, nxt_members, prev)
    return tmp_path, prev, nxt


def test_delta_receipt_round_trip(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    assert body["kind"] == "epoch_delta.v1"
    assert epoch_delta_errors(body) == []
    assert verify_epoch_delta(body, root) == []
    tr = body["transitions"]
    assert [r["name"] for r in tr["added"]] == ["epsilon.json"]
    assert [r["name"] for r in tr["removed"]] == ["gamma.json"]
    assert [r["name"] for r in tr["changed"]] == ["delta.json"]
    assert tr["unchanged_count"] == 2
    assert body["prev_epoch"]["n_members"] == 4
    assert body["next_epoch"]["n_members"] == 4


def test_delta_emits_sealed_file(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    path = write_epoch_delta(root, prev, nxt, root)
    payload = json.loads(path.read_text())
    seal = payload.pop("receipt_sha256")
    assert seal == hash_bytes(canonical_json_bytes(payload))
    assert path.name == f"epoch_delta_{seal[:16]}.json"
    assert verify_epoch_delta(payload, root) == []


def test_delta_rejects_same_epoch(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, _ = two_epochs
    with pytest.raises(ValueError, match="must differ"):
        epoch_delta_receipt(root, prev, prev)


def test_delta_dropped_transition_fails(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    body["transitions"]["removed"] = []
    # Self-contained check catches the accounting break; deep check catches
    # the missing row outright.
    assert "prev_accounting" in epoch_delta_errors(body)
    errors = verify_epoch_delta(body, root)
    assert "removed_set_incomplete" in errors


def test_delta_forged_row_cannot_anchor(two_epochs: tuple[Path, str, str]) -> None:
    """A row claiming a member not in the next epoch cannot forge a path."""
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    fake = {"name": "gamma.json", "sha256": BASE["gamma.json"]}
    fake["path"] = []
    fake["leaf_index"] = 0
    fake["n_members"] = 4
    body["transitions"]["added"].append(fake)
    errors = epoch_delta_errors(body)
    assert any(e.startswith("added_path_invalid") for e in errors)
    assert "next_accounting" in errors
    assert "added_set_incomplete" in verify_epoch_delta(body, root)


def test_delta_changed_row_needs_both_paths(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    row = body["transitions"]["changed"][0]
    row["from_sha256"] = hashlib.sha256(b"impostor").hexdigest()
    errors = epoch_delta_errors(body)
    assert "changed_prev_path_invalid:delta.json" in errors
    assert "changed_set_incomplete" in verify_epoch_delta(body, root)


def test_delta_tampered_epoch_fails_closed(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    body["prev_epoch"]["receipt"] = "corpus_epoch_0000000000000000.json"
    errors = verify_epoch_delta(body, root)
    assert "prev_epoch_epoch_receipt_missing" in errors


def test_delta_unrelated_epoch_rejected_cleanly(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    prev = _write_epoch(root, BASE, None)
    with pytest.raises(ValueError, match="not found"):
        epoch_delta_receipt(root, prev, "corpus_epoch_deadbeef00000000.json")


def test_map_digest_is_canonical() -> None:
    assert member_map_sha256(BASE) == member_map_sha256(dict(reversed(list(BASE.items()))))
    assert member_map_sha256(BASE) != member_map_sha256({**BASE, "z.json": "0" * 64})


def test_merkle_root_binding(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    next_members = _members(BASE)
    next_members.pop("gamma.json")
    next_members["epsilon.json"] = hashlib.sha256(b"e").hexdigest()
    next_members["delta.json"] = hashlib.sha256(b"d2").hexdigest()
    assert body["next_epoch"]["merkle_root"] == merkle_root(next_members)
    assert body["prev_epoch"]["merkle_root"] == merkle_root(BASE)


def test_delta_schema_guards(two_epochs: tuple[Path, str, str]) -> None:
    root, prev, nxt = two_epochs
    body = epoch_delta_receipt(root, prev, nxt)
    bad = dict(body, kind="corpus_proof.v1")
    assert "kind_not_epoch_delta" in epoch_delta_errors(bad)
    bad2 = dict(body, research_only=False)
    assert "research_only_not_true" in epoch_delta_errors(bad2)
