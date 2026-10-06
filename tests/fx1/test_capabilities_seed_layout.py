"""Adversarial probes for the compact capability seed layout.

``fx1.capabilities`` claims one arithmetic progression per extension owner —
the compact form of the expanded shards under
``scripts/generated_capability_declarations/``. The audited contract:

- ``resolve_seed_id`` maps a seed id back to ``{kind, seed_id, owner,
  <owner-field>}`` and refuses ids outside the registered layout. The
  progressions partition ``[0, 1_000_000]`` exactly — every in-range id is
  claimed once.
- ``owner_references`` reproduces exactly ``count`` ids per owner and refuses
  unknown kinds/owners.
- The progressions are disjoint: ``resolve_seed_id`` returns the first match
  in table order, so any overlap would silently mis-attribute an id.
- Every table owner names a live generated shard (hyphens flatten to
  underscores on disk).

Correctness probes only — seed ids are discovery identifiers, never
capabilities or market evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fx1.capabilities import _TABLES, owner_references, resolve_seed_id

_LEDGER_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "generated_capability_declarations"
_KIND_DIR = {"feature": "features", "skill": "skills", "plugin": "plugins"}
_OWNER_FIELD = {"feature": "feature", "skill": "command", "plugin": "source"}


def test_resolve_seed_id_refuses_outside_layout() -> None:
    for bad in (-1, -7, 1_000_001, 2_000_000):
        with pytest.raises(KeyError, match="outside the registered layout"):
            resolve_seed_id(bad)


def test_seed_progressions_partition_the_layout() -> None:
    # The registered layout claims every id in [0, 1_000_000] exactly once —
    # a complete partition, so an in-range id can never be unclaimed.
    claimed = {
        first + stride * i
        for table in _TABLES.values()
        for _, first, stride, count in table
        for i in range(count)
    }
    assert claimed == set(range(1_000_001))


def test_resolve_seed_id_shape_and_owner_field() -> None:
    # doctor's progression starts at 0 — the layout's first claimable id.
    entry = resolve_seed_id(0)
    assert entry == {"kind": "skill", "seed_id": 0, "owner": "doctor", "command": "doctor"}


def test_owner_references_rejects_unknown_kind_and_owner() -> None:
    bad_kind: Any = "capability"
    with pytest.raises(ValueError, match="unknown extension kind"):
        owner_references(bad_kind, "doctor")
    with pytest.raises(KeyError, match="unknown feature extension owner"):
        owner_references("feature", "not_a_feature")
    with pytest.raises(KeyError, match="unknown skill extension owner"):
        owner_references("skill", "book-panel.py")  # filename, not owner


def test_owner_references_progression_shape() -> None:
    for kind, table in _TABLES.items():
        for owner, first, stride, count in table:
            refs = owner_references(kind, owner)
            assert len(refs) == count, (kind, owner)
            assert refs[0] == first
            assert refs[-1] == first + stride * (count - 1)
            assert all(b - a == stride for a, b in zip(refs, refs[1:], strict=False))


def test_seed_progressions_are_disjoint() -> None:
    # resolve_seed_id returns the first table match; an overlap anywhere —
    # inside a table or across kinds — would silently mis-attribute the id.
    claimed: set[int] = set()
    total = 0
    for table in _TABLES.values():
        for _owner, first, stride, count in table:
            total += count
            for i in range(count):
                claimed.add(first + stride * i)
    assert len(claimed) == total


def test_seed_id_round_trips_to_owning_entry() -> None:
    for kind, table in _TABLES.items():
        for owner, first, stride, count in table:
            for seed_id in (first, first + stride * (count - 1)):
                entry = resolve_seed_id(seed_id)
                assert entry["kind"] == kind, (kind, owner, seed_id)
                assert entry["owner"] == owner, (kind, owner, seed_id)
                assert entry["seed_id"] == seed_id
                assert set(entry) == {"kind", "seed_id", "owner", _OWNER_FIELD[kind]}
            past_end = first + stride * count
            try:
                entry = resolve_seed_id(past_end)
            except KeyError:
                continue  # beyond the layout ceiling — unclaimable
            assert not (entry["kind"] == kind and entry["owner"] == owner), (
                kind,
                owner,
            )


def test_every_table_owner_has_a_generated_shard() -> None:
    for kind, table in _TABLES.items():
        for owner, _first, _stride, _count in table:
            shard = _LEDGER_ROOT / _KIND_DIR[kind] / f"{owner.replace('-', '_')}.py"
            assert shard.is_file(), shard
