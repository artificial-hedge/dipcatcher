"""Crown-jewel pins: the gate-defining files can't drift silently."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.crown_jewels import (
    DEFAULT_JEWELS,
    crown_jewel_digests,
    crown_jewels_errors,
    load_crown_jewels_pin,
    write_crown_jewels_pin,
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    for rel in DEFAULT_JEWELS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n")
    return tmp_path


def test_round_trip_clean(repo: Path) -> None:
    pin = repo / "quality" / "crown_jewels.json"
    write_crown_jewels_pin(repo, pin)
    assert crown_jewels_errors(repo, pin) == []
    assert set(load_crown_jewels_pin(pin)) == set(DEFAULT_JEWELS)


def test_mutation_missing_and_symlink(repo: Path) -> None:
    pin = repo / "quality" / "crown_jewels.json"
    write_crown_jewels_pin(repo, pin)
    (repo / "pyproject.toml").write_text("# tampered\n")
    assert "jewel_mutated:pyproject.toml" in crown_jewels_errors(repo, pin)
    (repo / "AGENTS.md").unlink()
    assert "jewel_missing:AGENTS.md" in crown_jewels_errors(repo, pin)
    (repo / "uv.lock").unlink()
    (repo / "uv.lock").symlink_to("pyproject.toml")
    assert "jewel_symlink:uv.lock" in crown_jewels_errors(repo, pin)


def test_pin_cannot_shrink_or_grow(repo: Path) -> None:
    pin = repo / "quality" / "crown_jewels.json"
    write_crown_jewels_pin(repo, pin)
    doc = json.loads(pin.read_text())
    del doc["files"]["uv.lock"]
    pin.write_text(json.dumps(doc))
    assert "jewel_unpinned:uv.lock" in crown_jewels_errors(repo, pin)
    doc["files"]["uv.lock"] = "0" * 64
    doc["files"]["extra.txt"] = "0" * 64
    pin.write_text(json.dumps(doc))
    errors = crown_jewels_errors(repo, pin)
    assert "jewel_unexpected:extra.txt" in errors
    assert "jewel_mutated:uv.lock" in errors


def test_fail_closed_on_absent_or_malformed_pin(repo: Path) -> None:
    pin = repo / "quality" / "crown_jewels.json"
    assert crown_jewels_errors(repo, pin) == ["pin_missing"]
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text("not json")
    assert crown_jewels_errors(repo, pin)[0].startswith("pin_malformed")
    pin.write_text(json.dumps({"schema": "crown_jewels.v1", "files": {}}))
    # Empty pin ≡ corrupt for enforcement purposes — no unpinned pair checks,
    # but every jewel must be flagged unpinned.
    errors = crown_jewels_errors(repo, pin)
    assert len(errors) == len(DEFAULT_JEWELS)


def test_pin_is_json_sorted_and_atomic(repo: Path) -> None:
    pin = write_crown_jewels_pin(repo, repo / "quality" / "crown_jewels.json")
    doc = json.loads(pin.read_text())
    assert doc["schema"] == "crown_jewels.v1"
    assert list(doc["files"]) == sorted(doc["files"])
    digests = crown_jewel_digests(repo)
    assert digests["pyproject.toml"] != digests["uv.lock"]
