"""Fail-closed branches of the YAML config loader.

Covers: inheritance escapes, cycles, non-mapping documents, unknown keys, and
the resolved-config dump. Every fixture config tree is SYNTHETIC and built in
``tmp_path``; nothing here reads the repository's real configs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from quant_fund.config.loader import deep_merge, dump_resolved, load_config


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return path


def _base(name: str = "base") -> dict[str, object]:
    return {
        "runtime": {"experiment_id": name},
        "horizons": {"bars": [1, 5], "names": ["h1", "h5"]},
        "costs": {"commission_bps": 2.0, "half_spread_bps": 4.0},
    }


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_replaces_lists_instead_of_appending():
    merged = deep_merge({"a": [1, 2]}, {"a": [3]})
    assert merged["a"] == [3]


def test_deep_merge_keeps_parent_siblings_and_lets_overlay_win():
    merged = deep_merge({"b": {"x": 1, "y": 2}, "c": 1}, {"b": {"y": 9}, "c": 2})
    assert merged["b"] == {"x": 1, "y": 9}
    assert merged["c"] == 2


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"b": {"x": 1}}
    overlay = {"b": {"y": 2}}
    deep_merge(base, overlay)
    assert base == {"b": {"x": 1}}
    assert overlay == {"b": {"y": 2}}


# ---------------------------------------------------------------------------
# load_config — happy paths that the semantics depend on
# ---------------------------------------------------------------------------


def test_same_directory_inherit_merges_parent_and_child(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", _base())
    child = _write(
        tmp_path / "child.yaml",
        {"inherit": "base.yaml", "runtime": {"experiment_id": "child"}},
    )

    cfg = load_config(child)

    assert cfg.runtime.experiment_id == "child"
    assert cfg.horizons.bars == [1, 5]
    assert cfg.costs.commission_bps == 2.0


def test_inherit_is_consumed_and_not_rejected_as_unknown_key(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", _base())
    child = _write(tmp_path / "child.yaml", {"inherit": "base.yaml"})

    cfg = load_config(child)

    assert cfg.runtime.experiment_id == "base"


def test_load_config_accepts_a_string_path(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", _base("string"))

    assert load_config(str(path)).runtime.experiment_id == "string"


# ---------------------------------------------------------------------------
# load_config — fail-closed branches
# ---------------------------------------------------------------------------


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_inheritance_cycle_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "cyc" / "a.yaml", {"inherit": "b.yaml"})
    _write(tmp_path / "cyc" / "b.yaml", {"inherit": "a.yaml"})

    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "cyc" / "a.yaml")


def test_self_inheritance_is_rejected_as_a_cycle(tmp_path: Path) -> None:
    path = _write(tmp_path / "self.yaml", {"inherit": "self.yaml", **_base()})

    with pytest.raises(ValueError, match="cycle"):
        load_config(path)


def test_inheriting_outside_the_child_directory_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", _base())
    leaf = _write(tmp_path / "sub" / "leaf.yaml", {"inherit": "../base.yaml"})

    with pytest.raises(ValueError, match="escapes config root"):
        load_config(leaf)


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "bad.yaml", {"runtime": {"experiment_id": "x"}, "not_a_key": 1})

    with pytest.raises(ValueError):
        load_config(path)


def test_missing_inherit_target_raises_the_io_error(tmp_path: Path) -> None:
    child = _write(tmp_path / "child.yaml", {"inherit": "missing.yaml"})

    with pytest.raises(FileNotFoundError):
        load_config(child)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# dump_resolved
# ---------------------------------------------------------------------------


def test_dump_resolved_writes_sorted_json_and_creates_parents(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "cfg.yaml", _base("dump")))
    dest = tmp_path / "deep" / "nested" / "resolved.json"

    dump_resolved(cfg, dest)

    text = dest.read_text()
    assert text == json.dumps(cfg.dump(), indent=2, sort_keys=True)
    assert json.loads(text)["runtime"]["experiment_id"] == "dump"
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_dump_resolved_overwrites_an_existing_file(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "cfg.yaml", _base("second")))
    dest = tmp_path / "resolved.json"
    dest.write_text("stale")

    dump_resolved(cfg, dest)

    assert json.loads(dest.read_text())["runtime"]["experiment_id"] == "second"
