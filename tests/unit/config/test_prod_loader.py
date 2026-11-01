from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_fund.config.loader import dump_resolved, load_config
from quant_fund.config.models import AppConfig


def test_empty_config_uses_research_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    config = load_config(path)

    assert isinstance(config, AppConfig)
    assert config.runtime.allow_live is False


def test_truthy_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- runtime\n- research\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


def test_inheritance_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("inherit: second.yaml\n", encoding="utf-8")
    second.write_text("inherit: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inheritance cycle"):
        load_config(first)


def test_child_values_deep_merge_without_losing_parent_sections(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text("risk_gate:\n  max_gross: 1.5\n  max_net: 0.1\n", encoding="utf-8")
    child.write_text("inherit: parent.yaml\nrisk_gate:\n  max_net: 0.05\n", encoding="utf-8")

    config = load_config(child)

    assert config.risk_gate.max_gross == 1.5
    assert config.risk_gate.max_net == 0.05


def test_invalid_nested_config_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("risk_gate:\n  max_gross: .nan\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="max_gross"):
        load_config(path)


def test_config_io_errors_propagate_without_default_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")

    invalid_utf8 = tmp_path / "invalid.yaml"
    invalid_utf8.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        load_config(invalid_utf8)


def test_dump_resolved_writes_complete_json_and_creates_parents(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "resolved.json"

    dump_resolved(AppConfig(), destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["runtime"]["mode"] == "research"
    assert payload["runtime"]["allow_live"] is False
