"""forecast_scoring.v1 is a lazy lane on the receipt.v2 verifier."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from quant_fund.schemas import receipt_v2_leaf as leaf
from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file

_RECEIPTS = Path("/workspace/receipts")
_KIND = "forecast_scoring.v1"
_CHECKER = "quant_fund.research.forecast_scoring.forecast_scoring_v2_consistency_errors"


def test_dispatch_uses_the_full_kind_and_not_the_stripped_base() -> None:
    assert leaf._LANE_CONSISTENCY[_KIND] == _CHECKER
    assert "forecast_scoring" not in leaf._LANE_CONSISTENCY
    assert _KIND not in leaf._LANE_CONSISTENCY_OUTER_KIND


def test_leaf_import_does_not_load_forecast_scoring() -> None:
    source = Path(leaf.__file__ or "").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert not any("forecast_scoring" in name for name in modules)


def _verdict(path: Path) -> bool:
    return verify_receipt_file(path)["valid"] is True


def test_committed_receipts_do_not_flip_when_the_lane_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = sorted(_RECEIPTS.glob("*.json"))
    assert paths, "committed receipts are the before/after control"
    armed = [_verdict(path) for path in paths]
    reduced = dict(leaf._LANE_CONSISTENCY)
    reduced.pop(_KIND, None)
    monkeypatch.setattr(leaf, "_LANE_CONSISTENCY", reduced)
    plain = [_verdict(path) for path in paths]
    flipped = [
        path.name
        for path, with_lane, without_lane in zip(paths, armed, plain, strict=True)
        if without_lane and not with_lane
    ]
    assert flipped == []
    assert json.dumps(armed) == json.dumps(plain)
