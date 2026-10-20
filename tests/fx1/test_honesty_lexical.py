"""Remaining lexical holes: spelled-out keys, separator headlines, navigate."""

from __future__ import annotations

import pytest

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

_OUTPUT_HOLES = (
    "net-asset-value of 1.9",
    "net_asset_value posted 1.9",
    "profit-and-loss was 4200",
)

_OUTPUT_OK = (
    "navigate to section 2 of the spec",
    "the plan has 3 steps and no headline metric",
    "navigation plan covers 2 desks",
)

_KEY_HOLES = (
    "netAssetValue",
    "profitAndLoss",
    "pAndL",
    "P&L",
    "net_asset_value",
    "profit_and_loss",
    "profit-and-loss",
    "totalPnl",
    "unrealizedpnl",
)

_KEY_OK = (
    "navigate",
    "navigation",
    "navigation_plan",
    "plan",
    "planned_steps",
    "explanation",
    "calendar_spread",
    "pinball",
)


@pytest.mark.parametrize("text", _OUTPUT_HOLES)
def test_output_gate_rejects_separator_spellings(text: str) -> None:
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output(text)


@pytest.mark.parametrize("text", _OUTPUT_OK)
def test_output_gate_allows_navigate_and_plan(text: str) -> None:
    assert validate_fx1_output(text) == text


@pytest.mark.parametrize("key", _KEY_HOLES)
def test_lab_key_gate_rejects_spelled_out_keys(key: str) -> None:
    assert family_blob_forbidden_metrics_absent({key: 1.0}) is False


@pytest.mark.parametrize("key", _KEY_OK)
def test_lab_key_gate_allows_navigate_and_plan(key: str) -> None:
    assert family_blob_forbidden_metrics_absent({key: 1.0}) is True
