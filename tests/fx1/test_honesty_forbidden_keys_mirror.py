"""Parity: fx-1 core audit and lab catalog must agree on mapping-key hygiene."""

from __future__ import annotations

import pytest

from fx1.forecast.core_audit import _forbidden_metric_keys_absent
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


@pytest.mark.parametrize(
    "payload",
    [
        {"coverage": 0.9},
        {"nested": {"mean_width": 0.1}},
        {"live_pnl_claim": False, "crps": 0.2},
        {"note": "sharpe is forbidden in prose only"},
        {"sharpe": 1.0},
        {"sharpeRatio": 1.0},
        {"totalPnl": 100.0},
        {"profitAndLoss": 100.0},
        {"unrealizedpnl": -1.0},
        {"mean_nav": 1e6},
        {"netAssetValue": 1e6},
        {"outer": {"inner_sortino": 0.2}},
        {"live_pnl_claim": True},
        {"nested": {"live_pnl_claim": True}},
    ],
)
def test_forbidden_key_gates_match_lab_catalog(payload: object) -> None:
    lab = family_blob_forbidden_metrics_absent(payload)
    fx1 = _forbidden_metric_keys_absent(payload)
    assert lab == fx1
