"""The research metric exception is a false honesty flag, never a live claim."""

import pytest

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


@pytest.mark.parametrize("value", [True, 0, 0.0, 1, None, "false", "true", [], {}, {"crps": 0.2}])
@pytest.mark.parametrize("nested", [False, True])
def test_live_claim_exception_requires_literal_false(value: object, nested: bool) -> None:
    payload = {"live_pnl_claim": value, "crps": 0.2}
    if nested:
        payload = {"receipts": [{"results": (payload,)}]}
    assert family_blob_forbidden_metrics_absent(payload) is False


def test_false_claim_flags_remain_allowed_at_every_depth() -> None:
    payload = {
        "live_pnl_claim": False,
        "receipts": [{"live_pnl_claim": False, "results": ({"crps": 0.2},)}],
        "note": "sharpe is forbidden; live_pnl_claim must be false",
    }
    assert family_blob_forbidden_metrics_absent(payload) is True


def test_false_claim_flag_does_not_hide_sibling_metric_or_live_claim() -> None:
    assert (
        family_blob_forbidden_metrics_absent(
            {"live_pnl_claim": False, "receipts": [{"sharpe": 2.1}]}
        )
        is False
    )
    assert (
        family_blob_forbidden_metrics_absent(
            {"live_pnl_claim": False, "receipts": [{"live_pnl_claim": True}]}
        )
        is False
    )


@pytest.mark.parametrize("key", ["LIVE_PNL_CLAIM", "live-pnl-claim", "nested_live_pnl_claim"])
def test_only_canonical_flag_name_gets_the_exception(key: str) -> None:
    assert family_blob_forbidden_metrics_absent({key: False}) is False
