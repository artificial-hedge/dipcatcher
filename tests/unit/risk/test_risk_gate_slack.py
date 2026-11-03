"""Risk-ceiling slack is strict.

A breach equal to the allowed float slack stays inside the ceiling. One
slack more rejects. This is a limit check, not a performance result.
"""

from __future__ import annotations

from quant_fund.portfolio.risk_gate import LIMIT_ABS_SLACK, LIMIT_REL_SLACK, exceeds_limit


def test_breach_equal_to_slack_stays_inside_the_ceiling() -> None:
    limit = 0.0
    slack = max(LIMIT_ABS_SLACK, LIMIT_REL_SLACK * abs(limit))
    assert (slack - limit) == slack
    assert exceeds_limit(slack, limit) is False
    assert exceeds_limit(slack * 2.0, limit) is True
