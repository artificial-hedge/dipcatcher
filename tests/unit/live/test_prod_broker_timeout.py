from __future__ import annotations

import pytest

from quant_fund.live.brokers import AlpacaBroker
from quant_fund.live.errors import LiveGateError


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_broker_rejects_unbounded_or_invalid_timeouts(timeout: float) -> None:
    with pytest.raises(LiveGateError, match="timeout must be finite and positive"):
        AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), timeout=timeout)
