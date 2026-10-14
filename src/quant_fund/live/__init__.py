"""Armed order gateway.

The paper broker still refuses a live endpoint. This package is the separate
path that can submit orders after an explicit arm and a pre-trade allow.
Fills are broker accounting. They are not research evidence and they are not
a profit claim.
"""

from quant_fund.live.arming import LIVE_CAPITAL_PHRASE
from quant_fund.live.config import LiveConfig, load_live_config
from quant_fund.live.errors import BrokerRejected, LiveGateError
from quant_fund.live.session import LiveSession, open_session

__all__ = [
    "LIVE_CAPITAL_PHRASE",
    "BrokerRejected",
    "LiveConfig",
    "LiveGateError",
    "LiveSession",
    "load_live_config",
    "open_session",
]
