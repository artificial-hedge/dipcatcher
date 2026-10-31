"""Operator order surface. One package, two books, no live-capital claim.

The desk (shadow, in-process loopback, drop folder) is the default. It does
not open a broker socket. The gateway is selected by a gateway config and can
submit to Alpaca paper after an explicit arm. The Alpaca live host stays
named and is refused. Evidence conditions stay blocked. ``live_pnl_claim``
is false.
"""

from quant_fund.live.arming import CONFIRM_DROP, ENV_ARM, LIVE_CAPITAL_PHRASE, Arming
from quant_fund.live.config import LiveConfig, load_live_config
from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.errors import BrokerRejected, LiveBrokerRefused, LiveGateError
from quant_fund.live.session import LiveSession, open_session

__all__ = [
    "CONFIRM_DROP",
    "ENV_ARM",
    "LIVE_CAPITAL_PHRASE",
    "Arming",
    "BrokerRejected",
    "LiveBrokerRefused",
    "LiveConfig",
    "LiveDesk",
    "LiveDeskError",
    "LiveGateError",
    "LiveSession",
    "load_live_config",
    "open_session",
]
