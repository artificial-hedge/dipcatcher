"""Failures for the armed order gateway.

Messages are safe to print. They never include broker credentials.
"""

from __future__ import annotations


class LiveGateError(Exception):
    """The gateway refused to continue. No new broker call should be assumed."""


class BrokerRejected(LiveGateError):
    """The broker accepted the connection and refused the request."""


class LiveBrokerRefused(LiveGateError):
    """The Alpaca live host is configured and refused.

    The honesty contract forbids live broker connectivity. The venue name and
    the host stay in the source so the refusal is visible; paper and loopback
    are unchanged.
    """


class BrokerTimeout(Exception):
    """The broker did not answer in time.

    This is not a rejection. The order may still be live, so callers must not
    mark it dead or send a second copy.
    """


ALPACA_LIVE_REFUSAL = (
    "alpaca-live is refused: the honesty contract forbids live broker "
    "connectivity (AGENTS.md; docs/INSTITUTIONAL_READINESS.md). "
    "The venue and host api.alpaca.markets stay configured so this "
    "refusal is explicit. Paper and loopback are unchanged."
)
