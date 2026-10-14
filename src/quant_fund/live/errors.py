"""Failures for operator order routing.

These are operational refusals. They are not research scores, and none of
them marks an institutional live-trading claim as allowed.
"""

from __future__ import annotations

from quant_fund.schemas.errors import ConfigError, QuantFundError


class LiveRefused(ConfigError):
    """The desk refused to route before any broker mutation."""


class TransportError(QuantFundError):
    """The HTTP call did not complete. No response body was read."""


class BrokerRejected(QuantFundError):
    """The broker returned a definite reject. The kill latch is unchanged."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class BrokerUncertain(QuantFundError):
    """A mutating request may have reached the broker. The desk latches the kill switch."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status
