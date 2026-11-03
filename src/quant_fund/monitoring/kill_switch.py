"""Kill switch. Flatten requires explicit human authorization."""

from __future__ import annotations

from quant_fund.config.models import KillSwitchConfig
from quant_fund.schemas.errors import KillSwitchActive

ENABLED = "ENABLED"
HALT_NEW_ORDERS = "HALT_NEW_ORDERS"
CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
FLATTEN_OPTIONAL = "FLATTEN_OPTIONAL"

VALID_STATES = frozenset({ENABLED, HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL})
# Any stored non-ENABLED state blocks new orders — fail closed.
BLOCKING_STATES = frozenset({HALT_NEW_ORDERS, CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL})


class KillSwitch:
    def __init__(self, config: KillSwitchConfig) -> None:
        self.allow_auto_flatten = bool(config.allow_auto_flatten)
        # Fail closed: an unknown config state is rejected, not stored.
        self._state = str(config.state)
        if self._state not in VALID_STATES:
            raise KillSwitchActive(f"invalid kill switch state={self._state}")

    def set_state(self, state: str) -> None:
        """Transition kill-switch state. Invalid names halt, then raise."""
        if state not in VALID_STATES:
            self._state = HALT_NEW_ORDERS
            raise KillSwitchActive(f"invalid kill switch state={state}")
        self._state = state

    def assert_new_orders_allowed(self) -> None:
        # Only ENABLED permits new orders. Unknown / halt / flatten all raise.
        if self.state != ENABLED:
            raise KillSwitchActive(f"kill switch state={self.state}")

    def may_cancel_open_orders(self) -> bool:
        return self.state in {CANCEL_OPEN_ORDERS, FLATTEN_OPTIONAL}

    def may_flatten(self, human_authorized: bool) -> bool:
        if self.state != FLATTEN_OPTIONAL:
            return False
        if self.allow_auto_flatten:
            return False  # still require human; auto flatten never on exception
        return bool(human_authorized)

    @property
    def state(self) -> str:
        """Stored state. Assignment goes through ``set_state``."""
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self.set_state(str(value))
