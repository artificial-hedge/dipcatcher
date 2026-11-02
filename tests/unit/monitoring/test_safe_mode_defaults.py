"""Safe defaults for kill switches, paper order intake, and short locates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.pretrade.support import make_engine

from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig, KillSwitchConfig
from quant_fund.execution.order_recon import Order, OrderFillLedger, OrderLedgerError
from quant_fund.monitoring.kill_switch import ENABLED, HALT_NEW_ORDERS, KillSwitch
from quant_fund.paper.broker_adapter import PaperBrokerAdapter
from quant_fund.schemas.errors import KillSwitchActive


def test_kill_switch_config_defaults_to_halt() -> None:
    cfg = KillSwitchConfig()
    assert cfg.state == HALT_NEW_ORDERS
    assert cfg.allow_auto_flatten is False
    switch = KillSwitch(cfg)
    with pytest.raises(KillSwitchActive, match=HALT_NEW_ORDERS):
        switch.assert_new_orders_allowed()


def test_kill_switch_enabled_is_explicit() -> None:
    switch = KillSwitch(KillSwitchConfig(state=ENABLED))
    switch.assert_new_orders_allowed()


def test_app_config_defaults_to_halt_and_production_yaml_matches() -> None:
    assert AppConfig().kill_switch.state == HALT_NEW_ORDERS
    production = load_config("configs/production.yaml")
    assert production.kill_switch.state == HALT_NEW_ORDERS
    assert production.runtime.allow_live is False
    paper = load_config("configs/paper.yaml")
    assert paper.kill_switch.state == ENABLED


def test_order_ledger_refuses_orders_until_opt_in() -> None:
    halted = OrderFillLedger(endpoint_kind="paper")
    order = Order(order_id="o1", security_id="A", side="BUY", quantity=1.0)
    with pytest.raises(KillSwitchActive, match="explicit_opt_in"):
        halted.submit(order)
    assert halted.kill_state == HALT_NEW_ORDERS

    armed = OrderFillLedger(endpoint_kind="paper", accept_orders=True)
    assert armed.submit(order) == "accepted"

    halted.clear_kill()
    assert halted.submit(order) == "accepted"

    with pytest.raises(OrderLedgerError, match="bool"):
        OrderFillLedger(endpoint_kind="paper", accept_orders=1)  # type: ignore[arg-type]


def test_paper_broker_defaults_to_halt_until_opt_in(tmp_path: Path) -> None:
    order = Order(order_id="o1", security_id="A", side="BUY", quantity=1.0)
    halted = PaperBrokerAdapter(state_path=tmp_path / "halt.json")
    with pytest.raises(KillSwitchActive):
        halted.submit(order)

    armed = PaperBrokerAdapter(state_path=tmp_path / "armed.json", accept_orders=True)
    assert armed.submit(order) == "accepted"


def test_set_symbol_does_not_grant_a_locate() -> None:
    engine = make_engine()
    sid = engine.ensure_symbol("A")
    engine.set_symbol(sid, pos=0.0, ref_px=100.0, ref_ts_ns=1)
    assert engine.book.syms[sid].locate == 0
