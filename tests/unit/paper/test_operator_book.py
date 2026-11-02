"""Paper kill-file actions. Live desk behavior stays the existing trip."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.monitoring.kill_switch import FLATTEN, KillSwitch
from quant_fund.paper.operator_book import enforce_operator_book
from quant_fund.schemas.orders import Order as BrokerOrder
from quant_fund.schemas.orders import OrderSide, OrderStatus
from quant_fund.utils.operator_kill import ENV_PATH, write_operator_kill

_STAMP = datetime(2026, 11, 2, tzinfo=UTC)


def _broker(tmp_path, *, allow_capital: bool = True) -> SimulatedBroker:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    return SimulatedBroker(config=cfg, initial_cash=1000.0, allow_capital=allow_capital)


def _resting(order_id: str) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=_STAMP,
        decision_time=_STAMP,
        order_time=_STAMP,
        status=OrderStatus.ACKED,
    )


def _arm(monkeypatch: pytest.MonkeyPatch, path, state: str) -> None:
    write_operator_kill(path, state)
    monkeypatch.setenv(ENV_PATH, str(path))
    monkeypatch.delenv("DIPCATCHER_HALT_CANCELS_OPEN_ORDERS", raising=False)


def test_halt_cancels_only_when_asked(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path)
    broker.shares["A"] = 2.0
    broker.open_orders["b"] = _resting("b")
    broker.open_orders["a"] = _resting("a")
    _arm(monkeypatch, path, "HALT_NEW_ORDERS")
    held = enforce_operator_book(broker)
    assert held["action"] == "halt"
    assert held["cancelled"] == []
    assert set(broker.open_orders) == {"a", "b"}
    assert broker.shares["A"] == 2.0
    monkeypatch.setenv("DIPCATCHER_HALT_CANCELS_OPEN_ORDERS", "yes")
    cancelled = enforce_operator_book(broker)
    assert cancelled["action"] == "halt"
    assert cancelled["cancelled"] == ["a", "b"]
    assert broker.open_orders == {}
    assert broker.shares["A"] == 2.0


def test_cancel_open_orders_leaves_positions(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path)
    broker.shares["A"] = 2.0
    broker.open_orders["a"] = _resting("a")
    _arm(monkeypatch, path, "CANCEL_OPEN_ORDERS")
    report = enforce_operator_book(broker)
    assert report["action"] == "cancel"
    assert report["cancelled"] == ["a"]
    assert broker.shares["A"] == 2.0


def test_flatten_optional_does_not_move_the_book(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path)
    broker.shares["A"] = 2.0
    broker.open_orders["a"] = _resting("a")
    _arm(monkeypatch, path, "FLATTEN_OPTIONAL")
    report = enforce_operator_book(broker)
    assert report["action"] == "halt"
    assert report["cancelled"] == []
    assert broker.shares["A"] == 2.0
    assert "a" in broker.open_orders
    switch = KillSwitch(load_config("configs/paper.yaml").kill_switch)
    switch.set_state(FLATTEN)
    assert switch.may_flatten(True) is False


def test_flatten_is_deterministic_and_idempotent(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path)
    broker.cash = 1000.0
    broker.shares = {"A": 2.0, "B": -1.0}
    broker.last_marks = {"A": 10.0, "B": 4.0}
    broker.open_orders["b"] = _resting("b")
    broker.open_orders["a"] = _resting("a")
    _arm(monkeypatch, path, "FLATTEN")
    report = enforce_operator_book(broker, at=_STAMP)
    assert report["ok"] is True
    assert report["cancelled"] == ["a", "b"]
    assert report["names"] == ["A", "B"]
    assert broker.cash == pytest.approx(1016.0)
    assert broker.shares == {}
    again = enforce_operator_book(broker, at=_STAMP)
    assert again["names"] == []
    assert again["cancelled"] == []
    assert broker.cash == pytest.approx(1016.0)


def test_shadow_flatten_does_not_touch_cash(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path, allow_capital=False)
    broker.cash = 50.0
    broker.shares = {"A": 2.0}
    broker.last_marks = {"A": 10.0}
    _arm(monkeypatch, path, "FLATTEN")
    report = enforce_operator_book(broker)
    assert report["ok"] is True
    assert broker.shares == {}
    assert broker.cash == 50.0


@pytest.mark.parametrize("mark", [None, True, float("nan"), 0.0, -1.0])
def test_flatten_unmarked_cancels_and_keeps_positions(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mark: object
) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path)
    broker.cash = 1000.0
    broker.shares = {"A": 2.0, "B": -1.0}
    broker.open_orders["a"] = _resting("a")
    if mark is not None:
        broker.last_marks = {"A": 10.0, "B": mark}  # type: ignore[dict-item]
    else:
        broker.last_marks = {"A": 10.0}
    _arm(monkeypatch, path, "FLATTEN")
    report = enforce_operator_book(broker)
    assert report["ok"] is False
    assert report["problem"] == "flatten_unmarked"
    assert report["cancelled"] == ["a"]
    assert report["names"] == []
    assert broker.shares == {"A": 2.0, "B": -1.0}
    assert broker.cash == 1000.0
    assert broker.open_orders == {}


@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        ("garbage", "kill_switch_invalid"),
        ("symlink", "kill_switch_symlink"),
        ("directory", "kill_switch_directory"),
        ("oversize", "kill_switch_invalid"),
    ],
)
def test_bad_kill_file_does_not_mutate(
    tmp_path, monkeypatch: pytest.MonkeyPatch, setup: str, problem: str
) -> None:
    path = tmp_path / "kill_switch"
    broker = _broker(tmp_path)
    broker.shares = {"A": 2.0}
    broker.open_orders["a"] = _resting("a")
    broker.cash = 1000.0
    if setup == "garbage":
        path.write_text("NOPE\n", encoding="utf-8")
    elif setup == "symlink":
        real = tmp_path / "real"
        real.write_text("FLATTEN\n", encoding="utf-8")
        path.symlink_to(real)
    elif setup == "directory":
        path.mkdir()
    else:
        path.write_text("X" * 65, encoding="utf-8")
    monkeypatch.setenv(ENV_PATH, str(path))
    report = enforce_operator_book(broker)
    assert report["action"] == "blocked"
    assert report["ok"] is False
    assert report["problem"] == problem
    assert broker.shares == {"A": 2.0}
    assert "a" in broker.open_orders
    assert broker.cash == 1000.0


def test_missing_path_is_unconfigured(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(tmp_path)
    broker.shares = {"A": 1.0}
    monkeypatch.delenv(ENV_PATH, raising=False)
    report = enforce_operator_book(broker)
    assert report["action"] == "unconfigured"
    assert broker.shares == {"A": 1.0}


def test_live_kill_does_not_flatten_the_paper_book(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.cli.live_cmds import live_kill

    broker = _broker(tmp_path)
    broker.shares = {"A": 2.0}
    broker.open_orders["a"] = _resting("a")
    path = tmp_path / "kill_switch"
    monkeypatch.setattr("quant_fund.cli.live_cmds._use_gateway", lambda *_a, **_k: False)
    monkeypatch.setattr("quant_fund.cli.live_cmds._desk", lambda *_a, **_k: None)
    live_kill(reason="ops3-test", kill_switch_file=path)
    assert path.read_text(encoding="utf-8").strip() == "HALT_NEW_ORDERS"
    assert broker.shares == {"A": 2.0}
    assert "a" in broker.open_orders
    monkeypatch.setenv(ENV_PATH, str(path))
    monkeypatch.delenv("DIPCATCHER_HALT_CANCELS_OPEN_ORDERS", raising=False)
    report = enforce_operator_book(broker)
    assert report["cancelled"] == []
    assert broker.shares == {"A": 2.0}


def test_unsupported_book_is_not_guessed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Desk:
        shares = {"A": 1.0}

    desk = Desk()
    _arm(monkeypatch, tmp_path / "kill_switch", "FLATTEN")
    report = enforce_operator_book(desk)
    assert report["ok"] is False
    assert report["problem"] == "paper_book_unsupported"
    assert desk.shares == {"A": 1.0}


def test_ledger_flatten_is_deterministic(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    ledger = OrderFillLedger(starting_cash=1000.0, accept_orders=True)
    ledger.submit(Order(order_id="work", security_id="A", side="BUY", quantity=1.0))
    ledger._computed_cash = 1000.0
    ledger._computed_positions["A"] = 2.0
    ledger._computed_positions["B"] = -1.0
    _arm(monkeypatch, tmp_path / "kill_switch", "FLATTEN")
    report = enforce_operator_book(ledger, marks={"A": 10.0, "B": 4.0})
    assert report["ok"] is True
    assert report["cancelled"] == ["work"]
    assert report["names"] == ["A", "B"]
    view = ledger.reconcile()
    assert view["computed_cash"] == pytest.approx(1016.0)
    assert abs(view["computed_positions"]["A"]) <= 1e-9
    assert abs(view["computed_positions"]["B"]) <= 1e-9
    rows = {row["order_id"]: row for row in view["orders"]}
    assert rows["flatten:A"]["side"] == "SELL"
    assert rows["flatten:B"]["side"] == "BUY"
    assert ledger._fills["flatten-fill:A"].fill_time == _STAMP
    again = enforce_operator_book(ledger, at=_STAMP)
    assert again["names"] == []
    assert ledger.reconcile()["n_fills"] == 2
    assert ledger.reconcile()["computed_cash"] == pytest.approx(1016.0)


def test_ledger_unmarked_does_not_apply_a_partial_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    ledger = OrderFillLedger(starting_cash=1000.0)
    ledger._computed_positions["A"] = 2.0
    ledger._computed_positions["B"] = -1.0
    _arm(monkeypatch, tmp_path / "kill_switch", "FLATTEN")
    report = enforce_operator_book(ledger, at=_STAMP)
    assert report["problem"] == "flatten_unmarked"
    assert ledger._computed_positions == {"A": 2.0, "B": -1.0}
    assert ledger._computed_cash == 1000.0
    assert ledger._fills == {}


def test_record_fill_does_not_double_apply_a_flatten() -> None:
    ledger = OrderFillLedger(starting_cash=1000.0)
    ledger._computed_positions["A"] = 2.0
    ledger.flatten_operator_book({"A": 10.0}, at=_STAMP)
    first = ledger._computed_cash
    verdict = ledger.record_fill(
        Fill(
            fill_id="flatten-fill:A",
            order_id="flatten:A",
            security_id="A",
            quantity=2.0,
            price=10.0,
            fill_time=_STAMP,
        )
    )
    assert verdict == "duplicate_fill"
    assert ledger._computed_cash == pytest.approx(first)


def test_paper_loop_calls_enforce_for_champion_and_shadow(tmp_path, monkeypatch) -> None:
    import polars as pl

    from quant_fund.paper.loop import run_paper_loop
    from quant_fund.paper.operator_book import enforce_operator_book as real

    calls: list[str] = []

    def spy(book: object, *, at: datetime | None = None) -> dict[str, object]:
        calls.append(str(getattr(book, "slot", "")))
        return real(book, at=at)

    monkeypatch.setattr("quant_fund.paper.operator_book.enforce_operator_book", spy)
    monkeypatch.delenv(ENV_PATH, raising=False)
    times = [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3)]
    rows = []
    for stamp in times:
        rows.append(
            {
                "security_id": "A",
                "event_time": stamp,
                "open": 100.0,
                "close": 100.0,
                "close_total_return": 100.0,
                "volume": 1_000_000.0,
                "adv": 100_000_000.0,
                "vol_20": 0.02,
                "source": "synthetic",
            }
        )
    weights = pl.DataFrame(
        {
            "event_time": times[:2],
            "security_id": ["A", "A"],
            "target_weight": [0.1, 0.1],
        }
    )
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    result = run_paper_loop(
        pl.DataFrame(rows),
        cfg,
        champion_weights=weights,
        shadow_weights=weights,
        initial_nav=100_000.0,
        max_steps=2,
    )
    assert result.source_note == "SYNTHETIC"
    assert "champion" in calls
    assert "shadow" in calls
