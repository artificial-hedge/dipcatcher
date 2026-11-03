"""Unit tests for ``quant_fund.cli.live_cmds``.

``dipcatcher live`` exposes two very different sessions behind one command
group: the *operator desk* (shadow / SYNTHETIC loopback / drop folder) and
the *broker gateway* (an armed session with real capital). The command
module decides which one an invocation lands in, and that decision is the
part worth pinning:

* **desk flags win** — ``--venue`` or ``--root`` always means the desk, even
  when a gateway file is present, so an operator can never be routed into a
  capital-bearing session by accident;
* **gateway flags select the gateway** — ``--state-dir``, ``--actor``,
  ``--at``, ``--client-order-id`` or ``--limit`` all mean "the broker";
* a **gateway config file** also selects the gateway.

The failure paths matter as much: both the desk and the gateway report a
failure as a structured payload that explicitly denies a live-P&L claim and
exits non-zero, rather than printing a traceback or exiting zero.

Nothing here touches a broker, a credential file or the network: the
filesystem-free helpers are exercised directly, and the gateway-file probe is
monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from quant_fund.cli import live_cmds
from quant_fund.live.arming import ENV_ROOT
from quant_fund.live.errors import LiveGateError


@pytest.fixture(autouse=True)
def _no_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every helper off the operator's real configured root."""
    monkeypatch.delenv(ENV_ROOT, raising=False)


# --------------------------------------------------------------------------- #
# Root and config resolution
# --------------------------------------------------------------------------- #


def test_the_default_root_follows_the_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_ROOT, "/tmp/some-live-root")
    assert live_cmds._default_root() == Path("/tmp/some-live-root")


def test_the_default_root_falls_back_to_the_home_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_ROOT, raising=False)
    assert live_cmds._default_root() == Path.home() / ".dipcatcher" / "live"


def test_resolve_uses_the_default_root_when_none_is_given() -> None:
    assert live_cmds._resolve(None) == Path.home() / ".dipcatcher" / "live"


def test_resolve_keeps_an_explicit_root() -> None:
    assert live_cmds._resolve(Path("/explicit")) == Path("/explicit")


def test_the_stamp_uses_the_supplied_value() -> None:
    assert live_cmds._stamp(1_700_000_000_000_000_000) == 1_700_000_000_000_000_000


def test_the_stamp_falls_back_to_the_wall_clock() -> None:
    stamp = live_cmds._stamp(None)
    assert isinstance(stamp, int)
    assert stamp > 0


def test_the_desk_config_has_a_documented_default() -> None:
    assert live_cmds._desk_config(None) == Path("configs/pretrade_risk.yaml")


def test_an_explicit_desk_config_wins_over_the_default() -> None:
    assert live_cmds._desk_config(Path("/etc/pretrade.yaml")) == Path("/etc/pretrade.yaml")


# --------------------------------------------------------------------------- #
# Desk / gateway selection
# --------------------------------------------------------------------------- #


@pytest.fixture
def gateway_file_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a gateway config file was found, without reading any file."""
    monkeypatch.setattr(live_cmds, "_gateway_file", lambda path: True)


def test_no_flags_and_no_file_stay_on_the_desk() -> None:
    assert live_cmds._use_gateway(None) is False


def test_a_gateway_file_selects_the_gateway(gateway_file_present: None) -> None:
    assert live_cmds._use_gateway(None) is True


def test_a_venue_flag_always_means_the_desk(gateway_file_present: None) -> None:
    """A desk flag beats a gateway file — the operator cannot be routed by accident."""
    assert live_cmds._use_gateway(None, venue="shadow") is False


def test_a_root_flag_always_means_the_desk(gateway_file_present: None) -> None:
    assert live_cmds._use_gateway(None, root=Path("/r")) is False


def test_a_root_flag_beats_a_gateway_flag() -> None:
    assert live_cmds._use_gateway(None, root=Path("/r"), state_dir=Path("/s")) is False


def test_a_venue_flag_beats_an_actor_flag() -> None:
    assert live_cmds._use_gateway(None, venue="shadow", actor="ops") is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state_dir": Path("/s")},
        {"actor": "ops"},
        {"at": "2024-01-01"},
        {"client_order_id": "coid-1"},
        {"limit": 100.0},
    ],
)
def test_a_gateway_flag_selects_the_gateway(kwargs: dict[str, object]) -> None:
    assert live_cmds._use_gateway(None, **kwargs) is True


def test_a_blank_actor_is_not_a_gateway_flag() -> None:
    """Whitespace is not an operator identity."""
    assert live_cmds._use_gateway(None, actor="   ") is False


def test_a_zero_limit_is_still_a_gateway_flag() -> None:
    """The flag's presence is the signal, not its magnitude."""
    assert live_cmds._use_gateway(None, limit=0.0) is True


# --------------------------------------------------------------------------- #
# Desk failure output
# --------------------------------------------------------------------------- #


def test_a_desk_failure_reports_a_blocked_honesty_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        live_cmds._fail_desk(RuntimeError("boom"))
    assert excinfo.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "boom"
    assert payload["live_pnl_claim"] is False
    assert payload["would_promote_live"] is False
    assert payload["evidence_conditions"] == "blocked"
    assert payload["schema"] == "live_desk_report.v1"


def test_the_desk_report_is_emitted_with_sorted_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_cmds._emit_desk({"b": 2, "a": 1})
    out = capsys.readouterr().out
    assert out.index('"a"') < out.index('"b"')


# --------------------------------------------------------------------------- #
# Gateway output and exit codes
# --------------------------------------------------------------------------- #


def test_a_gateway_failure_denies_a_live_pnl_claim(
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_cmds._fail_gateway("no session")
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "no session"
    assert payload["live_pnl_claim"] is False
    assert payload["research_evidence"] is False


def test_a_successful_gateway_payload_is_echoed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_cmds._invoke_gateway(lambda: {"ok": True, "detail": "armed"})
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "detail": "armed"}


def test_a_gateway_payload_reporting_failure_exits_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as excinfo:
        live_cmds._invoke_gateway(lambda: {"ok": False, "detail": "refused"})
    assert excinfo.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_a_gate_error_is_reported_and_exits_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> dict[str, object]:
        raise LiveGateError("not armed")

    with pytest.raises(typer.Exit) as excinfo:
        live_cmds._invoke_gateway(boom)
    assert excinfo.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "not armed"
    assert payload["live_pnl_claim"] is False
