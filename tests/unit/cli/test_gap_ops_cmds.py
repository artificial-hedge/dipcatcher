"""Unit tests for ``quant_fund.cli.ops_cmds``.

Two guard rails live in this module, and both are about refusing to do the
wrong thing rather than about doing the right thing efficiently:

* ``--halt`` and ``--clear-halt`` are mutually exclusive kill-switch
  overrides. Passing both is a usage error, not a "last one wins" race that
  could silently pick the wrong kill-switch state for a paper run;
* the API server refuses a **non-loopback** binding unless an API key is
  configured, because an unauthenticated research API exposed on a routable
  interface is an open door.

The API guard is tested with ``uvicorn.run`` replaced by a recorder, so no
socket is ever opened.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import typer

from quant_fund.cli import ops_cmds

# --------------------------------------------------------------------------- #
# Kill-switch overrides
# --------------------------------------------------------------------------- #


def test_both_kill_switch_overrides_at_once_are_refused() -> None:
    with pytest.raises(typer.BadParameter, match="pass only one of"):
        ops_cmds._reject_halt_flags(halt=True, clear_halt=True)


@pytest.mark.parametrize(
    ("halt", "clear_halt"),
    [(True, False), (False, True), (False, False)],
)
def test_at_most_one_kill_switch_override_is_accepted(halt: bool, clear_halt: bool) -> None:
    assert ops_cmds._reject_halt_flags(halt=halt, clear_halt=clear_halt) is None


# --------------------------------------------------------------------------- #
# API binding guard
# --------------------------------------------------------------------------- #


@pytest.fixture
def recorded_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``uvicorn.run`` so no socket is ever opened."""
    import uvicorn

    calls: dict[str, Any] = {}

    def _fake_run(app: str, **kwargs: Any) -> None:
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    return calls


def test_a_loopback_binding_needs_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
    recorded_uvicorn: dict[str, Any],
) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    ops_cmds.api(host="127.0.0.1", port=8000)
    assert recorded_uvicorn["host"] == "127.0.0.1"
    assert recorded_uvicorn["port"] == 8000
    assert recorded_uvicorn["reload"] is False


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_every_loopback_host_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    recorded_uvicorn: dict[str, Any],
    host: str,
) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    ops_cmds.api(host=host, port=8001)
    assert recorded_uvicorn["host"] == host


def test_a_non_loopback_binding_without_an_api_key_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    recorded_uvicorn: dict[str, Any],
) -> None:
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    with pytest.raises(typer.BadParameter, match="non-loopback API binding requires"):
        ops_cmds.api(host="0.0.0.0", port=8000)
    assert recorded_uvicorn == {}


def test_a_non_loopback_binding_with_an_api_key_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    recorded_uvicorn: dict[str, Any],
) -> None:
    monkeypatch.setenv("QUANT_API_KEY", "test-key-not-a-real-secret")
    ops_cmds.api(host="0.0.0.0", port=8000)
    assert recorded_uvicorn["host"] == "0.0.0.0"


def test_an_empty_api_key_does_not_satisfy_the_guard(
    monkeypatch: pytest.MonkeyPatch,
    recorded_uvicorn: dict[str, Any],
) -> None:
    """An empty environment variable is not a configured key."""
    monkeypatch.setenv("QUANT_API_KEY", "")
    with pytest.raises(typer.BadParameter, match="non-loopback API binding requires"):
        ops_cmds.api(host="10.0.0.1", port=8000)
    assert recorded_uvicorn == {}


def test_the_environment_is_not_mutated_by_the_guard(
    monkeypatch: pytest.MonkeyPatch,
    recorded_uvicorn: dict[str, Any],
) -> None:
    before = dict(os.environ)
    monkeypatch.delenv("QUANT_API_KEY", raising=False)
    with pytest.raises(typer.BadParameter):
        ops_cmds.api(host="0.0.0.0", port=8000)
    assert dict(os.environ) == before
