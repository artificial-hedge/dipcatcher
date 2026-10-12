"""Correlation ids are sortable, secret-free, and safe to use in paths."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fx1.interactive.runctx import (
    RunContext,
    is_valid_id,
    is_valid_step_id,
    new_id,
    new_job_id,
    new_run_id,
    new_session_id,
    require_id,
    step_id,
)

NOW = datetime(2026, 10, 12, 2, 45, 0, tzinfo=UTC)


def test_id_format_and_sortability() -> None:
    first = new_id("s", now=NOW, rand="00000001")
    later = new_id("s", now=datetime(2026, 10, 12, 2, 45, 1, tzinfo=UTC), rand="00000000")
    assert first == "s-20261012T024500Z-00000001"
    assert first < later
    assert new_session_id().startswith("s-") and new_run_id().startswith("r-")
    assert new_job_id().startswith("j-")


@pytest.mark.parametrize(
    "bad",
    ["", "s-1", "../etc/passwd", "s-20261012T024500Z-ZZZZZZZZ", "x-20261012T024500Z-0000000a"],
)
def test_malformed_ids_rejected(bad: str) -> None:
    assert not is_valid_id(bad)
    with pytest.raises(ValueError, match="malformed identifier"):
        require_id(bad)


def test_kind_is_enforced() -> None:
    sid = new_session_id()
    assert is_valid_id(sid, "s")
    assert not is_valid_id(sid, "r")
    with pytest.raises(ValueError):
        require_id(sid, "r")
    with pytest.raises(ValueError, match="unknown id kind"):
        new_id("z")
    with pytest.raises(ValueError, match="8 lowercase hex"):
        new_id("s", rand="NOT-HEX!")


def test_step_ids() -> None:
    rid = new_run_id()
    sid = step_id(rid, 7)
    assert sid == f"{rid}.007"
    assert is_valid_step_id(sid)
    assert not is_valid_step_id(rid)
    with pytest.raises(ValueError):
        step_id(rid, 1000)
    with pytest.raises(ValueError):
        step_id(new_session_id(), 1)


def test_run_context_progression_and_fields() -> None:
    ctx = RunContext(session_id=new_session_id(), model="fx1")
    assert ctx.as_fields() == {"session_id": ctx.session_id, "model": "fx1"}
    run = ctx.with_run()
    step = run.with_step(2, family="volatility")
    assert step.step == f"{run.run_id}.002"
    assert step.as_fields()["family"] == "volatility"
    assert step.with_model("fx1-lite").model == "fx1-lite"
    with pytest.raises(ValueError, match="requires a run id"):
        ctx.with_step(0)


def test_run_context_validates_ids() -> None:
    with pytest.raises(ValueError):
        RunContext(session_id="not-an-id")
    with pytest.raises(ValueError):
        RunContext(session_id=new_session_id(), run_id="bad")
    with pytest.raises(ValueError):
        RunContext(session_id=new_session_id(), step="bad")
