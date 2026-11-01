"""Unit tests for ``quant_fund.cli.benchmark_cmds``.

The benchmark lanes score **real** data, so the output contract matters as
much as the maths: the operator must see the verdict, not the frozen
scenario table, and a payload that cannot be represented as JSON must fail
rather than be printed with a ``NaN`` in it.

The unit-level seams pinned here are the ones the CLI-level tests in
``test_benchmark_cmds.py`` do not reach directly:

* ``_emit`` drops the bulky frozen inputs by default and keeps them on
  request;
* ``_emit`` refuses a non-finite float (``allow_nan=False``) instead of
  emitting a document no strict JSON parser can read;
* ``_check_receipt_version`` accepts only the two declared receipt versions;
* the three sub-applications are mounted under their documented names.

Nothing here is a performance claim — these lanes score research artifacts.
"""

from __future__ import annotations

import json

import pytest
import typer

from quant_fund.cli import benchmark_cmds
from quant_fund.cli.main import app


def _captured(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_emit_drops_the_bulky_frozen_inputs_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_cmds._emit({"verdict": "ok", "scenarios": [1, 2], "benchmark_manifest": {"x": 1}})
    payload = _captured(capsys)
    assert payload == {"verdict": "ok"}


def test_emit_keeps_the_heavy_keys_when_asked(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark_cmds._emit({"verdict": "ok", "scenarios": [1, 2]}, drop_heavy=False)
    payload = _captured(capsys)
    assert payload["scenarios"] == [1, 2]


def test_emit_leaves_a_payload_without_heavy_keys_untouched(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark_cmds._emit({"a": 1, "b": "two"})
    assert _captured(capsys) == {"a": 1, "b": "two"}


def test_emit_refuses_a_non_finite_float() -> None:
    """A NaN in the payload would produce a document no strict parser accepts."""
    with pytest.raises(ValueError):
        benchmark_cmds._emit({"metric": float("nan")})


def test_emit_refuses_an_infinite_float() -> None:
    with pytest.raises(ValueError):
        benchmark_cmds._emit({"metric": float("inf")})


@pytest.mark.parametrize("version", [1, 2])
def test_the_declared_receipt_versions_are_accepted(version: int) -> None:
    benchmark_cmds._check_receipt_version(version)


@pytest.mark.parametrize("version", [0, 3, -1, 99])
def test_any_other_receipt_version_is_refused(version: int) -> None:
    with pytest.raises(typer.BadParameter, match="must be 1 or 2"):
        benchmark_cmds._check_receipt_version(version)


def test_the_three_benchmark_sub_apps_are_mounted_under_their_documented_names() -> None:
    groups = {group.name for group in app.registered_groups}
    assert {"real-benchmark", "net-tournament", "ranker-probability"} <= groups


def test_each_benchmark_sub_app_exposes_its_commands() -> None:
    for name, commands in {
        "real-benchmark": {"prepare", "score"},
        "net-tournament": {"prepare", "run"},
        "ranker-probability": {"run"},
    }.items():
        group = next(g for g in app.registered_groups if g.name == name)
        registered = {command.name for command in group.typer_instance.registered_commands}
        assert commands <= registered, name
