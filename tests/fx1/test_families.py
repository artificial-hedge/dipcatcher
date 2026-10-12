"""Fail-closed fx-1 client for the read-only family surface."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fx1.interactive import families as families_mod
from fx1.interactive.families import (
    SURFACE_SCHEMA,
    FamilyClient,
    FamilyClientError,
    harness_family_runner,
    selected_ids,
)


def _family(family_id: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "schema": SURFACE_SCHEMA,
        "id": family_id,
        "executable": True,
        "qualification": "required",
        "purpose": f"{family_id} diagnostics",
    }
    item.update(overrides)
    return item


def _select(
    selected: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SURFACE_SCHEMA,
        "goal": "volatility",
        "max_families": 4,
        "selected": [_family("volatility")] if selected is None else selected,
        "exclusions": [],
    }
    payload.update(extra)
    return payload


def _bind(stdout: str, *, code: int = 0, stderr: str = "") -> tuple[FamilyClient, list[list[str]]]:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> tuple[int, str, str]:
        calls.append(list(argv))
        return code, stdout, stderr

    return FamilyClient(runner), calls


def test_select_valid_payload_passes() -> None:
    payload = _select(
        [_family("volatility"), _family("jackknife_plus")],
        exclusions=[{"id": "tail", "reason": "over the cap"}],
        goal="volatility forecast",
        note="kept",
    )
    client, calls = _bind(json.dumps(payload))
    got = client.select("volatility forecast", max_families=2)
    assert got == payload
    assert got["note"] == "kept"
    assert calls == [["--select", "volatility forecast", "--max", "2"]]


def test_select_default_max_families_is_four() -> None:
    client, calls = _bind(json.dumps(_select()))
    client.select("ranking")
    assert calls == [["--select", "ranking", "--max", "4"]]


def test_select_executable_false_raises() -> None:
    payload = _select([_family("volatility", executable=False)])
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="executable"):
        client.select("volatility")


def test_select_does_not_drop_invalid_item() -> None:
    payload = _select([_family("ranking"), _family("alpha", executable=False)])
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="selected\\[1\\]"):
        client.select("ranking")


def test_select_qualification_live_optional_raises() -> None:
    payload = _select([_family("microstructure", qualification="live_optional")])
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="live_optional"):
        client.select("microstructure")


@pytest.mark.parametrize("qualification", ["retired", "unknown", "REQUIRED", None])
def test_select_rejects_other_qualifications(qualification: object) -> None:
    payload = _select([_family("ranking", qualification=qualification)])
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="qualification"):
        client.select("ranking")


@pytest.mark.parametrize("family_id", ["Ranking", "1alpha", "rank-ing", "alpha ", ""])
def test_select_bad_id_raises(family_id: str) -> None:
    payload = _select([_family(family_id)])
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="id"):
        client.select("ranking")


def test_select_bad_json_raises() -> None:
    client, _calls = _bind("{not-json")
    with pytest.raises(FamilyClientError, match="JSON"):
        client.select("volatility")


def test_select_nonzero_exit_includes_stderr_snippet() -> None:
    stderr = "surface refused: qualification live_optional\nsecond line"
    client, _calls = _bind("{", code=4, stderr=stderr)
    with pytest.raises(FamilyClientError) as caught:
        client.select("volatility")
    message = str(caught.value)
    assert stderr in message
    assert "4" in message


def test_select_wrong_schema_raises() -> None:
    payload = _select()
    payload["schema"] = "dipcatcher.family_surface.v0"
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="schema"):
        client.select("volatility")


def test_select_non_dict_raises() -> None:
    client, _calls = _bind(json.dumps([_family("ranking")]))
    with pytest.raises(FamilyClientError, match="JSON object"):
        client.select("ranking")


def test_select_missing_exclusions_raises() -> None:
    payload = _select()
    del payload["exclusions"]
    client, _calls = _bind(json.dumps(payload))
    with pytest.raises(FamilyClientError, match="exclusions"):
        client.select("volatility")


def test_selected_ids_preserves_order_and_dedupes() -> None:
    payload = {
        "schema": SURFACE_SCHEMA,
        "selected": [
            {"id": "ranking", "executable": True},
            {"id": "alpha"},
            {"id": "ranking"},
            {"id": "volatility"},
            {"id": "alpha"},
            {"id": "conformal"},
        ],
        "exclusions": [],
    }
    assert selected_ids(payload) == ["ranking", "alpha", "volatility", "conformal"]


def test_selected_ids_from_select_payload() -> None:
    payload = _select([_family("tail"), _family("drawdown"), _family("tail")])
    assert selected_ids(payload) == ["tail", "drawdown"]


def test_harness_family_runner_passes_extra_args() -> None:
    class _Harness:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        def run(self, name: str, extra_args: list[str]) -> SimpleNamespace:
            self.calls.append((name, extra_args))
            return SimpleNamespace(exit_code=0, stdout="out", stderr="err")

    harness = _Harness()
    extra = ["--select", "qlike", "--max", "1"]
    assert harness_family_runner(harness)(extra) == (0, "out", "err")
    assert harness.calls == [("families", extra)]


def test_harness_family_runner_missing_exit_code() -> None:
    class _Harness:
        def run(self, name: str, extra_args: list[str]) -> SimpleNamespace:
            return SimpleNamespace(stdout="", stderr="")

    with pytest.raises(FamilyClientError, match="exit_code"):
        harness_family_runner(_Harness())(["--id", "ranking"])


def test_harness_family_runner_feeds_client() -> None:
    payload = _select([_family("evalues")])

    class _Harness:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        def run(self, name: str, extra_args: list[str]) -> SimpleNamespace:
            self.calls.append((name, list(extra_args)))
            return SimpleNamespace(exit_code=0, stdout=json.dumps(payload), stderr="")

    harness = _Harness()
    got = FamilyClient(harness_family_runner(harness)).select("e-values")
    assert got == payload
    assert harness.calls == [("families", ["--select", "e-values", "--max", "4"])]


def test_list_executable_list_form() -> None:
    items = [_family("ranking"), _family("alpha")]
    client, calls = _bind(json.dumps(items))
    assert client.list_executable() == items
    assert calls == [[]]


def test_list_executable_families_object_form() -> None:
    items = [_family("volatility"), _family("distribution")]
    client, calls = _bind(json.dumps({"families": items}))
    assert client.list_executable() == items
    assert calls == [[]]


def test_list_executable_schema_envelope() -> None:
    items = [_family("regime"), _family("tail")]
    payload = {"schema": SURFACE_SCHEMA, "families": items}
    client, calls = _bind(json.dumps(payload))
    assert client.list_executable() == items
    assert calls == [[]]


def test_list_executable_rejects_non_required_item() -> None:
    items = [_family("ranking"), _family("microstructure", qualification="live_optional")]
    client, _calls = _bind(json.dumps(items))
    with pytest.raises(FamilyClientError, match="qualification"):
        client.list_executable()


def test_list_executable_rejects_object_without_families() -> None:
    client, _calls = _bind(json.dumps(_select()))
    with pytest.raises(FamilyClientError, match="families"):
        client.list_executable()


def test_status_requires_schema_and_id() -> None:
    payload = {
        "schema": SURFACE_SCHEMA,
        "id": "ranking",
        "executable": False,
        "qualification": "live_optional",
        "purpose": "visible to status",
    }
    client, calls = _bind(json.dumps(payload))
    assert client.status("ranking") == payload
    assert calls == [["--id", "ranking"]]


def test_status_missing_id_raises() -> None:
    client, _calls = _bind(json.dumps({"schema": SURFACE_SCHEMA, "purpose": "x"}))
    with pytest.raises(FamilyClientError, match="id"):
        client.status("ranking")


def test_status_nonzero_exit_includes_stderr() -> None:
    client, _calls = _bind("", code=2, stderr="unknown family")
    with pytest.raises(FamilyClientError, match="unknown family"):
        client.status("ranking")


def test_module_does_not_import_lab_or_subprocess() -> None:
    tree = ast.parse(Path(families_mod.__file__).read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    for name in modules:
        assert not name.startswith("quant_fund")
        assert name != "subprocess" and not name.startswith("subprocess.")


def test_family_client_error_is_value_error() -> None:
    assert issubclass(FamilyClientError, ValueError)
    assert SURFACE_SCHEMA == "dipcatcher.family_surface.v1"
