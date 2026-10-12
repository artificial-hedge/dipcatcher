"""Family client against the real lab surface and hostile documents."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from fx1.interactive.families import (
    SURFACE_SCHEMA,
    FamilyClient,
    FamilyClientError,
    selected_ids,
)
from quant_fund.research.catalog import registry

_ROOT = Path(__file__).resolve().parents[2]


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(_ROOT / "src"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "FX1_CONFIG_DIR": str(tmp_path / "fx1"),
        "FX1_SESSION_DIR": str(tmp_path / "sessions"),
        "OMP_NUM_THREADS": "1",
    }


def _lab(tmp_path: Path, argv: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "families", *argv],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        timeout=60,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _family(family_id: str = "volatility", **overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "schema": SURFACE_SCHEMA,
        "id": family_id,
        "executable": True,
        "qualification": "required",
    }
    item.update(overrides)
    return item


def _select(selected: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema": SURFACE_SCHEMA,
        "goal": "volatility",
        "max_families": 4,
        "selected": [_family()] if selected is None else selected,
        "exclusions": [],
    }


def test_live_select_is_byte_stable_and_the_client_accepts_it(tmp_path: Path) -> None:
    argv = ["--select", "qlike volatility forecast", "--max", "2"]
    first = _lab(tmp_path, argv)
    second = _lab(tmp_path, argv)
    assert first[0] == 0, first[2]
    assert first[1] == second[1]
    client = FamilyClient(lambda _argv: first)
    payload = client.select("qlike volatility forecast", max_families=2)
    assert [row["id"] for row in payload["selected"]] == ["volatility"]
    assert selected_ids(payload) == ["volatility"]
    assert all(row["executable"] is True for row in payload["selected"])


def test_live_status_of_a_retired_family_names_the_reason(tmp_path: Path) -> None:
    retired = next(iter(registry.RETIRED_BENCHMARK_FAMILIES))
    code, stdout, stderr = _lab(tmp_path, ["--id", retired])
    assert code == 0, stderr
    payload = FamilyClient(lambda _argv: (code, stdout, stderr)).status(retired)
    assert payload["qualification"] == "retired"
    assert payload["executable"] is False
    assert "template" in str(payload["retirement_reason"]).lower()


def test_live_list_is_exactly_the_required_runners(tmp_path: Path) -> None:
    code, stdout, stderr = _lab(tmp_path, [])
    assert code == 0, stderr
    rows = FamilyClient(lambda _argv: (code, stdout, stderr)).list_executable()
    assert [row["id"] for row in rows] == [row["id"] for row in json.loads(stdout)["families"]]
    assert {row["id"] for row in rows} == set(registry.REQUIRED_BENCHMARK_FAMILIES)
    assert all(row["qualification"] == "required" and row["executable"] is True for row in rows)


def test_client_rejects_a_wrong_schema_on_the_list_envelope() -> None:
    items = [_family("ranking"), _family("alpha")]
    hostile = {"schema": "dipcatcher.family_surface.v9", "families": items}
    client = FamilyClient(lambda _argv: (0, json.dumps(hostile), ""))
    with pytest.raises(FamilyClientError, match="schema"):
        client.list_executable()


def test_client_rejects_non_object_exclusions_and_bad_ids() -> None:
    payload = _select()
    payload["exclusions"] = ["retired"]
    client = FamilyClient(lambda _argv: (0, json.dumps(payload), ""))
    with pytest.raises(FamilyClientError, match="exclusions"):
        client.select("volatility")
    with pytest.raises(FamilyClientError, match="id"):
        selected_ids({"selected": [{"id": "vol;rm"}]})


def test_hostile_goal_and_id_never_reach_the_runner() -> None:
    def runner(argv: list[str]) -> tuple[int, str, str]:
        raise AssertionError(argv)

    client = FamilyClient(runner)
    with pytest.raises(FamilyClientError, match="NUL"):
        client.select("volatility\x00ranking")
    with pytest.raises(FamilyClientError, match="4096"):
        client.select("v" * 5000)
    with pytest.raises(FamilyClientError, match="max_families"):
        client.select("volatility", max_families=24)
    with pytest.raises(FamilyClientError, match="family id"):
        client.status("Volatility")
    with pytest.raises(FamilyClientError, match="family id"):
        client.status("x" * 10_000)


def test_two_hundred_mutated_documents_only_raise_the_typed_error() -> None:
    rng = random.Random(6)
    base = _select([_family("volatility"), _family("ranking")])
    crashes: list[str] = []
    for index in range(200):
        doc = copy.deepcopy(base)
        _mutate(doc, rng)
        raw = json.dumps(doc)
        client = FamilyClient(lambda _argv, raw=raw: (0, raw, ""))
        try:
            client.select("volatility")
        except FamilyClientError:
            continue
        except Exception as exc:
            crashes.append(f"{index}:{type(exc).__name__}:{exc}")
    assert crashes == []


def _mutate(doc: dict[str, object], rng: random.Random) -> None:
    op = rng.randrange(8)
    if op == 0:
        doc["schema"] = rng.choice([None, 1, "evil", SURFACE_SCHEMA, ""])
    elif op == 1:
        doc["selected"] = rng.choice([None, {}, "x", [], [_family(executable=False)]])
    elif op == 2:
        doc["exclusions"] = rng.choice([None, "bad", [{"id": 1}], [{"id": "tail"}]])
    elif op == 3:
        doc["injected"] = {"argv": ["sh", "-c", "true"]}
    elif op == 4:
        selected = doc["selected"]
        if isinstance(selected, list) and selected and isinstance(selected[0], dict):
            selected[0]["qualification"] = rng.choice(
                ["retired", "required", None, "live_optional"]
            )
    elif op == 5:
        selected = doc["selected"]
        if isinstance(selected, list) and selected and isinstance(selected[0], dict):
            selected[0]["id"] = rng.choice(["Volatility", "vol;rm", "a" * 5000, "volatility", 3])
    elif op == 6:
        selected = doc["selected"]
        if isinstance(selected, list) and selected and isinstance(selected[0], dict):
            selected[0]["schema"] = "dipcatcher.family_surface.v0"
    else:
        doc["goal"] = "volatility\n$(reboot)"
