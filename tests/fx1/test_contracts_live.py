"""The fx1 contract client must parse what the lab command really prints.

``test_contracts.py`` pins the client against fixtures; this test pins the two
halves together so a producer/consumer drift (for example ``"default": null``
on an optional option) cannot ship unnoticed.
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 — fixed argv, the lab command is the system under test
import sys
from pathlib import Path

import pytest

from fx1.interactive.contracts import CapabilityCatalog

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def live_document() -> dict[str, object]:
    env = {**os.environ, "PYTHONPATH": str(_ROOT / "src"), "NO_COLOR": "1"}
    proc = subprocess.run(  # noqa: S603 — argv is a fixed local list
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "capabilities",
            "list",
            "--json",
            "--executable-only",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    document: dict[str, object] = json.loads(proc.stdout)
    return document


def test_client_parses_the_live_executable_capability_contract(
    live_document: dict[str, object],
) -> None:
    catalog = CapabilityCatalog.from_json(live_document)
    assert {"score-forecasts", "vol-bench", "verify-receipt", "research"} <= set(
        catalog.capabilities
    )
    assert all(cap.executable for cap in catalog.capabilities.values())


def test_optional_options_without_a_default_parse_as_having_none(
    live_document: dict[str, object],
) -> None:
    catalog = CapabilityCatalog.from_json(live_document)
    cap = catalog.get("score-forecasts")
    assert cap is not None
    by_name = {arg.name: arg for arg in cap.args}
    assert by_name["--forecast-col"].has_default is False
    assert by_name["--time-col"].has_default is True
    assert by_name["--data"].required is True
