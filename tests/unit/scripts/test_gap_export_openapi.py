"""Tests for ``scripts/export_fx1_api_openapi.py``.

The exporter writes the fx-1 harness OpenAPI schema to a committed file so
the generated TypeScript client can be rebuilt from it. Two properties
matter:

* the exported schema is the *live* app schema — the script must not hand-
  assemble a document that can drift from the routes the server actually
  serves;
* the written file is deterministic (sorted keys, stable indentation) so a
  rebuild produces a byte-identical artifact and the diff is reviewable.

The output goes to ``tmp_path``; nothing is written into the repository.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "export_fx1_api_openapi.py"
_spec = importlib.util.spec_from_file_location("gap_export_fx1_api_openapi", _SCRIPT_PATH)
assert _spec and _spec.loader, "export_fx1_api_openapi.py must be importable"
mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_export_fx1_api_openapi"] = mod
_spec.loader.exec_module(mod)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the exporter once into a temporary directory."""
    out = tmp_path_factory.mktemp("openapi") / "openapi.json"
    argv = sys.argv
    sys.argv = ["export_fx1_api_openapi.py", str(out)]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    return out


def test_the_exporter_writes_a_parseable_openapi_document(exported: Path) -> None:
    spec = json.loads(exported.read_text())
    assert spec["openapi"].startswith("3.")


def test_the_exported_schema_describes_the_live_routes(exported: Path) -> None:
    """The document must match ``create_app().openapi()``, not a hand copy."""
    from fx1.serve.api import create_app

    spec = json.loads(exported.read_text())
    live = create_app().openapi()
    assert set(spec["paths"]) == set(live["paths"])
    assert set(spec["paths"]) >= {"/harness/runs", "/harness/jobs", "/harness/complete"}


def test_the_exported_file_is_deterministic(exported: Path) -> None:
    """Keys are sorted and the file ends with a newline — rebuild-stable."""
    raw = exported.read_text()
    assert raw.endswith("\n")
    spec = json.loads(raw)
    assert raw == json.dumps(spec, indent=2, sort_keys=True) + "\n"


def test_the_exporter_creates_the_output_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "openapi.json"
    argv = sys.argv
    sys.argv = ["export_fx1_api_openapi.py", str(out)]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    assert out.is_file()
    assert json.loads(out.read_text())["openapi"].startswith("3.")
