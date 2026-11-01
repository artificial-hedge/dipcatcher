"""Tests for the shared helpers in the ``scripts/*_real_drill.py`` family.

Ten drill scripts share two helpers, and both are load-bearing for the
receipt chain:

* ``_atomic_write_text`` — a receipt must never be half-written. The helper
  writes to a sibling temporary file and then ``os.replace``s it into place,
  which is atomic on the same filesystem. If it wrote in place, a crash
  mid-write would leave a truncated receipt that later looks valid.
* ``_seal`` — the receipt self-seal. It strips the volatile fields
  (``code_revision``, ``meta``) *before* hashing, so a receipt produced on a
  different machine or commit still verifies, and the written file must
  verify against the repo's own receipt verifier.

The drills read ``sys.argv`` at import time, so the import here runs with a
clean argument vector. Every write lands in ``tmp_path``; the repository's
``receipts/`` directory is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

#: Every drill script that ships the atomic writer.
_ATOMIC_WRITERS = (
    "calib_real_drill",
    "conformal_real_drill",
    "coverage_cs_real_drill",
    "coverage_real_drill",
    "cp_real_drill",
    "drift_real_drill",
    "loss_cs_real_drill",
    "mcs_real_drill",
    "panel_audit_drill",
    "tail_real_drill",
)

#: The subset that also ships the receipt sealer, keyed by the sealer's
#: signature: ``(report, out_path)`` for most, ``(report, name)`` for the
#: conformal drill, which derives the file name from ``OUT_DIR``.
_PATH_SEALERS = (
    "cp_real_drill",
    "drift_real_drill",
    "loss_cs_real_drill",
)
_NAME_SEALER = "conformal_real_drill"


@pytest.fixture(autouse=True)
def _clean_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The drills parse argv at import time; keep pytest's flags out of it."""
    monkeypatch.setattr(sys, "argv", ["drill-probe"])


def _load(name: str):
    """Import one drill script under a private module name."""
    spec = importlib.util.spec_from_file_location(f"gap_drill_{name}", _SCRIPTS / f"{name}.py")
    assert spec and spec.loader, f"{name}.py must be importable"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", _ATOMIC_WRITERS)
def test_the_atomic_writer_is_present(name: str) -> None:
    assert callable(getattr(_load(name), "_atomic_write_text", None))


@pytest.mark.parametrize("name", _ATOMIC_WRITERS)
def test_the_atomic_writer_writes_the_exact_text(name: str, tmp_path: Path) -> None:
    module = _load(name)
    target = tmp_path / "receipt.json"
    module._atomic_write_text(target, '{"a": 1}\n')
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'


@pytest.mark.parametrize("name", _ATOMIC_WRITERS)
def test_the_atomic_writer_replaces_an_existing_file(name: str, tmp_path: Path) -> None:
    module = _load(name)
    target = tmp_path / "receipt.json"
    target.write_text("old content that must not survive", encoding="utf-8")
    module._atomic_write_text(target, "new content")
    assert target.read_text(encoding="utf-8") == "new content"


@pytest.mark.parametrize("name", _ATOMIC_WRITERS)
def test_the_atomic_writer_leaves_no_temporary_file_behind(name: str, tmp_path: Path) -> None:
    """The staged file must be renamed into place, never left next to it."""
    module = _load(name)
    target = tmp_path / "receipt.json"
    module._atomic_write_text(target, "content")
    assert [entry.name for entry in tmp_path.iterdir()] == ["receipt.json"]


@pytest.mark.parametrize("name", _ATOMIC_WRITERS)
def test_the_atomic_writer_stages_in_the_target_directory(name: str, tmp_path: Path) -> None:
    """``os.replace`` is only atomic within one filesystem."""
    module = _load(name)
    target = tmp_path / "receipt.json"
    module._atomic_write_text(target, "content")
    # The only file that exists is the target: the staged file lived beside it.
    assert all(entry.parent == tmp_path for entry in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# Receipt seals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", _PATH_SEALERS)
def test_the_sealer_writes_a_self_sealed_receipt(name: str, tmp_path: Path) -> None:
    module = _load(name)
    out = tmp_path / f"{name}.json"
    report = {"schema": "x.v1", "kind": "k", "value": 1.0}
    result = module._seal(dict(report), out)
    assert result["valid"] is True
    assert result["errors"] == []
    payload = json.loads(out.read_text())
    assert payload["receipt_sha256"]
    assert payload["schema"] == "x.v1"


@pytest.mark.parametrize("name", _PATH_SEALERS)
def test_the_sealer_strips_the_volatile_fields_before_hashing(name: str, tmp_path: Path) -> None:
    """``code_revision`` and ``meta`` change between runs and must not be sealed."""
    module = _load(name)
    out = tmp_path / f"{name}.json"
    report = {
        "schema": "x.v1",
        "kind": "k",
        "value": 1.0,
        "code_revision": "abcdef",
        "meta": {"machine": "one"},
    }
    module._seal(dict(report), out)
    payload = json.loads(out.read_text())
    assert "code_revision" not in payload
    assert "meta" not in payload


@pytest.mark.parametrize("name", _PATH_SEALERS)
def test_the_sealed_receipt_verifies_against_the_repo_verifier(name: str, tmp_path: Path) -> None:
    module = _load(name)
    out = tmp_path / f"{name}.json"
    module._seal({"schema": "x.v1", "kind": "k", "value": 1.0}, out)
    verification = module.verify_receipt_file(out)
    assert verification["valid"] is True
    assert verification["errors"] == []


@pytest.mark.parametrize("name", _PATH_SEALERS)
def test_a_sealed_receipt_writes_sorted_keys_and_a_trailing_newline(
    name: str,
    tmp_path: Path,
) -> None:
    module = _load(name)
    out = tmp_path / f"{name}.json"
    module._seal({"schema": "x.v1", "kind": "k", "value": 1.0}, out)
    raw = out.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("name", _PATH_SEALERS)
def test_changing_a_sealed_field_invalidates_the_receipt(name: str, tmp_path: Path) -> None:
    """The seal must actually bind the content it describes."""
    module = _load(name)
    out = tmp_path / f"{name}.json"
    module._seal({"schema": "x.v1", "kind": "k", "value": 1.0}, out)
    payload = json.loads(out.read_text())
    payload["value"] = 999.0
    out.write_text(json.dumps(payload), encoding="utf-8")
    assert module.verify_receipt_file(out)["valid"] is False


# --------------------------------------------------------------------------- #
# The name-addressed sealer
# --------------------------------------------------------------------------- #


def _seal_by_name(module, name: str, tmp_path: Path) -> Path:
    """Seal through the ``OUT_DIR``-addressed variant and return the file."""
    module.OUT_DIR = tmp_path
    module._seal({"schema": "x.v1", "kind": "k", "value": 1.0}, name)
    return tmp_path / f"conformal_real_drill_{name}.json"


def test_the_name_addressed_sealer_writes_into_its_output_dir(tmp_path: Path) -> None:
    module = _load(_NAME_SEALER)
    out = _seal_by_name(module, "probe", tmp_path)
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert payload["receipt_sha256"]
    assert payload["schema"] == "x.v1"


def test_the_name_addressed_sealer_strips_the_volatile_fields(tmp_path: Path) -> None:
    module = _load(_NAME_SEALER)
    module.OUT_DIR = tmp_path
    module._seal(
        {
            "schema": "x.v1",
            "kind": "k",
            "value": 1.0,
            "code_revision": "abcdef",
            "meta": {"machine": "one"},
        },
        "probe",
    )
    payload = json.loads((tmp_path / "conformal_real_drill_probe.json").read_text())
    assert "code_revision" not in payload
    assert "meta" not in payload


def test_the_name_addressed_sealer_verifies_against_the_repo_verifier(tmp_path: Path) -> None:
    module = _load(_NAME_SEALER)
    out = _seal_by_name(module, "probe", tmp_path)
    assert module.verify_receipt_file(out)["valid"] is True


def test_the_name_addressed_sealer_leaves_no_temporary_file(tmp_path: Path) -> None:
    module = _load(_NAME_SEALER)
    _seal_by_name(module, "probe", tmp_path)
    assert [entry.name for entry in tmp_path.iterdir()] == ["conformal_real_drill_probe.json"]
