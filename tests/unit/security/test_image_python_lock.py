"""The image lock check fails closed on drift and accepts a matching freeze."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_image_python_lock.py"
_SPEC = importlib.util.spec_from_file_location("check_image_python_lock", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOCK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOCK)

_LOCK_TEXT = """
[[package]]
name = "fx-1"
source = { editable = "." }

[[package]]
name = "Demo_Pkg"
version = "1.2.3"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "dev-only"
version = "9.9.9"
source = { registry = "https://pypi.org/simple" }
"""


def test_matching_image_has_no_mismatches() -> None:
    locked, local = _LOCK.locked_versions(_LOCK_TEXT)
    required = _LOCK.parse_requirements("demo-pkg==1.2.3\n")
    assert local == {"fx-1"}
    assert locked["demo-pkg"] == "1.2.3"
    problems = _LOCK.lock_mismatches(
        {"fx-1": "0.0.0", "Demo.Pkg": "1.2.3"},
        locked,
        local,
        required,
    )
    assert problems == []


def test_version_drift_unknown_dist_and_missing_requirement_fail() -> None:
    locked, local = _LOCK.locked_versions(_LOCK_TEXT)
    required = _LOCK.parse_requirements("demo-pkg==1.2.3\n")
    problems = _LOCK.lock_mismatches(
        {"demo-pkg": "1.2.4", "extra": "0.1.0"},
        locked,
        local,
        required,
    )
    assert problems == [
        "image is missing the local project fx-1",
        "demo-pkg image=1.2.4 lock=1.2.3",
        "image has extra==0.1.0 which is not in uv.lock",
    ]


def test_installed_dev_package_is_outside_the_runtime_export() -> None:
    locked, local = _LOCK.locked_versions(_LOCK_TEXT)
    required = _LOCK.parse_requirements("demo-pkg==1.2.3\n")
    problems = _LOCK.lock_mismatches(
        {"fx-1": "9", "demo-pkg": "1.2.3", "dev-only": "9.9.9"},
        locked,
        local,
        required,
    )
    assert problems == ["image has dev-only==9.9.9 outside the runtime lock export"]


def test_false_environment_markers_are_skipped() -> None:
    from packaging.markers import Marker

    required = _LOCK.parse_requirements("demo-pkg==1.2.3\npywin32==312 ; sys_platform == 'win32'\n")
    assert required["demo-pkg"] == "1.2.3"
    windows_only = Marker("sys_platform == 'win32'").evaluate()
    assert ("pywin32" in required) is windows_only


def test_dev_only_lock_entries_are_not_required() -> None:
    locked, local = _LOCK.locked_versions(_LOCK_TEXT)
    required = _LOCK.parse_requirements("# runtime\ndemo-pkg==1.2.3 ; python_version >= '3.12'\n")
    problems = _LOCK.lock_mismatches({"fx-1": "9", "demo-pkg": "1.2.3"}, locked, local, required)
    assert "dev-only" not in problems
    assert problems == []
