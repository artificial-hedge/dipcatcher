"""Tests for ``scripts/audit_cli_mounts.py``.

The audit exists because ``cli/sota_cmds.py`` defined three complete operator
groups — including the ``forward-shadow freeze`` command the readiness matrix
names for closing institutional condition 3 — while ``cli/_app.py`` never
mounted them. The module imported cleanly, so nothing failed loudly until its
37 tests went red.

These tests pin both halves of that guarantee: an unmounted app must be
*detected* (so the bug class cannot reappear silently), and the currently
legitimate exceptions must stay declared (so the audit does not rot into a
blanket suppression).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
_AUDIT_PATH = ROOT / "scripts" / "audit_cli_mounts.py"
_spec = importlib.util.spec_from_file_location("audit_cli_mounts", _AUDIT_PATH)
assert _spec and _spec.loader, "audit_cli_mounts.py must be importable"
audit_mod = importlib.util.module_from_spec(_spec)
sys.modules["audit_cli_mounts"] = audit_mod
_spec.loader.exec_module(audit_mod)


def _write_module(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_detects_a_typer_app_that_is_never_mounted(tmp_path: Path) -> None:
    """The core regression: defined-but-unreachable must be reported."""
    _write_module(
        tmp_path,
        "orphan_cmds.py",
        "import typer\norphan_app = typer.Typer(help='x')\n",
    )
    report = audit_mod.audit(src=tmp_path)
    assert report["verdict"] == "FAIL"
    assert [u["name"] for u in report["unmounted"]] == ["orphan_app"]


def test_mounted_app_is_not_reported(tmp_path: Path) -> None:
    _write_module(
        tmp_path,
        "mounted_cmds.py",
        "import typer\nmounted_app = typer.Typer(help='x')\n",
    )
    # The mount target itself is an app that is (correctly) never mounted, so
    # assert specifically about the child rather than about the whole list.
    _write_module(
        tmp_path,
        "root.py",
        "import typer\nfrom mounted_cmds import mounted_app\n"
        "app = typer.Typer()\napp.add_typer(mounted_app, name='mounted')\n",
    )
    report = audit_mod.audit(src=tmp_path)
    names = [u["name"] for u in report["unmounted"]]
    assert "mounted_app" not in names
    assert names == ["app"]  # only the fixture's own root app
    assert report["verdict"] == "PASS"  # `app` is a declared exemption


def test_expected_exemption_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared exemption keeps the audit green but stays visible."""
    monkeypatch.setitem(audit_mod.EXPECTED_UNMOUNTED, "app", "own entry point")
    _write_module(tmp_path, "x.py", "import typer\napp = typer.Typer()\n")
    report = audit_mod.audit(src=tmp_path)
    assert report["verdict"] == "PASS"
    assert len(report["unmounted"]) == 1
    assert report["unmounted"][0]["expected"] is True


def test_unparseable_module_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A syntax error in one module must not abort the scan."""
    (tmp_path / "broken.py").write_text("def (:", encoding="utf-8")
    _write_module(tmp_path, "ok.py", "import typer\nfine_app = typer.Typer()\n")
    report = audit_mod.audit(src=tmp_path)
    assert [u["name"] for u in report["unmounted"]] == ["fine_app"]


def test_non_typer_call_is_not_treated_as_an_app(tmp_path: Path) -> None:
    """``typer.Typer`` is an attribute call — a bare ``Typer()`` must not match."""
    _write_module(tmp_path, "y.py", "from typer import Typer\nthing = Typer(help='x')\n")
    _write_module(
        tmp_path,
        "z.py",
        "import typer\nother = typer.Opt(help='x')\n",
    )
    assert audit_mod.audit(src=tmp_path)["unmounted"] == []


def test_real_repo_cli_surface_has_no_unexpected_unmounted_app() -> None:
    """The live tree must be clean, and the exemptions must all be declared."""
    report = audit_mod.audit()
    assert report["verdict"] == "PASS", report["unmounted"]
    for entry in report["unmounted"]:
        assert entry["name"] in audit_mod.EXPECTED_UNMOUNTED, entry


def test_main_emits_one_json_document(capsys: pytest.CaptureFixture[str]) -> None:
    """--json must emit exactly one parseable document (the real repo passes)."""
    code = audit_mod.main(["--json"])
    out = capsys.readouterr().out
    payload: dict[str, Any] = __import__("json").loads(out)
    assert payload["schema"] == "cli_mount_audit.v1"
    assert code == (0 if payload["verdict"] == "PASS" else 1)


def test_ast_helper_is_robust() -> None:
    """``_is_typer_construction`` must not raise on arbitrary nodes."""
    for node in (
        None,
        ast.parse("x = 1").body[0],
        ast.parse("import typer\napp = typer.Typer()").body[1],
    ):
        assert isinstance(audit_mod._is_typer_construction(node), bool)
