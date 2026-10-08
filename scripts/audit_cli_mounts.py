#!/usr/bin/env python
"""Detect dropped wiring: Typer apps defined in ``src/`` but never mounted.

Why this exists
---------------
``src/quant_fund/cli/sota_cmds.py`` defined three complete operator groups
(17 subcommands, including ``forward-shadow freeze`` — the command
``docs/INSTITUTIONAL_READINESS.md`` names as the mechanism for closing
institutional condition 3) while ``cli/_app.py`` mounted 17 other sub-apps and
not that one. The module imported fine; it simply was not reachable from the
CLI. ``tests/unit/cli/test_sota_cmds.py`` was red the whole time, which is what
finally surfaced it.

That is a *lost mount*, and it is the same failure shape as a security fix
dropped by a bad merge: correct code, correct tests, missing wiring. This script
makes the class mechanically checkable instead of something you only notice
when a test goes red.

Legitimate exceptions (all verified by hand on 2026-10-08):

- ``src/fx1/cli.py`` and ``src/fx1/interactive/app.py`` — fx1 ships its own
  console scripts (``fx1``, ``fxi``). ``AGENTS.md`` and
  ``tests/unit/test_fx1_dependency_edge.py`` forbid the harness importing fx1,
  so these are correct.
- ``src/quant_fund/cli/_app.py`` — this *is* the root ``app`` other groups are
  mounted onto.
- ``src/quant_fund/mc_engine/cli.py`` — ``pyproject.toml`` declares
  ``mc-engine = "quant_fund.mc_engine.cli:app"``, so it has its own entry point.

Usage::

    uv run python scripts/audit_cli_mounts.py            # human-readable
    uv run python scripts/audit_cli_mounts.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Typer apps that are *correctly* not mounted into the `dipcatcher` root app.
# Each entry records why, so the exemption is auditable rather than a blanket skip.
EXPECTED_UNMOUNTED: dict[str, str] = {
    "app": (
        "either the fx1 package (own console scripts `fx1`/`fxi`; the harness must "
        "not import fx1) or the root app in cli/_app.py itself"
    ),
}


def _is_typer_construction(node: ast.AST | None) -> bool:
    """True for ``typer.Typer(...)`` — an attribute call, not a bare name."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Typer"
    )


def _display(path: Path) -> str:
    """Repo-relative when possible; absolute otherwise.

    Tests scan ``tmp_path``, which is not under ROOT, so a bare
    ``relative_to`` would raise ValueError and abort the audit.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def defined_typer_apps(src: Path = SRC) -> list[tuple[str, str]]:
    """Return ``[(name, path)]`` for every module-level Typer app under *src*."""
    found: list[tuple[str, str]] = []
    for path in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and _is_typer_construction(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.append((target.id, _display(path)))
            elif (
                isinstance(node, ast.AnnAssign)
                and _is_typer_construction(node.value)
                and isinstance(node.target, ast.Name)
            ):
                found.append((node.target.id, _display(path)))
    return found


def mounted_names(src: Path = SRC) -> set[str]:
    """Return every app object passed as the first arg to ``add_typer(...)``."""
    names: set[str] = set()
    for path in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_typer"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                names.add(node.args[0].id)
    return names


def audit(src: Path = SRC) -> dict[str, Any]:
    defined = defined_typer_apps(src)
    mounted = mounted_names(src)
    unmounted = [
        {"name": n, "file": f, "expected": n in EXPECTED_UNMOUNTED}
        for n, f in defined
        if n not in mounted
    ]
    unexpected = [u for u in unmounted if not u["expected"]]
    return {
        "schema": "cli_mount_audit.v1",
        "defined": len(defined),
        "mounted": len(mounted),
        "unmounted": unmounted,
        "verdict": "PASS" if not unexpected else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_cli_mounts.py",
        description="Detect Typer apps defined in src/ but never mounted into a Typer app.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = audit()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"CLI mount audit — {report['verdict']}")
        print(f"  typer apps defined : {report['defined']}")
        print(f"  apps mounted       : {report['mounted']}")
        print(f"  defined, unmounted : {len(report['unmounted'])}")
        for entry in report["unmounted"]:
            tag = "expected" if entry["expected"] else "UNEXPECTED"
            print(f"    [{tag}] {entry['name']}  ({entry['file']})")
        if report["verdict"] != "PASS":
            print(
                "\n  An app defined here but mounted nowhere is unreachable. Either\n"
                "  mount it with add_typer(), or declare it in EXPECTED_UNMOUNTED\n"
                "  with a reason so the exemption stays auditable."
            )
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
