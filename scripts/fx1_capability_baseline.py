#!/usr/bin/env python3
"""Phase 0 baseline snapshot for the fx-1 independent-capability inventory.

Implements Phase 0 of ``docs/ULTRA_INTENSE_CAPABILITY_PURSUIT_PLAN.md``: recompute
IDs, distinct source paths, per-kind counts and all three LOC measures from the
*working tree* (tracked **and** untracked), reconcile them against
``src/fx1/operations/registry.py`` and the progress/guide docs, classify every
mismatch, and emit a per-capability verification matrix.

Read-only. It writes only to the paths given by ``--output-json`` /
``--output-markdown`` and never touches the repository otherwise.

LOC definitions are **not** re-invented here: physical / comment / docstring /
code-token lines come from ``scripts/code_quality_inventory.measure_file``, the
repo's canonical inventory tool, so the two cannot drift. Only ``nonblank_lines``
is computed locally, because the canonical tool does not expose it.

Counting rules (frozen — see ULTRA_INTENSE_CAPABILITY_PURSUIT_PLAN.md rule 5):

* **physical_lines** — ``len(text.splitlines())``, comments and docstrings included.
* **nonblank_lines** — physical lines that are not whitespace-only.
* **code_token_lines** — lines carrying a real token, excluding blanks, comments
  and AST-identified docstrings (``measure_file``'s ``source_lines``).
* **accepted capability** — a module exporting ``OPERATION`` whose ``id`` is a key
  of ``registry._IMPLEMENTATIONS``. Orphan modules (export ``OPERATION``, absent
  from the registry) are reported separately and are **never** counted as accepted:
  the harness cannot discover, describe or execute them.
* **infrastructure** — ``__init__.py``, ``base.py``, ``registry.py``, ``_numeric.py``
  and any module exporting no ``OPERATION``. Excluded from every LOC total, matching
  the progress ledger's stated scope.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OPERATIONS_DIR = ROOT / "src" / "fx1" / "operations"
EXTENSIONS_DIR = ROOT / "src" / "fx1" / "extensions"
TESTS_DIR = ROOT / "tests" / "fx1"

INFRASTRUCTURE_STEMS = frozenset({"__init__", "base", "registry", "_numeric"})

# Doc claims to reconcile against. Parsed from source at runtime so this table
# cannot silently rot when the docs are edited.
DOC_CLAIM_SOURCES = (
    ("docs/FX1_CAPABILITY_PROGRESS.md", "progress ledger"),
    ("docs/FX1_CAPABILITIES.md", "capabilities overview"),
    ("docs/FX1_OPERATIONS.md", "operation guide"),
)


def _load_canonical_measurer():
    """Import measure_file from the repo's canonical inventory tool."""
    path = ROOT / "scripts" / "code_quality_inventory.py"
    spec = importlib.util.spec_from_file_location("code_quality_inventory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical inventory tool from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: the canonical tool uses @dataclass, and
    # dataclasses._is_type resolves cls.__module__ through sys.modules. Without
    # this the import dies with AttributeError: 'NoneType' has no '__dict__'.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module.measure_file


MEASURE_FILE = _load_canonical_measurer()


@dataclass(frozen=True)
class FileMeasure:
    """The three frozen LOC measures plus the canonical tool's own fields."""

    path: str
    physical_lines: int
    nonblank_lines: int
    code_token_lines: int
    comment_lines: int
    docstring_lines: int
    functions: int
    classes: int


@dataclass(frozen=True)
class Capability:
    operation_id: str | None
    kind: str | None
    version: str | None
    stem: str
    path: str
    classification: str  # registered | orphan | infrastructure
    has_invocation_case: bool
    measure: FileMeasure


def _nonblank_lines(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if line.strip())


def measure(path: Path) -> FileMeasure:
    """Canonical measurement + the one field the canonical tool omits."""
    canonical = MEASURE_FILE(path, repo=ROOT)
    return FileMeasure(
        path=canonical.path,
        physical_lines=canonical.physical_lines,
        nonblank_lines=_nonblank_lines(path),
        code_token_lines=canonical.source_lines,
        comment_lines=canonical.comment_lines,
        docstring_lines=canonical.docstring_lines,
        functions=canonical.functions,
        classes=canonical.classes,
    )


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def operation_version_default() -> str | None:
    """The ``version`` default declared on the ``Operation`` dataclass.

    Resolved from source rather than hardcoded, so the matrix stays truthful if
    the default ever changes. Modules that omit ``version`` inherit it.
    """
    base = OPERATIONS_DIR / "base.py"
    tree = ast.parse(base.read_text(encoding="utf-8"), filename=str(base))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Operation":
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "version"
                and statement.value is not None
            ):
                value = _literal(statement.value)
                return value if isinstance(value, str) else None
    return None


VERSION_DEFAULT = operation_version_default()


def parse_operation(path: Path) -> dict[str, Any] | None:
    """AST-extract the module-level ``OPERATION = Operation(...)`` metadata.

    Parsing rather than importing keeps this snapshot read-only, deterministic and
    free of import side effects across ~170 modules.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets if isinstance(t, ast.Name) and t.id == "OPERATION"]
        if not targets or not isinstance(node.value, ast.Call):
            continue
        fields: dict[str, Any] = {}
        for keyword in node.value.keywords:
            if keyword.arg in {"id", "kind", "version", "description"}:
                fields[keyword.arg] = _literal(keyword.value)
        if "id" in fields:
            # Modules that omit `version` inherit the dataclass default; report the
            # effective value rather than a misleading blank.
            fields.setdefault("version", VERSION_DEFAULT)
            fields["version_inherited"] = "version" not in {k.arg for k in node.value.keywords}
            return fields
    return None


def registry_ids() -> dict[str, str]:
    """Authoritative registration table, imported from the registry itself."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from fx1.operations.registry import _IMPLEMENTATIONS  # noqa: PLC0415

        return dict(_IMPLEMENTATIONS)
    finally:
        sys.path.pop(0)


def invocation_case_ids() -> set[str]:
    """Operation ids with a hand-checked invocation case (the Phase 1 evidence)."""
    ids: set[str] = set()
    sources = sorted(TESTS_DIR.glob("test_operations_registry_invocation.py")) + sorted(
        TESTS_DIR.glob("operation_cases_chunk*.py")
    )
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id]
                if isinstance(node.target, ast.Name)
                else []
            )
            if "CASES" not in names or not isinstance(node.value, ast.Dict):
                continue
            for key in node.value.keys:
                literal = _literal(key) if key is not None else None
                if isinstance(literal, str):
                    ids.add(literal)
    return ids


def scan_operations() -> list[Capability]:
    cases = invocation_case_ids()
    registered = registry_ids()
    registered_stems = set(registered.values())
    capabilities: list[Capability] = []

    for path in sorted(OPERATIONS_DIR.glob("*.py")):
        stem = path.stem
        operation = None if stem in INFRASTRUCTURE_STEMS else parse_operation(path)
        if operation is None:
            classification = "infrastructure"
            operation_id = kind = version = None
        else:
            operation_id = operation.get("id")
            kind = operation.get("kind")
            version = operation.get("version")
            if operation_id in registered and registered[operation_id] == stem:
                classification = "registered"
            elif stem in registered_stems:
                # Registered under a different id than the module declares: a real
                # inconsistency, reported as registered so it surfaces in the matrix.
                classification = "registered"
            else:
                classification = "orphan"
        capabilities.append(
            Capability(
                operation_id=operation_id,
                kind=kind,
                version=version,
                stem=stem,
                path=path.relative_to(ROOT).as_posix(),
                classification=classification,
                has_invocation_case=bool(operation_id and operation_id in cases),
                measure=measure(path),
            )
        )
    return capabilities


def scan_extensions() -> dict[str, Any]:
    """Extension tree shape: hand-written core vs generated bindings."""
    if not EXTENSIONS_DIR.is_dir():
        return {"present": False}
    core: list[str] = []
    generated: dict[str, int] = Counter()
    total = FileMeasure("", 0, 0, 0, 0, 0, 0, 0)
    for path in sorted(EXTENSIONS_DIR.rglob("*.py")):
        relative = path.relative_to(EXTENSIONS_DIR)
        measure_ = measure(path)
        total = FileMeasure(
            "",
            total.physical_lines + measure_.physical_lines,
            total.nonblank_lines + measure_.nonblank_lines,
            total.code_token_lines + measure_.code_token_lines,
            total.comment_lines + measure_.comment_lines,
            total.docstring_lines + measure_.docstring_lines,
            total.functions + measure_.functions,
            total.classes + measure_.classes,
        )
        if relative.parent == Path(".") and relative.stem in {
            "__init__",
            "contracts",
            "registry",
            "naming",
            "feature_catalog",
        }:
            core.append(relative.as_posix())
        else:
            generated[relative.parent.as_posix() or "."] += 1
    return {
        "present": True,
        "hand_written_core": core,
        "generated_bindings_by_subpackage": dict(sorted(generated.items())),
        "generated_binding_count": sum(generated.values()),
        "total_files": len(core) + sum(generated.values()),
        "loc": {
            "physical_lines": total.physical_lines,
            "nonblank_lines": total.nonblank_lines,
            "code_token_lines": total.code_token_lines,
        },
    }


def totals(capabilities: list[Capability], classification: str) -> dict[str, int]:
    selected = [c for c in capabilities if c.classification == classification]
    return {
        "files": len(selected),
        "physical_lines": sum(c.measure.physical_lines for c in selected),
        "nonblank_lines": sum(c.measure.nonblank_lines for c in selected),
        "code_token_lines": sum(c.measure.code_token_lines for c in selected),
        "comment_lines": sum(c.measure.comment_lines for c in selected),
        "docstring_lines": sum(c.measure.docstring_lines for c in selected),
    }


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    porcelain = run("status", "--porcelain")
    lines = porcelain.splitlines()
    return {
        "head": run("rev-parse", "HEAD"),
        "head_short": run("rev-parse", "--short=10", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "staged": sum(1 for line in lines if line[:1] not in {" ", "?"} and line.strip()),
        "modified": sum(1 for line in lines if line.startswith(" M")),
        "untracked": sum(1 for line in lines if line.startswith("??")),
    }


def parse_doc_claims() -> dict[str, Any]:
    """Extract the numeric claims the docs currently make, with line numbers."""
    import re

    claims: dict[str, Any] = {}
    patterns = {
        "implementations": re.compile(r"\*\*(\d[\d,]*) (?:implementations|operations)\*\*"),
        "separate_files": re.compile(r"(\d[\d,]*)\s+separate files"),
        "physical_lines": re.compile(r"\*\*([\d,]+) physical"),
        "nonblank_lines": re.compile(r"([\d,]+) nonblank"),
        "code_token_lines": re.compile(r"\*\*([\d,]+) lines contain\s*\n?\s*code tokens?\*\*"),
        "features": re.compile(r"(\d+)\s+(?:numeric\s+)?features"),
        "skills": re.compile(r"(\d+)\s+(?:data-audit[^\s]*\s+)?skills"),
        "plugins": re.compile(r"(\d+)\s+(?:workspace data\s+)?plugins"),
        "remaining": re.compile(r"\*\*([\d,]+) implementations remain\*\*"),
    }
    for relative, label in DOC_CLAIM_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        found: dict[str, Any] = {}
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                line_no = text[: match.start()].count("\n") + 1
                found[name] = {"value": int(match.group(1).replace(",", "")), "line": line_no}
        if found:
            claims[relative] = {"label": label, "claims": found}
    return claims


def build_report() -> dict[str, Any]:
    capabilities = scan_operations()
    registered = [c for c in capabilities if c.classification == "registered"]
    orphans = [c for c in capabilities if c.classification == "orphan"]
    infrastructure = [c for c in capabilities if c.classification == "infrastructure"]

    kind_counts = Counter(c.kind for c in registered if c.kind)
    orphan_kind_counts = Counter(c.kind for c in orphans if c.kind)

    registry = registry_ids()
    registered_ids = {c.operation_id for c in registered if c.operation_id}
    missing_source = sorted(set(registry) - registered_ids)
    unregistered_ids = sorted({c.operation_id for c in orphans if c.operation_id} - set(registry))

    cases = invocation_case_ids()
    without_case = sorted(registered_ids - cases)
    stale_cases = sorted(cases - registered_ids)

    return {
        "schema_version": "fx1-capability-baseline/v1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": "docs/ULTRA_INTENSE_CAPABILITY_PURSUIT_PLAN.md",
        "phase": "Phase 0 — protect the baseline and freeze the counter",
        "git": git_snapshot(),
        "counting_rules": {
            "physical_lines": "len(text.splitlines()); comments and docstrings included",
            "nonblank_lines": "physical lines that are not whitespace-only",
            "code_token_lines": (
                "lines carrying a real token, excluding blanks, comments and "
                "AST-identified docstrings (canonical tool's source_lines)"
            ),
            "accepted_capability": (
                "module exporting OPERATION whose id is a key of registry._IMPLEMENTATIONS"
            ),
            "orphan": (
                "module exporting OPERATION that is absent from _IMPLEMENTATIONS; "
                "unreachable via the harness and never counted as accepted"
            ),
            "loc_scope": (
                "registered implementation files only; infrastructure, tests, "
                "documentation and generated artifacts excluded — matching the "
                "progress ledger's stated scope"
            ),
            "measurement_source": "scripts/code_quality_inventory.measure_file (reused, not reimplemented)",
        },
        "operations_directory": {
            "path": OPERATIONS_DIR.relative_to(ROOT).as_posix(),
            "total_python_files": len(capabilities),
            "registered": len(registered),
            "orphans": len(orphans),
            "infrastructure": len(infrastructure),
            "infrastructure_stems": sorted(c.stem for c in infrastructure),
        },
        "registry": {
            "implementation_count": len(registry),
            "per_kind": dict(sorted(kind_counts.items())),
            "orphan_per_kind": dict(sorted(orphan_kind_counts.items())),
            "ids_missing_a_source_file": missing_source,
            "source_files_not_registered": unregistered_ids,
        },
        "loc": {
            "registered_only": totals(capabilities, "registered"),
            "orphans_only": totals(capabilities, "orphan"),
            "all_operation_modules": {
                "files": len(registered) + len(orphans),
                "physical_lines": (
                    totals(capabilities, "registered")["physical_lines"]
                    + totals(capabilities, "orphans")["physical_lines"]
                ),
                "nonblank_lines": (
                    totals(capabilities, "registered")["nonblank_lines"]
                    + totals(capabilities, "orphans")["nonblank_lines"]
                ),
                "code_token_lines": (
                    totals(capabilities, "registered")["code_token_lines"]
                    + totals(capabilities, "orphans")["code_token_lines"]
                ),
            },
            "infrastructure": totals(capabilities, "infrastructure"),
        },
        "verification_evidence": {
            "registered_with_invocation_case": len(registered_ids & cases),
            "registered_without_invocation_case": len(without_case),
            "ids_without_case": without_case,
            "cases_for_unregistered_ids": stale_cases,
            "gate": "tests/fx1/test_operations_registry_invocation.py::test_invocation_cases_match_the_registry",
        },
        "doc_claims": parse_doc_claims(),
        "extensions": scan_extensions(),
        "capabilities": [
            {
                "operation_id": c.operation_id,
                "kind": c.kind,
                "version": c.version,
                "path": c.path,
                "classification": c.classification,
                "has_invocation_case": c.has_invocation_case,
                "physical_lines": c.measure.physical_lines,
                "nonblank_lines": c.measure.nonblank_lines,
                "code_token_lines": c.measure.code_token_lines,
            }
            for c in sorted(capabilities, key=lambda c: (c.classification, c.stem))
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    ops = report["operations_directory"]
    reg = report["registry"]
    loc = report["loc"]
    ev = report["verification_evidence"]
    git = report["git"]
    lines: list[str] = []
    add = lines.append

    add("# FX-1 capability baseline — Phase 0 snapshot")
    add("")
    add(f"**Generated:** {report['generated_at_utc']}  ")
    add(f"**Commit:** `{git['branch']}` @ `{git['head_short']}`  ")
    add(
        f"**Tree:** {git['staged']} staged · {git['modified']} modified · {git['untracked']} untracked  "
    )
    add(
        "**Reproduce:** `uv run python scripts/fx1_capability_baseline.py "
        "--output-json artifacts/fx1_capability_baseline.json "
        "--output-markdown docs/FX1_CAPABILITY_BASELINE.md`"
    )
    add("")
    add("Phase 0 of `docs/ULTRA_INTENSE_CAPABILITY_PURSUIT_PLAN.md`. Read-only inventory;")
    add("no capability file was modified to produce this snapshot.")
    add("")

    add("## Frozen counting rules")
    add("")
    add("| Measure | Definition |")
    add("|---|---|")
    for key, value in report["counting_rules"].items():
        add(f"| `{key}` | {value} |")
    add("")

    add("## Inventory")
    add("")
    add(f"`{ops['path']}/` holds **{ops['total_python_files']}** `.py` files:")
    add("")
    add(f"- **{ops['registered']}** registered implementations (accepted candidates)")
    add(f"- **{ops['orphans']}** orphan modules — export `OPERATION`, absent from the registry")
    add(
        f"- **{ops['infrastructure']}** infrastructure: {', '.join(f'`{s}.py`' for s in ops['infrastructure_stems'])}"
    )
    add("")
    add(
        f"Registry `_IMPLEMENTATIONS` holds **{reg['implementation_count']}** ids: "
        f"{', '.join(f'{k}={v}' for k, v in reg['per_kind'].items())}."
    )
    if reg["orphan_per_kind"]:
        add("")
        add(f"Orphans by kind: {', '.join(f'{k}={v}' for k, v in reg['orphan_per_kind'].items())}.")
    add("")

    add("## LOC (registered implementation files only)")
    add("")
    add("Scope matches the progress ledger: shared infrastructure, tests, documentation and")
    add("generated artifacts excluded.")
    add("")
    add("| Measure | Registered | Orphans | All operation modules |")
    add("|---|---:|---:|---:|")
    for label, key in (
        ("physical lines", "physical_lines"),
        ("nonblank lines", "nonblank_lines"),
        ("code-token lines", "code_token_lines"),
    ):
        add(
            f"| {label} | {loc['registered_only'][key]:,} | "
            f"{loc['orphans_only'][key]:,} | {loc['all_operation_modules'][key]:,} |"
        )
    add(
        f"| files | {loc['registered_only']['files']} | "
        f"{loc['orphans_only']['files']} | {loc['all_operation_modules']['files']} |"
    )
    add("")

    add("## Reconciliation — every mismatch classified")
    add("")
    if reg["ids_missing_a_source_file"]:
        add(
            f"**Registry ids with no source file ({len(reg['ids_missing_a_source_file'])}):** "
            + ", ".join(f"`{i}`" for i in reg["ids_missing_a_source_file"])
        )
    else:
        add(
            "- Registry ids with no source file: **none** — every registered id resolves to a module."
        )
    add("")
    if reg["source_files_not_registered"]:
        add(
            f"**Source modules not in the registry ({len(reg['source_files_not_registered'])}) — "
            "unreachable via `fx1 harness operations`, `describe-operation`, `execute-operation`:**"
        )
        add("")
        for operation_id in reg["source_files_not_registered"]:
            add(f"- `{operation_id}`")
    else:
        add("- Source modules not in the registry: **none**.")
    add("")

    add("### Doc claims vs measured")
    add("")
    add("| Doc | Claim | Claimed | Measured | Verdict |")
    add("|---|---|---:|---:|---|")
    measured = {
        "implementations": reg["implementation_count"],
        "separate_files": loc["registered_only"]["files"],
        "physical_lines": loc["registered_only"]["physical_lines"],
        "nonblank_lines": loc["registered_only"]["nonblank_lines"],
        "code_token_lines": loc["registered_only"]["code_token_lines"],
        "features": reg["per_kind"].get("feature", 0),
        "skills": reg["per_kind"].get("skill", 0),
        "plugins": reg["per_kind"].get("plugin", 0),
    }
    display = {
        "implementations": "implementations",
        "separate_files": "separate files",
        "physical_lines": "physical lines",
        "nonblank_lines": "nonblank lines",
        "code_token_lines": "code-token lines",
        "features": "features",
        "skills": "skills",
        "plugins": "plugins",
    }
    for relative, entry in sorted(report["doc_claims"].items()):
        for claim_name, claim in sorted(entry["claims"].items()):
            if claim_name not in measured:
                continue
            actual = measured[claim_name]
            verdict = (
                "match" if claim["value"] == actual else f"**DRIFT {claim['value'] - actual:+d}**"
            )
            add(
                f"| `{relative}`:{claim['line']} | {display[claim_name]} | "
                f"{claim['value']:,} | {actual:,} | {verdict} |"
            )
    add("")

    add("## Verification evidence (Phase 1 input)")
    add("")
    add(
        f"- Registered ids **with** a hand-checked invocation case: **{ev['registered_with_invocation_case']}**"
    )
    add(f"- Registered ids **without** one: **{ev['registered_without_invocation_case']}**")
    add(
        f"- Cases referencing ids that are not registered: **{len(ev['cases_for_unregistered_ids'])}**"
    )
    add(f"- Gate: `{ev['gate']}`")
    add("")
    if ev["ids_without_case"]:
        add("Ids still needing a behavioral case:")
        add("")
        for operation_id in ev["ids_without_case"]:
            add(f"- `{operation_id}`")
        add("")

    add("## Verification matrix")
    add("")
    add("Harness path per capability. `case` = a hand-checked invocation case exists.")
    add("")
    add(
        "| # | Operation id | Kind | Ver | Class | Case | phys | nonblank | code | Harness command |"
    )
    add("|---:|---|---|---|---|:--:|---:|---:|---:|---|")
    index = 0
    for capability in report["capabilities"]:
        if capability["classification"] == "infrastructure":
            continue
        index += 1
        operation_id = capability["operation_id"] or "?"
        command = (
            f"`fx1 harness execute-operation {operation_id}`"
            if capability["classification"] == "registered"
            else "unreachable — not registered"
        )
        add(
            f"| {index} | `{operation_id}` | {capability['kind'] or '?'} | "
            f"{capability['version'] or '?'} | {capability['classification']} | "
            f"{'yes' if capability['has_invocation_case'] else '**no**'} | "
            f"{capability['physical_lines']} | {capability['nonblank_lines']} | "
            f"{capability['code_token_lines']} | {command} |"
        )
    add("")

    ext = report["extensions"]
    if ext.get("present"):
        add("## Extension tree (generated bindings — not accepted capabilities)")
        add("")
        add(f"- Hand-written core: {', '.join(f'`{p}`' for p in ext['hand_written_core'])}")
        add(
            f"- Generated bindings: **{ext['generated_binding_count']}** across "
            + ", ".join(f"`{k}`={v}" for k, v in ext["generated_bindings_by_subpackage"].items())
        )
        add(
            f"- LOC: {ext['loc']['physical_lines']:,} physical / "
            f"{ext['loc']['nonblank_lines']:,} nonblank / "
            f"{ext['loc']['code_token_lines']:,} code-token"
        )
        add("")
        add("Per the plan's independence rule, generated wrappers **do not count** toward the")
        add("accepted total. `docs/FX1_CAPABILITIES.md` already states this.")
        add("")

    add("## Gate 0 status")
    add("")
    blockers = []
    if reg["source_files_not_registered"]:
        blockers.append(
            f"{len(reg['source_files_not_registered'])} complete modules are unregistered "
            "(unreachable dead code)"
        )
    if ev["registered_without_invocation_case"]:
        blockers.append(
            f"{ev['registered_without_invocation_case']} registered ids lack a behavioral case"
        )
    if blockers:
        add("**Gate 0 is NOT closed.** Outstanding inventory disagreements:")
        add("")
        for blocker in blockers:
            add(f"- {blocker}")
        add("")
        add("Per the plan: *no candidate work begins until inventory disagreements and")
        add("counting rules are explicit.* The counting rules above are now explicit and")
        add("frozen; the disagreements listed here need an owner disposition (register,")
        add("reject, or defer) before Phase 1 can close.")
    else:
        add("**Gate 0 closed** — inventory reconciled, counting rules frozen.")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, help="write the machine-readable baseline")
    parser.add_argument("--output-markdown", type=Path, help="write the verification matrix")
    parser.add_argument("--print-summary", action="store_true", help="print key figures to stdout")
    args = parser.parse_args(argv)

    report = build_report()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    if args.print_summary or not (args.output_json or args.output_markdown):
        ops = report["operations_directory"]
        reg = report["registry"]
        loc = report["loc"]["registered_only"]
        ev = report["verification_evidence"]
        print(f"operations .py files : {ops['total_python_files']}")
        print(f"  registered         : {ops['registered']}  ({reg['per_kind']})")
        print(f"  orphans            : {ops['orphans']}")
        print(f"  infrastructure     : {ops['infrastructure']}")
        print(
            f"LOC (registered only): physical={loc['physical_lines']:,} "
            f"nonblank={loc['nonblank_lines']:,} code_token={loc['code_token_lines']:,}"
        )
        print(
            f"with invocation case : {ev['registered_with_invocation_case']} / {ops['registered']}"
        )
        print(f"missing a case       : {ev['registered_without_invocation_case']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
