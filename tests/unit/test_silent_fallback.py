"""Fail closed when a high-risk path swallows a missing required key.

The extra-conformal scorecard used to catch ``KeyError`` and return ``{}``.
``family_blob_executed`` treats an empty dict as "not run", so a renamed
score disappeared from the receipt. This module pins that class of bug:

- a missing ``synthetic_coverage`` raises
- an evidence report missing a required section raises
- a new silent ``except`` / ``suppress`` / required-key ``.get`` in the
  live, risk, ledger, reconcile, scorecard, or report packages fails
  unless ``tests/silent_fallback_allowlist.txt`` documents it
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "tests" / "silent_fallback_allowlist.txt"
HIGH_PREFIXES = (
    "src/quant_fund/live/",
    "src/quant_fund/execution/",
    "src/quant_fund/pretrade/",
    "src/quant_fund/risk/",
    "src/quant_fund/reporting/",
    "src/quant_fund/paper/",
    "src/quant_fund/audit/",
    "src/quant_fund/portfolio/",
    "src/quant_fund/research/",
)
EXTRA_FILES = frozenset(
    {
        "src/quant_fund/microstructure/sim_real_ledger.py",
        "src/fx1/data/ledger.py",
    }
)
SWALLOW_TYPES = frozenset({"Exception", "BaseException", "KeyError", "bare"})
REQUIRED_GET_KEYS = frozenset(
    {"candidates", "provenance", "health", "promotion", "synthetic_coverage"}
)
LOG_NAMES = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log", "warn"})


def _in_scope(rel: str) -> bool:
    if rel in EXTRA_FILES:
        return True
    if rel.endswith("/ledger.py") and rel.startswith("src/quant_fund/"):
        return True
    return rel.startswith(HIGH_PREFIXES)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Tuple):
            continue
        names: list[str] = []
        for elt in node.value.elts:
            if not isinstance(elt, ast.Name):
                names = []
                break
            names.append(elt.id)
        if names:
            found[target.id] = tuple(names)
    return found


def _type_names(node: ast.expr | None, alias: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    if node is None:
        return ("bare",)
    if isinstance(node, ast.Name):
        return alias.get(node.id, (node.id,))
    if isinstance(node, ast.Attribute):
        return (node.attr,)
    if isinstance(node, ast.Tuple):
        names: list[str] = []
        for elt in node.elts:
            names.extend(_type_names(elt, alias))
        return tuple(names)
    return ("?",)


def _return_kind(node: ast.Return) -> str | None:
    value = node.value
    if value is None:
        return "None"
    if isinstance(value, ast.Constant):
        if value.value is None:
            return "None"
        if isinstance(value.value, bool):
            return None
        if value.value == {}:
            return "{}"
        if value.value == []:
            return "[]"
        if value.value == ():
            return "()"
        if value.value == 0 or value.value == 0.0:
            return "0"
        if value.value == "":
            return "empty-str"
        if isinstance(value.value, float) and value.value != value.value:
            return "nan"
    if isinstance(value, ast.Dict) and not value.keys:
        return "{}"
    if isinstance(value, ast.List) and not value.elts:
        return "[]"
    if isinstance(value, ast.Tuple) and not value.elts:
        return "()"
    if isinstance(value, ast.Call):
        name = _call_name(value)
        if name == "dict" and not value.args and not value.keywords:
            return "{}"
        if name in {"list", "tuple"} and not value.args and not value.keywords:
            return "[]"
        if (
            name == "float"
            and value.args
            and isinstance(value.args[0], ast.Constant)
            and str(value.args[0].value).lower() == "nan"
        ):
            return "nan"
    return None


def _is_log(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _call_name(stmt.value) in LOG_NAMES
    )


def _body_kind(body: list[ast.stmt]) -> str | None:
    kinds: list[str] = []
    for stmt in body:
        if _is_log(stmt):
            kinds.append("log")
            continue
        if isinstance(stmt, ast.Pass):
            kinds.append("pass")
            continue
        if isinstance(stmt, ast.Continue):
            kinds.append("continue")
            continue
        if isinstance(stmt, ast.Return):
            kind = _return_kind(stmt)
            if kind is not None:
                kinds.append(f"return {kind}")
                continue
        return None
    return "+".join(kinds) if kinds else "empty"


def _default_kind(node: ast.expr) -> str | None:
    if isinstance(node, ast.Dict) and not node.keys:
        return "{}"
    if isinstance(node, ast.List) and not node.elts:
        return "[]"
    if (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and (node.value == 0 or node.value == 0.0)
    ):
        return "0"
    return None


def scan_source(rel: str, source: str) -> list[tuple[str, str, str, str]]:
    """Return swallow fingerprints ``(path, qualname, types, kind)``."""
    tree = ast.parse(source)
    alias = _aliases(tree)
    found: list[tuple[str, str, str, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []
            self.bound: dict[str, str] = {}

        def _qual(self) -> str:
            return ".".join(self.stack) or "<module>"

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_For(self, node: ast.For) -> None:
            bound: str | None = None
            if isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.Tuple, ast.List)):
                keys = [
                    elt.value
                    for elt in node.iter.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                required = sorted(REQUIRED_GET_KEYS.intersection(keys))
                if required:
                    bound = node.target.id
                    self.bound[bound] = ",".join(required)
            self.generic_visit(node)
            if bound is not None:
                del self.bound[bound]

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            names = _type_names(node.type, alias)
            if any(name in SWALLOW_TYPES for name in names):
                kind = _body_kind(node.body)
                if kind is not None:
                    found.append((rel, self._qual(), ",".join(names), kind))
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                ctx = item.context_expr
                if not isinstance(ctx, ast.Call) or _call_name(ctx) != "suppress":
                    continue
                names: list[str] = []
                for arg in ctx.args:
                    names.extend(_type_names(arg, alias))
                if any(name in {"Exception", "BaseException", "KeyError"} for name in names):
                    found.append((rel, self._qual(), "suppress:" + ",".join(names), "suppress"))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get" and len(node.args) >= 2:
                default = _default_kind(node.args[1])
                key = node.args[0]
                label: str | None = None
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value in REQUIRED_GET_KEYS:
                        label = key.value
                elif isinstance(key, ast.Name) and key.id in self.bound:
                    label = self.bound[key.id]
                if default is not None and label is not None:
                    found.append((rel, self._qual(), f"get:{label}", default))
            self.generic_visit(node)

    Visitor().visit(tree)
    return _number_duplicates(found)


def _number_duplicates(
    rows: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    seen: dict[tuple[str, str, str, str], int] = {}
    numbered: list[tuple[str, str, str, str]] = []
    for path, qual, types, kind in rows:
        key = (path, qual, types, kind)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            qual = f"{qual}#{seen[key]}"
        numbered.append((path, qual, types, kind))
    return numbered


def scan_tree() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if not _in_scope(rel):
            continue
        rows.extend(scan_source(rel, path.read_text(encoding="utf-8")))
    return rows


def _allowlist() -> dict[tuple[str, str, str, str], str]:
    rows: dict[tuple[str, str, str, str], str] = {}
    order: list[tuple[str, str, str, str]] = []
    for lineno, raw in enumerate(ALLOWLIST.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 5:
            raise AssertionError(f"{ALLOWLIST}:{lineno}: expected 5 tab-separated fields")
        path, qual, types, kind, reason = parts
        if not reason.strip():
            raise AssertionError(f"{ALLOWLIST}:{lineno}: reason is empty")
        key = (path, qual, types, kind)
        if key in rows:
            raise AssertionError(f"{ALLOWLIST}:{lineno}: duplicate fingerprint")
        rows[key] = reason
        order.append(key)
    if order != sorted(order):
        raise AssertionError("silent_fallback_allowlist.txt is not sorted")
    return rows


def test_missing_synthetic_coverage_is_not_an_empty_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fund.models.extra_conformal as core
    from quant_fund.research.benches_w17 import bench_extra_conformal

    monkeypatch.setattr(core, "bench_extra_conformal", lambda **_kwargs: {})
    monkeypatch.setattr(
        core,
        "bench_extra_harm",
        lambda **_kwargs: {"synthetic_coverage": 1.0},
    )
    with pytest.raises(KeyError, match="synthetic_coverage"):
        bench_extra_conformal()


def test_extra_conformal_scorecard_publishes_coverage() -> None:
    from quant_fund.research.benches_w17 import bench_extra_conformal

    blob = bench_extra_conformal()
    assert blob
    assert math.isfinite(blob["xc_coverage"])
    assert "synthetic_coverage" not in blob


def test_evidence_markdown_requires_report_sections() -> None:
    from quant_fund.reporting.report import build_evidence_report, write_report_text

    report = build_evidence_report(candidates={"ranker": {"mean_ic": 0.1}})
    text = write_report_text(report)
    assert "## Provenance" in text
    assert "provenance_missing" in text
    incomplete = dict(report)
    del incomplete["provenance"]
    with pytest.raises(KeyError, match="provenance"):
        write_report_text(incomplete)


def test_guard_catches_the_empty_blob_keyerror() -> None:
    source = (
        "_BENCH_EXC = (ValueError, KeyError)\n"
        "def bench(blob):\n"
        "    try:\n"
        "        return {'xc_coverage': blob['synthetic_coverage']}\n"
        "    except _BENCH_EXC:\n"
        "        return {}\n"
        "def render(report):\n"
        "    for section in ('candidates', 'provenance'):\n"
        "        report.get(section, {})\n"
        "def lookup():\n"
        "    from contextlib import suppress\n"
        "    with suppress(KeyError):\n"
        "        blob['synthetic_coverage']\n"
    )
    hits = scan_source("src/quant_fund/research/example.py", source)
    assert (
        "src/quant_fund/research/example.py",
        "bench",
        "ValueError,KeyError",
        "return {}",
    ) in hits
    assert (
        "src/quant_fund/research/example.py",
        "render",
        "get:candidates,provenance",
        "{}",
    ) in hits
    assert any(row[2].startswith("suppress:KeyError") for row in hits)


def test_guard_leaves_numeric_and_import_degrades() -> None:
    source = (
        "def bench():\n"
        "    try:\n"
        "        return {'xc_coverage': 1.0}\n"
        "    except ImportError:\n"
        "        return {}\n"
        "    except (ValueError, RuntimeError, FloatingPointError, TypeError):\n"
        "        return {}\n"
        "def verify():\n"
        "    try:\n"
        "        return True\n"
        "    except KeyError:\n"
        "        return False\n"
        "def checked():\n"
        "    try:\n"
        "        return blob['synthetic_coverage']\n"
        "    except KeyError:\n"
        "        raise\n"
    )
    assert scan_source("src/quant_fund/research/example.py", source) == []


def test_high_risk_swallows_match_the_allowlist() -> None:
    found = scan_tree()
    allowed = _allowlist()
    missing = [row for row in found if row not in allowed]
    stale = [row for row in allowed if row not in set(found)]
    assert missing == [], "undocumented silent fallback:\n" + "\n".join(
        "\t".join(row) for row in missing
    )
    assert stale == [], "allowlist row no longer matches the tree:\n" + "\n".join(
        "\t".join(row) for row in stale
    )
    jewels = set(json.loads((ROOT / "quality" / "crown_jewels.json").read_text())["files"])
    for key, reason in allowed.items():
        path = key[0]
        assert (ROOT / path).is_file(), path
        if path in jewels:
            assert "crown-jewel" in reason, path
