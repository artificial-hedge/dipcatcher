"""Deterministic qualification audit for the wave-generated model corpus.

This module classifies every module in ``src/quant_fund/models`` as either
``QUALIFYING`` or ``NON_QUALIFYING_TEMPLATE`` using objective, documented and
hash-pinned rules (``RULESET`` / :func:`ruleset_hash`). It exists so corpus
integrity claims are reproducible and tamper-evident: two runs over the same
tree emit byte-identical JSON (``quality/canon_qualification_audit.json``) and
any rule change changes ``ruleset_hash``.

A ``NON_QUALIFYING_TEMPLATE`` verdict means the module is a template copy of a
shared skeleton (near-duplicate AST shape shared with other corpus modules)
whose entire computation is boolean plumbing over constant arguments — i.e.
the ``sum(checks) / len(checks)`` bench over constant ``True``/``False``
checks with no data. Such modules are correctness fixtures for a generator,
never distinct implementations, and never market evidence.

Verdict rule (see ``RULESET["verdict_rule"]``): a module is
``NON_QUALIFYING_TEMPLATE`` iff ALL of

1. ``ast_shape_shared``        — its normalized AST shape occurs in >= 2
   corpus modules (near-duplicate template structure).
2. ``no_substantive_bench``    — no ``bench_*``/``_bench_*`` does real work
   (any bench present is constant-check aggregation only).
3. ``control_flow_count == 0`` — no ``if``/``for``/``while``/``try``/``with``/
   ``match``.
4. ``effective_body_lines < 40`` — nonblank, non-comment, non-docstring
   source lines stay under the pinned threshold.
5. ``data_params == 0``        — no parameter annotated with a numeric/array/
   data type (``bool`` and ``int`` are plumbing: flags and seeds).

Corroborating signals (recorded, not required): no imports beyond
``__future__``/``typing``; module docstring equals a shared template skeleton
with the module stem filled in.

Nothing here reads network, clock or environment state; every output derives
from file bytes, so the audit is reproducible. SYNTHETIC-labelled corpus
modules are correctness fixtures only — never market evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RULESET_VERSION = 1

#: Data-carrying parameter annotations. ``bool`` flags and ``int`` seeds or
#: counts are plumbing, not data; the mission's "zero parameters of numeric/
#: array type" signal is about parameters that carry observations.
DATA_PARAM_TOKENS: tuple[str, ...] = (
    "Array",
    "DataFrame",
    "NDArray",
    "Series",
    "dict",
    "float",
    "list",
    "ndarray",
    "str",
    "tuple",
)

#: Imports that do not evidence substantive dependency use.
IMPORTS_ALLOWLIST: tuple[str, ...] = ("__future__", "typing")

CONTROL_FLOW_NODES: tuple[type[ast.stmt], ...] = (
    ast.For,
    ast.If,
    ast.Match,
    ast.Try,
    ast.While,
    ast.With,
)

RULESET: dict[str, Any] = {
    "ruleset_name": "canon_qualification",
    "version": RULESET_VERSION,
    "population": {
        "models_globs": (
            "src/quant_fund/models/*.py",
            "attic/**/src/quant_fund/models/*.py",
        ),
        "wiring_globs": (
            "src/quant_fund/research/benches_w*.py",
            "attic/**/benches_w*.py",
        ),
        "excluded_stems": ("__init__",),
    },
    "signals": {
        "ast_shape_shared": (
            "normalized-AST SHA-256 (identifiers and literal values replaced "
            "by type placeholders; boolean/None constants kept) occurs in >= "
            "ast_shape_shared_min corpus modules"
        ),
        "no_substantive_bench": (
            "no bench_* / _bench_* function performs real work: every bench "
            "either aggregates checks over constant-argument calls of local "
            "predicates (returning sum(checks)/len(checks)) or wraps a local "
            "_bench_* in a synthetic_ dict; loops, randomness, imports, "
            "subscripts and data parameters make a bench substantive"
        ),
        "control_flow_count": (
            "count of if/for/while/try/with/match statements equals zero"
        ),
        "effective_body_lines": (
            "nonblank, non-comment, non-docstring source lines below "
            "effective_body_lines_max"
        ),
        "data_params": (
            "count of parameters annotated with data-carrying types in "
            "DATA_PARAM_TOKENS equals zero"
        ),
    },
    "corroborating_signals": {
        "nontrivial_imports": (
            "no imports beyond __future__/typing (IMPORTS_ALLOWLIST)"
        ),
        "docstring_skeleton_shared": (
            "module docstring with stem variants replaced by <STEM> is shared "
            "by >= docstring_skeleton_shared_min corpus modules"
        ),
    },
    "verdict_rule": (
        "NON_QUALIFYING_TEMPLATE iff ast_shape_shared AND "
        "no_substantive_bench AND control_flow_count == 0 AND "
        "effective_body_lines < effective_body_lines_max AND data_params == 0; "
        "otherwise QUALIFYING"
    ),
    "thresholds": {
        "ast_shape_shared_min": 2,
        "docstring_skeleton_shared_min": 2,
        "effective_body_lines_max": 40,
    },
    "imports_allowlist": IMPORTS_ALLOWLIST,
    "data_param_tokens": DATA_PARAM_TOKENS,
}


def ruleset_hash() -> str:
    """SHA-256 of the canonical JSON encoding of :data:`RULESET`."""
    canonical = json.dumps(RULESET, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModuleRecord:
    """Feature record for one corpus module (see module docstring)."""

    path: str
    stem: str
    physical_lines: int
    nonblank_lines: int
    effective_body_lines: int
    control_flow_count: int
    nontrivial_imports: tuple[str, ...]
    data_params: int
    bench_profile: str
    ast_shape_hash: str
    docstring_skeleton: str
    docstring_mentions_stem: bool


@dataclass(frozen=True)
class Verdict:
    """Qualification verdict with the evidence behind it."""

    verdict: str
    fired_rules: tuple[str, ...]
    corroborating: tuple[str, ...]


@dataclass
class _ShapeCollector(ast.NodeTransformer):
    """Normalize identifiers and literals so only structure survives."""

    local_names: frozenset[str] = field(default_factory=frozenset)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_def(node)

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        node = node
        generic = self.generic_visit(node)
        assert isinstance(generic, (ast.FunctionDef, ast.AsyncFunctionDef))
        generic.name = "<f>"
        return generic

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        generic = self.generic_visit(node)
        assert isinstance(generic, ast.ClassDef)
        generic.name = "<c>"
        return generic

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="<n>", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        generic = self.generic_visit(node)
        assert isinstance(generic, ast.arg)
        generic.arg = "<a>"
        return generic

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        generic = self.generic_visit(node)
        assert isinstance(generic, ast.Attribute)
        generic.attr = "<attr>"
        return generic

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        generic = self.generic_visit(node)
        assert isinstance(generic, ast.keyword)
        generic.arg = "<kw>"
        return generic

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value = node.value
        if value is None:
            placeholder: ast.expr = ast.Constant(value=None)
        elif isinstance(value, bool):
            placeholder = ast.Constant(value=value)
        elif isinstance(value, str):
            placeholder = ast.Constant(value="<str>")
        elif isinstance(value, (int, float, complex)):
            placeholder = ast.Constant(value=0)
        else:
            placeholder = ast.Constant(value="<const>")
        return ast.copy_location(placeholder, node)


def ast_shape_hash(tree: ast.Module) -> str:
    """Hash of the normalized AST structure (names/literals normalized)."""
    normalized = _ShapeCollector().visit(ast.parse(ast.dump(tree, include_attributes=False)))
    assert isinstance(normalized, ast.Module)
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode("utf-8")).hexdigest()


def _docstring_spans(tree: ast.Module) -> set[int]:
    """Line numbers covered by module/class/function docstrings."""
    spans: set[int] = set()
    for node in ast.walk(tree):
        bodies: list[list[ast.stmt]] = []
        if isinstance(node, ast.Module):
            bodies.append(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bodies.append(node.body)
        for body in bodies:
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                spans.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return spans


def effective_body_lines(source: str, tree: ast.Module) -> int:
    """Count source lines that are code: not blank, comment-only or docstring."""
    docstring_lines = _docstring_spans(tree)
    count = 0
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if lineno in docstring_lines:
            continue
        count += 1
    return count


def count_control_flow(tree: ast.Module) -> int:
    """Count if/for/while/try/with/match statements."""
    return sum(1 for node in ast.walk(tree) if isinstance(node, CONTROL_FLOW_NODES))


def nontrivial_imports(tree: ast.Module) -> tuple[str, ...]:
    """Sorted import roots beyond :data:`IMPORTS_ALLOWLIST`."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return tuple(sorted(roots - set(IMPORTS_ALLOWLIST)))


def data_param_count(tree: ast.Module) -> int:
    """Count parameters annotated with data-carrying (numeric/array) types."""
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if arg.annotation is None:
                continue
            text = ast.unparse(arg.annotation)
            if any(token in text for token in DATA_PARAM_TOKENS):
                count += 1
    return count


def _stem_variants(stem: str) -> tuple[str, ...]:
    spaced = stem.replace("_", " ")
    titled = " ".join(part.capitalize() for part in stem.split("_"))
    return (stem, spaced, titled, stem.upper(), stem.lower())


def docstring_skeleton(tree: ast.Module, stem: str) -> tuple[str, bool]:
    """Module docstring with stem variants replaced by ``<STEM>``.

    Returns the whitespace-collapsed skeleton and whether the docstring
    mentions the module stem at all (docstring-templating evidence).
    """
    doc = ast.get_docstring(tree) or ""
    mentions = False
    skeleton = doc
    for variant in sorted(_stem_variants(stem), key=len, reverse=True):
        if variant and variant in skeleton:
            mentions = True
            skeleton = skeleton.replace(variant, "<STEM>")
    collapsed = " ".join(skeleton.split())
    return collapsed, mentions


def _call_root_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _call_root_name(func.value)
    return None


def _constant_check_expr(node: ast.expr, local_names: frozenset[str]) -> bool:
    """True for literal booleans, ``not`` thereof and constant-arg local calls."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (bool, int, float))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _constant_check_expr(node.operand, local_names)
    if isinstance(node, ast.Call):
        root = _call_root_name(node.func)
        if root not in local_names:
            return False
        args_ok = all(isinstance(arg, ast.Constant) for arg in node.args)
        kwargs_ok = all(
            isinstance(arg.value, ast.Constant) for arg in node.keywords if arg.arg is not None
        )
        no_starred = all(isinstance(arg, ast.Constant) for arg in node.args) and not any(
            kw.arg is None for kw in node.keywords
        )
        return args_ok and kwargs_ok and no_starred
    return False


def _is_sum_over_checks(node: ast.expr) -> bool:
    """True for ``sum(checks) / len(checks)`` (optionally ``float(...)``)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float":
        return len(node.args) == 1 and _is_sum_over_checks(node.args[0])
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    left, right = node.left, node.right
    if not (isinstance(left, ast.Call) and isinstance(right, ast.Call)):
        return False
    left_ok = (
        isinstance(left.func, ast.Name) and left.func.id == "sum" and _is_checks_ref(left.args)
    )
    right_ok = (
        isinstance(right.func, ast.Name) and right.func.id == "len" and _is_checks_ref(right.args)
    )
    return left_ok and right_ok


def _is_checks_ref(args: list[ast.expr]) -> bool:
    return len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "checks"


def _is_synth_dict_wrapper(node: ast.expr, local_names: frozenset[str]) -> bool:
    """True for ``{"synthetic_...": local_bench(seed)}``-style wrapper returns."""
    if not isinstance(node, ast.Dict) or not node.keys:
        return False
    for key in node.keys:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return False
        if not key.value.startswith("synthetic_"):
            return False
    for value in node.values:
        if not isinstance(value, ast.Call):
            return False
        root = _call_root_name(value.func)
        if root not in local_names and not _is_local_bench(value.func):
            return False
        if not all(isinstance(arg, (ast.Constant, ast.Name)) for arg in value.args):
            return False
        if any(kw.arg is None for kw in value.keywords):
            return False
    return True


def _is_local_bench(func: ast.expr) -> bool:
    return isinstance(func, ast.Attribute) and func.attr.startswith("_bench")


def _is_constant_check_bench(fn: ast.FunctionDef, local_names: frozenset[str]) -> bool:
    """True when a bench only aggregates constant checks (template shape)."""
    saw_checks = False
    saw_return = False
    for stmt in fn.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue  # docstring
        if isinstance(stmt, ast.Assign) and _assigns_empty_checks(stmt):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if _is_checks_append(call) and _constant_check_expr(call.args[0], local_names):
                saw_checks = True
                continue
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if _is_sum_over_checks(stmt.value) or _is_synth_dict_wrapper(stmt.value, local_names):
                saw_return = True
                continue
        return False
    return saw_checks or saw_return


def _assigns_empty_checks(stmt: ast.Assign) -> bool:
    targets = stmt.targets
    value = stmt.value
    return (
        len(targets) == 1
        and isinstance(targets[0], ast.Name)
        and targets[0].id == "checks"
        and isinstance(value, ast.List)
        and not value.elts
    )


def _is_checks_append(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "append"
        and isinstance(func.value, ast.Name)
        and func.value.id == "checks"
        and len(call.args) == 1
        and not call.keywords
    )


def bench_profile(tree: ast.Module) -> str:
    """Classify benches: ``substantive`` | ``constant_check`` | ``none``.

    Any substantive bench wins; constant-check aggregation alone (over
    constant-argument predicate calls with no data) is the template shape.
    """
    local_names = frozenset(
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    profiles: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if "bench" not in node.name:
            continue
        if _is_constant_check_bench(node, local_names):
            profiles.append("constant_check")
        else:
            profiles.append("substantive")
    if not profiles:
        return "none"
    return "substantive" if "substantive" in profiles else "constant_check"


def analyze_module(path: Path, source: str, repo_root: Path) -> ModuleRecord:
    """Extract the qualification feature record for one module."""
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(repo_root).as_posix()
    stem = path.stem
    skeleton, mentions = docstring_skeleton(tree, stem)
    return ModuleRecord(
        path=rel,
        stem=stem,
        physical_lines=len(source.splitlines()),
        nonblank_lines=sum(1 for line in source.splitlines() if line.strip()),
        effective_body_lines=effective_body_lines(source, tree),
        control_flow_count=count_control_flow(tree),
        nontrivial_imports=nontrivial_imports(tree),
        data_params=data_param_count(tree),
        bench_profile=bench_profile(tree),
        ast_shape_hash=ast_shape_hash(tree),
        docstring_skeleton=skeleton,
        docstring_mentions_stem=mentions,
    )


def verdict_for(
    record: ModuleRecord, shape_shared: bool, skeleton_shared: bool
) -> Verdict:
    """Apply the pinned verdict rule (see module docstring)."""
    thresholds = RULESET["thresholds"]
    assert isinstance(thresholds, dict)
    body_max = int(thresholds["effective_body_lines_max"])
    fired: list[str] = []
    if shape_shared:
        fired.append("ast_shape_shared")
    if record.bench_profile != "substantive":
        fired.append("no_substantive_bench")
    if record.control_flow_count == 0:
        fired.append("control_flow_count")
    if record.effective_body_lines < body_max:
        fired.append("effective_body_lines")
    if record.data_params == 0:
        fired.append("data_params")
    corroborating: list[str] = []
    if not record.nontrivial_imports:
        corroborating.append("nontrivial_imports")
    if skeleton_shared and record.docstring_mentions_stem:
        corroborating.append("docstring_skeleton_shared")
    verdict = "NON_QUALIFYING_TEMPLATE" if len(fired) == 5 else "QUALIFYING"
    return Verdict(verdict, tuple(fired), tuple(corroborating))


@dataclass(frozen=True)
class AdapterRecord:
    """One ``benches_w*`` adapter: wave id and the families it wires."""

    path: str
    wave: int | None
    location: str
    families: tuple[tuple[str, tuple[str, ...]], ...]


def _module_stems_used(fn: ast.FunctionDef, imported: dict[str, str]) -> tuple[str, ...]:
    stems: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in imported:
            stems.add(imported[node.id])
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in imported:
                stems.add(imported[node.value.id])
        elif isinstance(node, ast.ImportFrom) and node.module:
            stems.add(node.module.rsplit(".", 1)[-1])
    return tuple(sorted(stems))


def scan_adapter(path: Path, source: str, repo_root: Path) -> AdapterRecord:
    """Extract wave id and per-family model-module wiring from an adapter."""
    tree = ast.parse(source, filename=str(path))
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quant_fund.models"):
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                if node.module == "quant_fund.models":
                    imported[local] = alias.name
                else:
                    imported[local] = node.module.rsplit(".", 1)[-1]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name.startswith("quant_fund.models."):
                    imported[local] = alias.name.rsplit(".", 1)[-1]
    families: list[tuple[str, tuple[str, ...]]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("bench_"):
            continue
        family = node.name[len("bench_") :]
        if family.endswith("_family"):
            family = family[: -len("_family")]
        families.append((family, _module_stems_used(node, imported)))
    rel = path.relative_to(repo_root).as_posix()
    location = "attic" if rel.startswith("attic/") else "live"
    digits = path.stem.removeprefix("benches_w")
    wave = int(digits) if digits.isdigit() else None
    return AdapterRecord(
        path=rel, wave=wave, location=location, families=tuple(sorted(families))
    )


def _collect_paths(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path)
    return sorted(found)


def _record_payload(record: ModuleRecord) -> dict[str, Any]:
    return {
        "path": record.path,
        "stem": record.stem,
        "physical_lines": record.physical_lines,
        "nonblank_lines": record.nonblank_lines,
        "effective_body_lines": record.effective_body_lines,
        "control_flow_count": record.control_flow_count,
        "nontrivial_imports": list(record.nontrivial_imports),
        "data_params": record.data_params,
        "bench_profile": record.bench_profile,
        "ast_shape_hash": record.ast_shape_hash,
        "docstring_skeleton": record.docstring_skeleton,
        "docstring_mentions_stem": record.docstring_mentions_stem,
    }


def audit_tree(repo_root: Path) -> dict[str, Any]:
    """Build the full deterministic audit document for *repo_root*."""
    population = RULESET["population"]
    assert isinstance(population, dict)
    model_paths = _collect_paths(repo_root, population["models_globs"])
    excluded = set(population["excluded_stems"])
    records: list[ModuleRecord] = []
    for path in model_paths:
        if path.stem in excluded:
            continue
        records.append(analyze_module(path, path.read_text(encoding="utf-8"), repo_root))
    shape_counts: dict[str, int] = {}
    skeleton_counts: dict[str, int] = {}
    for record in records:
        shape_counts[record.ast_shape_hash] = shape_counts.get(record.ast_shape_hash, 0) + 1
        skeleton_counts[record.docstring_skeleton] = (
            skeleton_counts.get(record.docstring_skeleton, 0) + 1
        )
    thresholds = RULESET["thresholds"]
    assert isinstance(thresholds, dict)
    shape_min = int(thresholds["ast_shape_shared_min"])
    skeleton_min = int(thresholds["docstring_skeleton_shared_min"])

    adapters = [
        scan_adapter(path, path.read_text(encoding="utf-8"), repo_root)
        for path in _collect_paths(repo_root, population["wiring_globs"])
    ]
    stem_verdicts: dict[str, str] = {}
    family_wiring: dict[str, set[str]] = {}
    family_waves: dict[str, set[int]] = {}
    for adapter in adapters:
        for family, stems in adapter.families:
            for stem in stems:
                family_wiring.setdefault(family, set()).add(stem)
            if adapter.wave is not None:
                family_waves.setdefault(family, set()).add(adapter.wave)

    files: list[dict[str, Any]] = []
    for record in records:
        verdict = verdict_for(
            record,
            shape_counts[record.ast_shape_hash] >= shape_min,
            skeleton_counts[record.docstring_skeleton] >= skeleton_min,
        )
        stem_verdicts[record.stem] = verdict.verdict
        location = "attic" if record.path.startswith("attic/") else "live"
        families = sorted(f for f, stems in family_wiring.items() if record.stem in stems)
        payload = _record_payload(record)
        payload["location"] = location
        payload["verdict"] = verdict.verdict
        payload["fired_rules"] = list(verdict.fired_rules)
        payload["corroborating"] = list(verdict.corroborating)
        payload["family_ids"] = families
        payload["waves"] = sorted(family_waves.get(families[0], set())) if families else []
        files.append(payload)

    families_payload: list[dict[str, Any]] = []
    for family in sorted(family_wiring):
        stems = sorted(family_wiring[family])
        verdicts = sorted({stem_verdicts.get(stem, "UNRESOLVED") for stem in stems})
        classification = (
            "RETIRED_CANDIDATE" if "NON_QUALIFYING_TEMPLATE" in verdicts else "LIVE_CANDIDATE"
        )
        families_payload.append(
            {
                "family_id": family,
                "modules": stems,
                "waves": sorted(family_waves.get(family, set())),
                "module_verdicts": verdicts,
                "classification": classification,
            }
        )

    live_nonqualifying = sorted(
        row["path"]
        for row in files
        if row["location"] == "live" and row["verdict"] == "NON_QUALIFYING_TEMPLATE"
    )
    totals = _totals(files)
    live_totals = _totals([row for row in files if row["location"] == "live"])
    attic_totals = _totals([row for row in files if row["location"] == "attic"])
    return {
        "schema": "canon_qualification_audit.v1",
        "ruleset_version": RULESET_VERSION,
        "ruleset_hash": ruleset_hash(),
        "ruleset": RULESET,
        "counts": {
            "baseline_corpus": {
                "modules": len(files),
                "qualifying": sum(1 for row in files if row["verdict"] == "QUALIFYING"),
                "non_qualifying_template": sum(
                    1 for row in files if row["verdict"] == "NON_QUALIFYING_TEMPLATE"
                ),
                **totals,
            },
            "live_tree": {
                "modules": sum(1 for row in files if row["location"] == "live"),
                "qualifying": sum(
                    1 for row in files if row["location"] == "live" and row["verdict"] == "QUALIFYING"
                ),
                "non_qualifying_template": len(live_nonqualifying),
                **live_totals,
            },
            "attic": {
                "modules": sum(1 for row in files if row["location"] == "attic"),
                **attic_totals,
            },
            "families": {
                "wired": len(families_payload),
                "live_candidate": sum(
                    1 for row in families_payload if row["classification"] == "LIVE_CANDIDATE"
                ),
                "retired_candidate": sum(
                    1 for row in families_payload if row["classification"] == "RETIRED_CANDIDATE"
                ),
            },
            "wiring": {
                "adapters": len(adapters),
                "live_adapters": sum(1 for a in adapters if a.location == "live"),
                "attic_adapters": sum(1 for a in adapters if a.location == "attic"),
            },
        },
        "outstanding_quarantine": live_nonqualifying,
        "files": files,
        "families": families_payload,
        "wiring": [
            {
                "path": adapter.path,
                "wave": adapter.wave,
                "location": adapter.location,
                "families": [
                    {"family_id": family, "modules": list(stems)}
                    for family, stems in adapter.families
                ],
            }
            for adapter in adapters
        ],
    }


def _totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "physical_lines": sum(int(row["physical_lines"]) for row in rows),
        "nonblank_lines": sum(int(row["nonblank_lines"]) for row in rows),
    }


# ---------------------------------------------------------------------------
# Import-graph closure (quarantine safety guard).
#
# A file may move to ``attic/`` only when every one of its importers moves in
# the same step. Otherwise the file and its connected component stay live and
# are listed in ``outstanding_quarantine`` with the blocking importer named.
# ---------------------------------------------------------------------------

#: Owner-guarded stems inside ``src/quant_fund/models`` that are never
#: quarantined (vol-scope lane), matched as fnmatch patterns on file names.
HARD_EXCLUDED_MODEL_FILES: tuple[str, ...] = (
    "__init__.py",
    "garch*.py",
    "har*.py",
    "vol*.py",
)

_GRAPH_ROOTS: tuple[str, ...] = ("src", "tests", "scripts", "research")

_MODULE_REF_RE = r"quant_fund(?:\.[A-Za-z0-9_]+)+"


def _dotted_to_path_map(repo_root: Path) -> tuple[dict[str, str], list[str]]:
    """Map dotted module names to repo-relative file paths (plus package names)."""
    dotted: dict[str, str] = {}
    dotted_names: list[str] = []
    for root in _GRAPH_ROOTS:
        for path in _collect_paths((repo_root / root).as_posix() and repo_root, [f"{root}/**/*.py"]):
            rel = path.relative_to(repo_root).as_posix()
            if rel.endswith("__init__.py"):
                rel = rel[: -len("/__init__.py")] or rel
            name = rel[: -len(".py")] if rel.endswith(".py") else rel
            dotted[name.replace("/", ".")] = path.relative_to(repo_root).as_posix()
            dotted_names.append(name.replace("/", "."))
    for extra in ("conftest.py",):
        path = repo_root / extra
        if path.is_file():
            dotted["conftest"] = extra
            dotted_names.append("conftest")
    return dotted, sorted(dotted_names)


def _string_module_refs(tree: ast.Module) -> set[str]:
    """Dotted ``quant_fund.*`` string literals that name importable modules.

    Catches lazy import maps (e.g. ``models/__init__.py``) without treating
    bare family-name strings as imports.
    """
    import re

    refs: set[str] = set()
    pattern = re.compile(rf"^{_MODULE_REF_RE}$")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if pattern.match(node.value):
                refs.add(node.value)
    return refs


def _import_refs(tree: ast.Module) -> set[str]:
    """Absolute and relative import targets as dotted names (best effort)."""
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                refs.add(node.module)
                for alias in node.names:
                    refs.add(f"{node.module}.{alias.name}")
            elif node.module:
                refs.add(node.module)
    return refs


def _resolve_ref(ref: str, dotted: dict[str, str]) -> str | None:
    """Resolve a dotted reference to a repo-relative file path, if any."""
    candidate = ref
    while "." in candidate or candidate:
        if candidate in dotted:
            return dotted[candidate]
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]
    return None


@dataclass(frozen=True)
class ImportGraph:
    """Repo-internal import edges keyed by repo-relative paths."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    wildcard_importers: tuple[tuple[str, str], ...]


def scan_import_graph(repo_root: Path) -> ImportGraph:
    """Build the intra-repo import graph (file -> imported file)."""
    dotted, _ = _dotted_to_path_map(repo_root)
    nodes: set[str] = set(dotted.values())
    edges: set[tuple[str, str]] = set()
    wildcards: set[tuple[str, str]] = set()
    import re

    dynamic = re.compile(r"""(?:__import__|import_module)\(\s*f?["']([^"']+)""")
    for root in _GRAPH_ROOTS:
        for path in _collect_paths(repo_root, [f"{root}/**/*.py"]):
            rel = path.relative_to(repo_root).as_posix()
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            refs = _import_refs(tree) | _string_module_refs(tree)
            for ref in sorted(refs):
                target = _resolve_ref(ref, dotted)
                if target is not None and target != rel:
                    edges.add((rel, target))
            for match in dynamic.finditer(source):
                spec = match.group(1)
                if "{" in spec:
                    prefix = spec.split("{", 1)[0].rstrip(".")
                    if prefix:
                        wildcards.add((rel, prefix))
                    continue
                target = _resolve_ref(spec, dotted)
                if target is not None and target != rel:
                    edges.add((rel, target))
    return ImportGraph(tuple(sorted(nodes)), tuple(sorted(edges)), tuple(sorted(wildcards)))


def _hard_excluded(rel_path: str) -> bool:
    import fnmatch

    if not rel_path.startswith("src/quant_fund/models/"):
        return False
    name = rel_path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pattern) for pattern in HARD_EXCLUDED_MODEL_FILES)


def _component_map(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Undirected connected-component ids over the given nodes and edges."""
    parent: dict[str, str] = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        if left not in parent or right not in parent:
            continue
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)
    roots = sorted({find(node) for node in parent})
    ids = {root: index + 1 for index, root in enumerate(roots)}
    return {node: ids[find(node)] for node in sorted(parent)}


def quarantine_plan(
    repo_root: Path,
    records: list[ModuleRecord],
    verdicts: dict[str, str],
    adapters: list[AdapterRecord],
    graph: ImportGraph,
) -> dict[str, Any]:
    """Decide which candidate files may move, closing under the import graph.

    Candidates: ``NON_QUALIFYING_TEMPLATE`` model modules (minus owner-guarded
    files), ``benches_w*`` adapters whose wired modules are all non-qualifying,
    and wave tests whose referenced modules/adapters are all candidates. A
    candidate moves only if every importer moves with it; anything else stays
    live, whole connected component, with blocking importers recorded.
    """
    adapter_paths = {adapter.path for adapter in adapters}
    candidate_modules = {
        record.path
        for record in records
        if verdicts.get(record.stem) == "NON_QUALIFYING_TEMPLATE"
        and not _hard_excluded(record.path)
    }
    candidate_adapters = {
        adapter.path
        for adapter in adapters
        if all(stem in _stems(candidate_modules) for _, stems in adapter.families for stem in stems)
    }
    candidate_tests = _candidate_tests(repo_root, candidate_modules | candidate_adapters)
    candidates = candidate_modules | candidate_adapters | candidate_tests
    _ = adapter_paths

    importers: dict[str, set[str]] = {}
    for importer, imported in graph.edges:
        importers.setdefault(imported, set()).add(importer)

    comp = _component_map(candidates, (e for e in graph.edges if e[0] in candidates and e[1] in candidates))
    blocked: dict[int, set[str]] = {}
    for importer, prefix in graph.wildcard_importers:
        if importer in candidates:
            continue
        for candidate in candidates:
            if candidate.startswith(prefix.replace(".", "/")) or candidate.startswith("src/quant_fund/models/"):
                if candidate.startswith("src/quant_fund/models/"):
                    blocked.setdefault(comp[candidate], set()).add(f"{importer} (dynamic f-string import)")

    changed = True
    while changed:
        changed = False
        for candidate in sorted(candidates):
            component = comp[candidate]
            if component in blocked:
                continue
            reasons = _block_reasons(candidate, importers, candidates, comp, blocked)
            if reasons:
                blocked[component] = reasons
                changed = True

    moving = sorted(c for c in candidates if comp[c] not in blocked)
    staying = sorted(c for c in candidates if comp[c] in blocked)
    outstanding = [
        {
            "path": path,
            "verdict": verdicts.get(path.rsplit("/", 1)[-1][: -len(".py")], "UNCLASSIFIED"),
            "component_id": comp[path],
            "blocking_importers": sorted(blocked[comp[path]]),
        }
        for path in staying
    ]
    components = []
    for component_id in sorted(set(comp.values())):
        members = sorted(node for node, cid in comp.items() if cid == component_id)
        components.append(
            {
                "component_id": component_id,
                "members": members,
                "decision": "BLOCKED" if component_id in blocked else "MOVE",
                "blocking_importers": sorted(blocked.get(component_id, set())),
            }
        )
    return {
        "candidate_modules": sorted(candidate_modules),
        "candidate_adapters": sorted(candidate_adapters),
        "candidate_tests": sorted(candidate_tests),
        "moving": moving,
        "staying_live": staying,
        "outstanding_quarantine": outstanding,
        "components": components,
    }


def _stems(paths: set[str]) -> set[str]:
    return {path.rsplit("/", 1)[-1][: -len(".py")] for path in paths}


def _block_reasons(
    candidate: str,
    importers: dict[str, set[str]],
    candidates: set[str],
    comp: dict[str, int],
    blocked: dict[int, set[str]],
) -> set[str]:
    reasons: set[str] = set()
    for importer in sorted(importers.get(candidate, ())):
        if importer in candidates and comp[importer] not in blocked:
            continue
        reasons.add(importer)
    return reasons


def _candidate_tests(repo_root: Path, moved: set[str]) -> set[str]:
    """Wave tests whose referenced corpus files are all candidates to move."""
    import re

    pattern = re.compile(r"quant_fund\.(?:models|research)\.([A-Za-z0-9_]+)")
    candidates: set[str] = set()
    moved_stems = _stems(moved)
    test_paths = _collect_paths(repo_root, ["tests/unit/models/test_w*.py"])
    test_paths += _collect_paths(repo_root, ["tests/unit/research/test_benches_w*.py"])
    for path in test_paths:
        rel = path.relative_to(repo_root).as_posix()
        source = path.read_text(encoding="utf-8")
        stems = set(pattern.findall(source))
        stems = {s for s in stems if s.startswith(("benches_w",)) or s in moved_stems}
        referenced = {s for s in pattern.findall(source)}
        if referenced and referenced <= moved_stems | {s for s in referenced if s.startswith("benches_w")}:
            if all(s in moved_stems for s in referenced if not s.startswith("benches_w")):
                candidates.add(rel)
    return candidates


def dumps_audit(audit: dict[str, Any]) -> str:
    """Canonical deterministic JSON serialization (byte-stable across runs)."""
    return json.dumps(audit, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


__all__ = [
    "RULESET",
    "RULESET_VERSION",
    "AdapterRecord",
    "ModuleRecord",
    "Verdict",
    "analyze_module",
    "audit_tree",
    "bench_profile",
    "count_control_flow",
    "data_param_count",
    "docstring_skeleton",
    "dumps_audit",
    "effective_body_lines",
    "nontrivial_imports",
    "ruleset_hash",
    "scan_adapter",
    "verdict_for",
]
