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

Quarantine safety (import-graph closure): a candidate file may move to
``attic/`` only when every importer moves with it and its whole connected
component is movable. Otherwise the component stays live and is listed in
``outstanding_quarantine`` with the blocking importer named. See
:func:`quarantine_plan`.

Nothing here reads network, clock or environment state; every output derives
from file bytes, so the audit is reproducible. SYNTHETIC-labelled corpus
modules are correctness fixtures only — never market evidence.
"""

from __future__ import annotations

import ast
import copy
import fnmatch
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RULESET_VERSION = 1

#: Data-carrying parameter annotations. ``bool`` flags and ``int`` seeds or
#: counts are plumbing, not data; the "zero parameters of numeric/array type"
#: signal is about parameters that carry observations.
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
        "excluded_stems": ("__init__", "canon_qualification"),
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
        "control_flow_count": ("count of if/for/while/try/with/match statements equals zero"),
        "effective_body_lines": (
            "nonblank, non-comment, non-docstring source lines below effective_body_lines_max"
        ),
        "data_params": (
            "count of parameters annotated with data-carrying types in "
            "DATA_PARAM_TOKENS equals zero"
        ),
    },
    "corroborating_signals": {
        "nontrivial_imports": ("no imports beyond __future__/typing (IMPORTS_ALLOWLIST)"),
        "docstring_skeleton_shared": (
            "module docstring with stem variants replaced by <STEM> is shared "
            "by >= docstring_skeleton_shared_min corpus modules and mentions "
            "the module stem (docstring filled from a template)"
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
    "quarantine_closure_rule": (
        "a candidate file moves to attic/ only when every importer of it also "
        "moves in the same step and its whole connected component is movable; "
        "import edges come from static imports, relative imports, "
        "lazy-import-map dotted quant_fund.* string references, constant-string "
        "importlib calls, and dynamic name loads evidenced by module-stem "
        "string literals in the loading file (parametrized wave tests, literal "
        "lazy-lane names); a computed-name dynamic import with no literal "
        "evidence under its spec prefix blocks every candidate under that "
        "prefix and is recorded in wildcard_importers; bare family-name "
        "strings outside loaders (e.g. catalog registries) never create "
        "edges; anything that cannot move stays live with the blocking "
        "importer named"
    ),
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


@dataclass(frozen=True)
class AdapterRecord:
    """One ``benches_w*`` adapter: wave id and the families it wires."""

    path: str
    wave: int | None
    location: str
    families: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ImportGraph:
    """Repo-internal import edges keyed by repo-relative paths.

    ``wildcard_importers`` lists ``(importer, spec_prefix)`` pairs for
    computed-name dynamic imports with no literal name evidence — each such
    loader is a potential importer of every module under its spec prefix.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    wildcard_importers: tuple[tuple[str, str], ...]


class _ShapeCollector(ast.NodeTransformer):
    """Normalize identifiers and literals so only structure survives."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_def(node)

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
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
        placeholder = _constant_placeholder(node.value)
        return ast.copy_location(placeholder, node)


def _constant_placeholder(value: object) -> ast.expr:
    if value is None:
        return ast.Constant(value=None)
    if isinstance(value, bool):
        return ast.Constant(value=value)
    if isinstance(value, str):
        return ast.Constant(value="<str>")
    if isinstance(value, (int, float, complex)):
        return ast.Constant(value=0)
    return ast.Constant(value="<const>")


def ast_shape_hash(tree: ast.Module) -> str:
    """Hash of the normalized AST structure (names/literals normalized).

    The transform runs on a deep copy so node classes survive verbatim while
    identifier and literal *values* are masked — identical structure hashes
    identically regardless of naming.
    """
    normalized = _ShapeCollector().visit(copy.deepcopy(tree))
    assert isinstance(normalized, ast.Module)
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _docstring_spans(tree: ast.Module) -> set[int]:
    """Line numbers covered by module/class/function docstrings."""
    spans: set[int] = set()
    for node in ast.walk(tree):
        body: list[ast.stmt] | None = None
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
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
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
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
        return _constant_arg_call(node, local_names)
    return False


def _constant_arg_call(node: ast.Call, local_names: frozenset[str]) -> bool:
    root = _call_root_name(node.func)
    if root not in local_names:
        return False
    args_ok = all(isinstance(arg, ast.Constant) for arg in node.args)
    kwargs_ok = all(
        isinstance(kw.value, ast.Constant) for kw in node.keywords if kw.arg is not None
    )
    no_starred = not any(kw.arg is None for kw in node.keywords)
    return args_ok and kwargs_ok and no_starred


def _is_checks_ref(args: list[ast.expr]) -> bool:
    return len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "checks"


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


def _is_local_bench(func: ast.expr) -> bool:
    return isinstance(func, ast.Attribute) and func.attr.startswith("_bench")


def _is_synth_dict_wrapper(node: ast.expr, local_names: frozenset[str]) -> bool:
    """True for ``{"synthetic_...": local_bench(seed)}``-style wrapper returns."""
    if not isinstance(node, ast.Dict) or not node.keys:
        return False
    if not all(_is_synth_key(key) for key in node.keys):
        return False
    return all(_is_wrapper_value(value, local_names) for value in node.values)


def _is_synth_key(key: ast.expr | None) -> bool:
    return (
        isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and key.value.startswith("synthetic_")
    )


def _is_wrapper_value(value: ast.expr, local_names: frozenset[str]) -> bool:
    if not isinstance(value, ast.Call):
        return False
    root = _call_root_name(value.func)
    if root not in local_names and not _is_local_bench(value.func):
        return False
    if not all(isinstance(arg, (ast.Constant, ast.Name)) for arg in value.args):
        return False
    return not any(kw.arg is None for kw in value.keywords)


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


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _handle_bench_stmt(
    stmt: ast.stmt, local_names: frozenset[str], saw_checks: bool, saw_return: bool
) -> tuple[bool, bool, bool]:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        if _is_checks_append(call) and _constant_check_expr(call.args[0], local_names):
            return True, True, saw_return
    if (
        isinstance(stmt, ast.Return)
        and stmt.value is not None
        and (_is_sum_over_checks(stmt.value) or _is_synth_dict_wrapper(stmt.value, local_names))
    ):
        return True, saw_checks, True
    return False, saw_checks, saw_return


def _is_constant_check_bench(fn: ast.FunctionDef, local_names: frozenset[str]) -> bool:
    """True when a bench only aggregates constant checks (template shape)."""
    saw_checks = False
    saw_return = False
    for stmt in fn.body:
        if _is_docstring_stmt(stmt):
            continue
        if isinstance(stmt, ast.Assign) and _assigns_empty_checks(stmt):
            continue
        handled, saw_checks, saw_return = _handle_bench_stmt(
            stmt, local_names, saw_checks, saw_return
        )
        if not handled:
            return False
    return saw_checks or saw_return


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
    lines = source.splitlines()
    return ModuleRecord(
        path=rel,
        stem=stem,
        physical_lines=len(lines),
        nonblank_lines=sum(1 for line in lines if line.strip()),
        effective_body_lines=effective_body_lines(source, tree),
        control_flow_count=count_control_flow(tree),
        nontrivial_imports=nontrivial_imports(tree),
        data_params=data_param_count(tree),
        bench_profile=bench_profile(tree),
        ast_shape_hash=ast_shape_hash(tree),
        docstring_skeleton=skeleton,
        docstring_mentions_stem=mentions,
    )


def verdict_for(record: ModuleRecord, shape_shared: bool, skeleton_shared: bool) -> Verdict:
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


def _model_import_map(tree: ast.Module) -> dict[str, str]:
    """Local name -> model module stem for ``quant_fund.models`` imports."""
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("quant_fund.models"):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                if module == "quant_fund.models":
                    imported[local] = alias.name
                else:
                    imported[local] = module.rsplit(".", 1)[-1]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("quant_fund.models."):
                    local = alias.asname or alias.name
                    imported[local] = alias.name.rsplit(".", 1)[-1]
    return imported


def _module_stems_used(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, imported: dict[str, str]
) -> tuple[str, ...]:
    stems: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in imported:
            stems.add(imported[node.id])
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in imported
        ):
            stems.add(imported[node.value.id])
    return tuple(sorted(stems))


def scan_adapter(path: Path, source: str, repo_root: Path) -> AdapterRecord:
    """Extract wave id and per-family model-module wiring from an adapter."""
    tree = ast.parse(source, filename=str(path))
    imported = _model_import_map(tree)
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
    return AdapterRecord(path=rel, wave=wave, location=location, families=tuple(sorted(families)))


def _collect_paths(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found.add(path)
    return sorted(found)


# ---------------------------------------------------------------------------
# Import-graph closure (quarantine safety guard).
#
# A file may move to ``attic/`` only when every one of its importers moves in
# the same step. Otherwise the file, the files it depends on, and its whole
# connected component stay live and are listed in ``outstanding_quarantine``
# with the blocking importer named.
# ---------------------------------------------------------------------------

#: Owner-guarded files inside ``src/quant_fund/models`` that are never
#: quarantined (vol-scope lane), matched as fnmatch patterns on file names.
HARD_EXCLUDED_MODEL_FILES: tuple[str, ...] = (
    "__init__.py",
    "garch*.py",
    "har*.py",
    "vol*.py",
)

_GRAPH_ROOTS: tuple[str, ...] = ("src", "tests", "scripts", "research")

_MODELS_GLOB = "src/quant_fund/models/*.py"
_ATTIC_MODELS_GLOB = "attic/**/src/quant_fund/models/*.py"
_WIRING_GLOB = "src/quant_fund/research/benches_w*.py"
_ATTIC_WIRING_GLOB = "attic/**/benches_w*.py"
_WAVE_TEST_GLOBS = ("tests/unit/models/test_w*.py", "tests/unit/research/test_benches_w*.py")

_MODULE_REF_PATTERN = re.compile(r"quant_fund(?:\.[A-Za-z0-9_]+)+")
_DYNAMIC_IMPORT_PATTERN = re.compile(r"""(?:__import__|import_module)\(\s*f?["']([^"']+)""")
_BARE_TOKEN_PATTERN = re.compile(r"""['"]([_a-z][_a-z0-9]*)['"]""")
_WAVE_TEST_PATTERN = re.compile(r"test_(?:benches_)?w\d+\.py")


def _rel_to_dotted(rel: str) -> str:
    """Repo-relative file path -> dotted module name (``src/`` stripped)."""
    name = rel[: -len(".py")] if rel.endswith(".py") else rel
    if name.endswith("/__init__"):
        name = name[: -len("/__init__")]
    if name.startswith("src/"):
        name = name[len("src/") :]
    return name.replace("/", ".")


def _dotted_to_path_map(repo_root: Path) -> dict[str, str]:
    """Map dotted module/package names to repo-relative file paths."""
    dotted: dict[str, str] = {}
    for root in _GRAPH_ROOTS:
        for path in _collect_paths(repo_root, [f"{root}/**/*.py"]):
            rel = path.relative_to(repo_root).as_posix()
            dotted[_rel_to_dotted(rel)] = rel
    if (repo_root / "conftest.py").is_file():
        dotted["conftest"] = "conftest.py"
    return dotted


def _resolve_ref(ref: str, dotted: dict[str, str]) -> str | None:
    """Resolve a dotted reference to a repo-relative file path, if any."""
    candidate = ref
    while candidate:
        if candidate in dotted:
            return dotted[candidate]
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]
    return None


def _package_prefix(rel_path: str, level: int) -> str:
    """Dotted package of *rel_path*, walked up *level* - 1 parents."""
    package = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
    dotted = package.replace("/", ".")
    for _ in range(max(level - 1, 0)):
        if "." not in dotted:
            return ""
        dotted = dotted.rsplit(".", 1)[0]
    return dotted


def _import_refs(tree: ast.Module, rel_path: str) -> set[str]:
    """Absolute and relative import targets as dotted names (best effort)."""
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = _package_prefix(rel_path, node.level) if node.level else ""
            base = ".".join(part for part in (prefix, node.module or "") if part)
            if base:
                refs.add(base)
                for alias in node.names:
                    refs.add(f"{base}.{alias.name}")
    return refs


def _string_module_refs(tree: ast.Module) -> set[str]:
    """Dotted ``quant_fund.*`` string literals that name importable modules.

    Catches lazy import maps (e.g. ``models/__init__.py``) without treating
    bare family-name strings as imports.
    """
    refs: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _MODULE_REF_PATTERN.fullmatch(node.value)
        ):
            refs.add(node.value)
    return refs


def _corpus_stem_paths(repo_root: Path) -> dict[str, str]:
    """Module stem -> live repo-relative path for loadable corpus targets.

    Covers model corpus modules, ``benches_w*`` adapters and research modules
    (the targets of dynamic loaders); live paths win over attic copies.
    """
    patterns = (
        _MODELS_GLOB,
        _WIRING_GLOB,
        "src/quant_fund/research/*.py",
        _ATTIC_MODELS_GLOB,
        _ATTIC_WIRING_GLOB,
        "attic/**/src/quant_fund/research/*.py",
    )
    stems: dict[str, str] = {}
    for path in _collect_paths(repo_root, patterns):
        rel = path.relative_to(repo_root).as_posix()
        current = stems.get(path.stem)
        if current is None or (current.startswith("attic/") and not rel.startswith("attic/")):
            stems[path.stem] = rel
    return stems


def _is_wave_test(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    if not _WAVE_TEST_PATTERN.fullmatch(name):
        return False
    return rel_path.startswith("tests/unit/models/") or rel_path.startswith("tests/unit/research/")


def _dynamic_spec_prefix(source: str) -> str | None:
    """Spec prefix of a computed-name dynamic import (e.g. ``quant_fund.models.``)."""
    for match in _DYNAMIC_IMPORT_PATTERN.finditer(source):
        spec = match.group(1)
        if "{" in spec and spec.startswith("quant_fund."):
            return spec.split("{", 1)[0]
    return None


def _evidenced_dynamic_targets(source: str, prefix: str, stems: dict[str, str]) -> set[str]:
    """Targets a computed-name loader provably loads: same-file stem literals.

    A bare string literal evidences a load only when it is a known corpus or
    research module stem under the loader's spec prefix — this resolves the
    parametrized wave-test names and literal lazy-lane names while catalog
    family-name lists create no edges (family names are not loader evidence).
    """
    targets: set[str] = set()
    for literal in _BARE_TOKEN_PATTERN.findall(source):
        rel = stems.get(literal)
        if rel is not None and _rel_to_dotted(rel).startswith(prefix):
            targets.add(rel)
    return targets


def _file_edges(
    rel: str,
    source: str,
    tree: ast.Module,
    dotted: dict[str, str],
    stems: dict[str, str],
) -> tuple[set[tuple[str, str]], tuple[str, str] | None]:
    """(edges, wildcard-loader) for one file.

    The wildcard loader is ``(rel, prefix)`` for a computed-name dynamic
    import with no literal name evidence — such a file may load anything
    under the prefix, so it blocks every candidate there.
    """
    edges: set[tuple[str, str]] = set()
    refs = _import_refs(tree, rel) | _string_module_refs(tree)
    for ref in refs:
        target = _resolve_ref(ref, dotted)
        if target is not None and target != rel:
            edges.add((rel, target))
    for match in _DYNAMIC_IMPORT_PATTERN.finditer(source):
        spec = match.group(1)
        if "{" in spec:
            continue
        target = _resolve_ref(spec, dotted)
        if target is not None and target != rel:
            edges.add((rel, target))
    prefix = _dynamic_spec_prefix(source)
    wildcard: tuple[str, str] | None = None
    if prefix is not None:
        evidenced = _evidenced_dynamic_targets(source, prefix, stems)
        if evidenced:
            for target in sorted(evidenced):
                if target != rel:
                    edges.add((rel, target))
        else:
            wildcard = (rel, prefix)
    if _is_wave_test(rel):
        for literal in _BARE_TOKEN_PATTERN.findall(source):
            target = stems.get(literal)
            if target and target != rel:
                edges.add((rel, target))
    return edges, wildcard


def scan_import_graph(repo_root: Path) -> ImportGraph:
    """Build the intra-repo import graph (file -> imported file).

    Edges come from static imports, relative imports, lazy-import-map dotted
    string references, constant-string ``importlib``/``__import__`` calls and
    dynamically loaded names evidenced by module-stem literals in the loading
    file. Computed-name loaders without literal evidence are recorded in
    ``wildcard_importers`` and block every candidate under their spec prefix.
    """
    dotted = _dotted_to_path_map(repo_root)
    stems = _corpus_stem_paths(repo_root)
    nodes: set[str] = set(dotted.values())
    edges: set[tuple[str, str]] = set()
    wildcards: set[tuple[str, str]] = set()
    for root in _GRAPH_ROOTS:
        for path in _collect_paths(repo_root, [f"{root}/**/*.py"]):
            rel = path.relative_to(repo_root).as_posix()
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            file_edges, wildcard = _file_edges(rel, source, tree, dotted, stems)
            edges.update(file_edges)
            if wildcard is not None:
                wildcards.add(wildcard)
    return ImportGraph(tuple(sorted(nodes)), tuple(sorted(edges)), tuple(sorted(wildcards)))


def _hard_excluded(rel_path: str) -> bool:
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


def _adapter_all_nonqualifying(adapter: AdapterRecord, verdicts: dict[str, str]) -> bool:
    stems = {stem for _, wired in adapter.families for stem in wired}
    return bool(stems) and all(verdicts.get(stem) == "NON_QUALIFYING_TEMPLATE" for stem in stems)


def _wave_test_paths(repo_root: Path) -> list[str]:
    paths: list[Path] = []
    for pattern in _WAVE_TEST_GLOBS:
        paths.extend(_collect_paths(repo_root, [pattern]))
    return sorted(path.relative_to(repo_root).as_posix() for path in paths)


def _file_kind(rel_path: str) -> str:
    if rel_path.startswith("tests/"):
        return "test"
    if "/benches_w" in rel_path:
        return "wiring"
    return "module"


def _candidate_tests(repo_root: Path, verdicts: dict[str, str], graph: ImportGraph) -> set[str]:
    """Wave tests exercising only non-qualifying corpus targets."""
    referenced: dict[str, set[str]] = {}
    for importer, imported in graph.edges:
        if _is_wave_test(importer):
            referenced.setdefault(importer, set()).add(imported)
    candidates: set[str] = set()
    for test in _wave_test_paths(repo_root):
        targets = referenced.get(test, set())
        if not targets:
            continue
        target_stems = {target.rsplit("/", 1)[-1][: -len(".py")] for target in targets}
        if all(verdicts.get(stem) == "NON_QUALIFYING_TEMPLATE" for stem in target_stems):
            candidates.add(test)
    return candidates


def _component_blockers(
    members: list[str],
    importers: dict[str, set[str]],
    candidates: set[str],
    comp: dict[str, int],
    staying: set[int],
    wildcard_blockers: list[tuple[str, str]],
) -> set[str]:
    """Live/staying importers that keep this component from moving."""
    blockers: set[str] = set()
    for member in members:
        for importer in sorted(importers.get(member, ())):
            if importer in candidates and comp.get(importer) not in staying:
                continue
            blockers.add(importer)
        member_dotted = _rel_to_dotted(member)
        for wildcard, prefix in wildcard_blockers:
            if member_dotted.startswith(prefix):
                blockers.add(f"{wildcard} (dynamic f-string import)")
    return blockers


def _quarantine_candidates(
    repo_root: Path,
    records: list[ModuleRecord],
    verdicts: dict[str, str],
    adapters: list[AdapterRecord],
    graph: ImportGraph,
) -> tuple[set[str], set[str], set[str]]:
    candidate_modules = {
        record.path
        for record in records
        if record.path.startswith("src/")
        and verdicts.get(record.stem) == "NON_QUALIFYING_TEMPLATE"
        and not _hard_excluded(record.path)
    }
    candidate_adapters = {
        adapter.path
        for adapter in adapters
        if adapter.location == "live" and _adapter_all_nonqualifying(adapter, verdicts)
    }
    candidate_tests = _candidate_tests(repo_root, verdicts, graph)
    return candidate_modules, candidate_adapters, candidate_tests


def _blocking_fixpoint(
    candidates: set[str],
    comp: dict[str, int],
    members: dict[int, list[str]],
    importers: dict[str, set[str]],
    wildcard_blockers: list[tuple[str, str]],
) -> tuple[set[int], dict[int, set[str]]]:
    staying: set[int] = set()
    reasons: dict[int, set[str]] = {}
    changed = True
    while changed:
        changed = False
        for cid in sorted(members):
            if cid in staying:
                continue
            blockers = _component_blockers(
                members[cid], importers, candidates, comp, staying, wildcard_blockers
            )
            if blockers:
                staying.add(cid)
                reasons[cid] = blockers
                changed = True
    return staying, reasons


def quarantine_plan(
    repo_root: Path,
    records: list[ModuleRecord],
    verdicts: dict[str, str],
    adapters: list[AdapterRecord],
    graph: ImportGraph,
) -> dict[str, Any]:
    """Decide which candidate files may move, closing under the import graph.

    Candidates: ``NON_QUALIFYING_TEMPLATE`` model modules (minus owner-guarded
    files), ``benches_w*`` adapters wiring only non-qualifying modules, and
    wave tests exercising only non-qualifying targets. A candidate moves only
    if every importer moves with it and its whole connected component is
    movable; anything else stays live with the blocking importer named.
    """
    candidate_modules, candidate_adapters, candidate_tests = _quarantine_candidates(
        repo_root, records, verdicts, adapters, graph
    )
    candidates = candidate_modules | candidate_adapters | candidate_tests
    importers: dict[str, set[str]] = {}
    for importer, imported in graph.edges:
        importers.setdefault(imported, set()).add(importer)
    internal = [edge for edge in graph.edges if edge[0] in candidates and edge[1] in candidates]
    comp = _component_map(candidates, internal)
    members: dict[int, list[str]] = {cid: [] for cid in sorted(set(comp.values()))}
    for node, cid in comp.items():
        members[cid].append(node)
    wildcard_blockers = sorted((w, p) for w, p in graph.wildcard_importers if w not in candidates)
    staying, reasons = _blocking_fixpoint(candidates, comp, members, importers, wildcard_blockers)
    return {
        "candidate_modules": sorted(candidate_modules),
        "candidate_adapters": sorted(candidate_adapters),
        "candidate_tests": sorted(candidate_tests),
        "moving": sorted(node for node in candidates if comp[node] not in staying),
        "staying_live": sorted(node for node in candidates if comp[node] in staying),
        "outstanding_quarantine": [
            {
                "path": node,
                "kind": _file_kind(node),
                "component_id": comp[node],
                "blocking_importers": sorted(reasons[comp[node]]),
            }
            for node in sorted(candidate_modules | candidate_adapters)
            if comp[node] in staying
        ],
        "components": [
            {
                "component_id": cid,
                "decision": "BLOCKED" if cid in staying else "MOVE",
                "members": sorted(members[cid]),
                "blocking_importers": sorted(reasons.get(cid, set())),
            }
            for cid in sorted(members)
        ],
    }


def _totals(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    physical = 0
    nonblank = 0
    for row in rows:
        physical += int(row["physical_lines"])
        nonblank += int(row["nonblank_lines"])
    return {"physical_lines": physical, "nonblank_lines": nonblank}


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


def _load_records(repo_root: Path) -> tuple[list[ModuleRecord], dict[str, int], dict[str, int]]:
    population = RULESET["population"]
    assert isinstance(population, dict)
    excluded = set(population["excluded_stems"])
    records: list[ModuleRecord] = []
    shape_counts: dict[str, int] = {}
    skeleton_counts: dict[str, int] = {}
    for path in _collect_paths(repo_root, population["models_globs"]):
        if path.stem in excluded:
            continue
        record = analyze_module(path, path.read_text(encoding="utf-8"), repo_root)
        records.append(record)
        shape_counts[record.ast_shape_hash] = shape_counts.get(record.ast_shape_hash, 0) + 1
        skeleton_counts[record.docstring_skeleton] = (
            skeleton_counts.get(record.docstring_skeleton, 0) + 1
        )
    return records, shape_counts, skeleton_counts


def _load_adapters(repo_root: Path) -> list[AdapterRecord]:
    population = RULESET["population"]
    assert isinstance(population, dict)
    return [
        scan_adapter(path, path.read_text(encoding="utf-8"), repo_root)
        for path in _collect_paths(repo_root, population["wiring_globs"])
    ]


def _family_maps(
    adapters: list[AdapterRecord],
) -> tuple[dict[str, set[str]], dict[str, set[int]]]:
    family_wiring: dict[str, set[str]] = {}
    family_waves: dict[str, set[int]] = {}
    for adapter in adapters:
        for family, stems in adapter.families:
            family_wiring.setdefault(family, set()).update(stems)
            if adapter.wave is not None:
                family_waves.setdefault(family, set()).add(adapter.wave)
    return family_wiring, family_waves


def _files_payload(
    records: list[ModuleRecord],
    shape_counts: dict[str, int],
    skeleton_counts: dict[str, int],
    family_wiring: dict[str, set[str]],
    family_waves: dict[str, set[int]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    thresholds = RULESET["thresholds"]
    assert isinstance(thresholds, dict)
    shape_min = int(thresholds["ast_shape_shared_min"])
    skeleton_min = int(thresholds["docstring_skeleton_shared_min"])
    files: list[dict[str, Any]] = []
    verdicts: dict[str, str] = {}
    for record in records:
        verdict = verdict_for(
            record,
            shape_counts[record.ast_shape_hash] >= shape_min,
            skeleton_counts[record.docstring_skeleton] >= skeleton_min,
        )
        verdicts[record.stem] = verdict.verdict
        families = sorted(f for f, stems in family_wiring.items() if record.stem in stems)
        waves: set[int] = set()
        for family in families:
            waves |= family_waves.get(family, set())
        payload = _record_payload(record)
        payload["location"] = "attic" if record.path.startswith("attic/") else "live"
        payload["verdict"] = verdict.verdict
        payload["fired_rules"] = list(verdict.fired_rules)
        payload["corroborating"] = list(verdict.corroborating)
        payload["family_ids"] = families
        payload["waves"] = sorted(waves)
        files.append(payload)
    return files, verdicts


def _families_payload(
    family_wiring: dict[str, set[str]],
    family_waves: dict[str, set[int]],
    verdicts: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted(family_wiring):
        stems = sorted(family_wiring[family])
        module_verdicts = sorted({verdicts.get(stem, "UNRESOLVED") for stem in stems})
        classification = (
            "RETIRED_CANDIDATE"
            if "NON_QUALIFYING_TEMPLATE" in module_verdicts
            else "LIVE_CANDIDATE"
        )
        rows.append(
            {
                "family_id": family,
                "modules": stems,
                "waves": sorted(family_waves.get(family, set())),
                "module_verdicts": module_verdicts,
                "classification": classification,
            }
        )
    return rows


def _counts_block(
    files: list[dict[str, Any]],
    families_payload: list[dict[str, Any]],
    adapters: list[AdapterRecord],
    plan: dict[str, Any],
) -> dict[str, Any]:
    live_rows = [row for row in files if row["location"] == "live"]
    attic_rows = [row for row in files if row["location"] == "attic"]

    def count_verdict(rows: list[dict[str, Any]], verdict: str) -> int:
        return sum(1 for row in rows if row["verdict"] == verdict)

    return {
        "baseline_corpus": {
            "modules": len(files),
            "qualifying": count_verdict(files, "QUALIFYING"),
            "non_qualifying_template": count_verdict(files, "NON_QUALIFYING_TEMPLATE"),
            **_totals(files),
        },
        "live_tree": {
            "modules": len(live_rows),
            "qualifying": count_verdict(live_rows, "QUALIFYING"),
            "non_qualifying_template": count_verdict(live_rows, "NON_QUALIFYING_TEMPLATE"),
            **_totals(live_rows),
        },
        "attic": {"modules": len(attic_rows), **_totals(attic_rows)},
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
            "moving_adapters": sum(1 for p in plan["moving"] if _file_kind(p) == "wiring"),
            "staying_adapters": sum(1 for p in plan["staying_live"] if _file_kind(p) == "wiring"),
        },
        "tests": {
            "wave_tests": len(plan["candidate_tests"]),
            "moving_wave_tests": sum(1 for p in plan["moving"] if _file_kind(p) == "test"),
            "staying_wave_tests": sum(1 for p in plan["staying_live"] if _file_kind(p) == "test"),
        },
        "quarantine": {
            "candidates": len(plan["candidate_modules"])
            + len(plan["candidate_adapters"])
            + len(plan["candidate_tests"]),
            "moving": len(plan["moving"]),
            "staying_live": len(plan["staying_live"]),
            "blocked_components": sum(1 for c in plan["components"] if c["decision"] == "BLOCKED"),
        },
    }


def audit_tree(repo_root: Path) -> dict[str, Any]:
    """Build the full deterministic audit document for *repo_root*."""
    records, shape_counts, skeleton_counts = _load_records(repo_root)
    adapters = _load_adapters(repo_root)
    family_wiring, family_waves = _family_maps(adapters)
    files, verdicts = _files_payload(
        records, shape_counts, skeleton_counts, family_wiring, family_waves
    )
    graph = scan_import_graph(repo_root)
    plan = quarantine_plan(repo_root, records, verdicts, adapters, graph)
    families_payload = _families_payload(family_wiring, family_waves, verdicts)
    outstanding = [
        {
            **row,
            "verdict": verdicts.get(row["path"].rsplit("/", 1)[-1][: -len(".py")], "UNRESOLVED"),
        }
        for row in plan["outstanding_quarantine"]
    ]
    candidate_paths = set(plan["moving"]) | set(plan["staying_live"])
    return {
        "schema": "canon_qualification_audit.v1",
        "ruleset_version": RULESET_VERSION,
        "ruleset_hash": ruleset_hash(),
        "ruleset": RULESET,
        "counts": _counts_block(files, families_payload, adapters, plan),
        "outstanding_quarantine": outstanding,
        "files": files,
        "families": families_payload,
        "wiring": [
            {
                "path": adapter.path,
                "wave": adapter.wave,
                "location": adapter.location,
                "families": [
                    {"family_id": family, "modules": list(fam_stems)}
                    for family, fam_stems in adapter.families
                ],
            }
            for adapter in adapters
        ],
        "import_graph": {
            "edges": [
                [importer, imported]
                for importer, imported in graph.edges
                if importer in candidate_paths or imported in candidate_paths
            ],
            "wildcard_importers": [list(pair) for pair in graph.wildcard_importers],
            "quarantine": plan,
        },
    }


def dumps_audit(audit: dict[str, Any]) -> str:
    """Canonical deterministic JSON serialization (byte-stable across runs)."""
    return json.dumps(audit, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


__all__ = [
    "HARD_EXCLUDED_MODEL_FILES",
    "RULESET",
    "RULESET_VERSION",
    "AdapterRecord",
    "ImportGraph",
    "ModuleRecord",
    "Verdict",
    "analyze_module",
    "ast_shape_hash",
    "audit_tree",
    "bench_profile",
    "count_control_flow",
    "data_param_count",
    "docstring_skeleton",
    "dumps_audit",
    "effective_body_lines",
    "nontrivial_imports",
    "quarantine_plan",
    "ruleset_hash",
    "scan_adapter",
    "scan_import_graph",
    "verdict_for",
]
