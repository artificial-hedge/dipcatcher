"""Ratchet: no union-merge damage signatures may live in src/.

A batch conflict resolver once concatenated both sides of code hunks,
which shipped doubled receipt writes, drifted contract checks, and dead
tails referencing undefined names to main. These three signatures are
mechanically detectable and always wrong:

1. Two adjacent identical statements — one side of a unioned hunk kept
   verbatim next to itself.
2. ``if``/``elif``/``else`` bodies containing only imports with no
   TYPE_CHECKING / sys.platform guard — a contract-dispatch branch
   whose real body was lost.
3. Statements after ``return``/``raise`` in a function body — a tail
   spliced in after the original epilogue.

Intentional idempotency re-asserts live in tests/ — this ratchet covers
src/ only, where a duplicated statement is always an artifact.
"""

from __future__ import annotations

import ast
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

_BENIGN_DUP_PARENTS = (ast.Expr, ast.Assign)  # docstrings, re-assignments
_SKIP_LOC = frozenset({"lineno", "col_offset", "end_lineno", "end_col_offset"})
_SCANS: tuple[list[str], list[str], list[str]] | None = None


def _stmt_eq(left: ast.AST, right: ast.AST) -> bool:
    """Same tree ``ast.dump`` would compare, without building the strings."""
    if type(left) is not type(right):
        return False
    left_fields = [(name, value) for name, value in ast.iter_fields(left) if name not in _SKIP_LOC]
    right_fields = [
        (name, value) for name, value in ast.iter_fields(right) if name not in _SKIP_LOC
    ]
    if len(left_fields) != len(right_fields):
        return False
    for (left_name, left_value), (right_name, right_value) in zip(
        left_fields, right_fields, strict=True
    ):
        if left_name != right_name or not _value_eq(left_value, right_value):
            return False
    return True


def _value_eq(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, ast.AST) and isinstance(right, ast.AST):
        return _stmt_eq(left, right)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _value_eq(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _scan_file(rel: str) -> tuple[list[str], list[str], list[str]]:
    tree = ast.parse((SRC / rel).read_text())
    duplicates: list[str] = []
    import_only: list[str] = []
    tails: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.dump(node.test)
            legit = (
                "TYPE_CHECKING",
                "attr='platform'",  # sys.platform
                "attr='version_info'",
                "attr='name'",  # os.name
                "find_spec",
                "getenv",
            )
            if not any(tok in test_src for tok in legit):
                for body in (node.body, node.orelse):
                    if body and all(
                        isinstance(stmt, (ast.Import, ast.ImportFrom)) for stmt in body
                    ):
                        import_only.append(f"{rel}:{node.lineno}")
        for attr in ("body", "orelse", "finalbody"):
            stmts = getattr(node, attr, None)
            if not isinstance(stmts, list) or not stmts:
                continue
            for i, stmt in enumerate(stmts[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise)) and not isinstance(
                    stmts[i + 1], (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
                ):
                    tails.append(f"{rel}:{stmts[i + 1].lineno}")
                    break
            for left, right in zip(stmts, stmts[1:], strict=False):
                if type(left) is not type(right) or isinstance(
                    left, (ast.Import, ast.ImportFrom, ast.Pass, *_BENIGN_DUP_PARENTS)
                ):
                    continue
                if _stmt_eq(left, right):
                    duplicates.append(f"{rel}:{left.lineno}")
    return duplicates, import_only, tails


def _scans() -> tuple[list[str], list[str], list[str]]:
    global _SCANS
    if _SCANS is None:
        rels = sorted(path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py"))
        workers = min(4, os.cpu_count() or 1)
        duplicates: list[str] = []
        import_only: list[str] = []
        tails: list[str] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for found_dup, found_imp, found_tail in pool.map(_scan_file, rels, chunksize=64):
                duplicates.extend(found_dup)
                import_only.extend(found_imp)
                tails.extend(found_tail)
        _SCANS = (duplicates, import_only, tails)
    return _SCANS


def test_src_has_no_adjacent_duplicate_statements() -> None:
    offenders, _, _ = _scans()
    assert offenders == [], (
        "adjacent duplicate statements (union-merge signature) in src/:\n" + "\n".join(offenders)
    )


def test_src_has_no_import_only_branches() -> None:
    _, offenders, _ = _scans()
    assert offenders == [], (
        "if/else bodies containing only imports (lost contract body "
        "signature) in src/:\n" + "\n".join(offenders)
    )


def test_src_has_no_unreachable_tail_statements() -> None:
    _, _, offenders = _scans()
    assert offenders == [], (
        "unreachable statements after return/raise (union-merge signature) "
        "in src/:\n" + "\n".join(offenders)
    )
