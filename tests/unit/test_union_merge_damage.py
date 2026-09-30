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
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

_BENIGN_DUP_PARENTS = (ast.Expr, ast.Assign)  # docstrings, re-assignments


def _iter_bodies(tree: ast.AST):
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            stmts = getattr(node, attr, None)
            if isinstance(stmts, list) and stmts:
                yield node, stmts


def test_src_has_no_adjacent_duplicate_statements() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for _, stmts in _iter_bodies(tree):
            for a, b in zip(stmts, stmts[1:], strict=False):
                if (
                    type(a) is type(b)
                    and ast.dump(a) == ast.dump(b)
                    and not isinstance(
                        a, (ast.Import, ast.ImportFrom, ast.Pass, *_BENIGN_DUP_PARENTS)
                    )
                ):
                    offenders.append(f"{path.relative_to(SRC)}:{a.lineno}")
    assert offenders == [], (
        "adjacent duplicate statements (union-merge signature) in src/:\n" + "\n".join(offenders)
    )


def test_src_has_no_import_only_branches() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.dump(node.test)
            # platform / typing-time / optional-dependency guards are the
            # legitimate users of an import-only branch
            legit = (
                "TYPE_CHECKING",
                "attr='platform'",  # sys.platform
                "attr='version_info'",
                "attr='name'",  # os.name
                "find_spec",
                "getenv",
            )
            if any(tok in test_src for tok in legit):
                continue
            for body in (node.body, node.orelse):
                if body and all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in body):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == [], (
        "if/else bodies containing only imports (lost contract body "
        "signature) in src/:\n" + "\n".join(offenders)
    )


def test_src_has_no_unreachable_tail_statements() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for _, stmts in _iter_bodies(tree):
            for i, stmt in enumerate(stmts[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise)) and not isinstance(
                    stmts[i + 1], (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
                ):
                    offenders.append(f"{path.relative_to(SRC)}:{stmts[i + 1].lineno}")
                    break
    assert offenders == [], (
        "unreachable statements after return/raise (union-merge signature) "
        "in src/:\n" + "\n".join(offenders)
    )
