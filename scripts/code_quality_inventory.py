#!/usr/bin/env python3
"""Measure reviewable Python code without rewarding generated padding.

This inventory deliberately reports several observable quantities instead of
calling every physical line "quality code".  It only reads files tracked by
Git, excludes generated paths, and uses Python's tokenizer and AST so blank
lines, comments, and docstrings cannot inflate the source-line count.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ("src", "tests", "scripts")
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "vendor",
    }
)


@dataclass(frozen=True)
class FileMetrics:
    path: str
    physical_lines: int
    source_lines: int
    comment_lines: int
    docstring_lines: int
    functions: int
    classes: int
    test_functions: int


@dataclass(frozen=True)
class Inventory:
    schema_version: str
    files: int
    physical_lines: int
    source_lines: int
    comment_lines: int
    docstring_lines: int
    functions: int
    classes: int
    test_functions: int
    minimum_source_lines: int | None
    meets_minimum: bool | None
    entries: tuple[FileMetrics, ...]


def tracked_python_files(repo: Path, roots: Sequence[str] = DEFAULT_ROOTS) -> list[Path]:
    """Return deterministic, tracked Python inputs below *roots*.

    Using ``git ls-files`` prevents ignored caches, dependencies, generated
    output, and an accidentally copied virtual environment from changing the
    result.  Explicit excluded path components provide defense in depth.
    """
    command = ["git", "ls-files", "-z", "--", *roots]
    result = subprocess.run(command, cwd=repo, check=True, capture_output=True)
    relative = [Path(raw.decode()) for raw in result.stdout.split(b"\0") if raw]
    return sorted(
        repo / path
        for path in relative
        if path.suffix == ".py" and not EXCLUDED_PARTS.intersection(path.parts)
    )


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    owners: Iterable[ast.AST] = ast.walk(tree)
    for owner in owners:
        if not isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not owner.body:
            continue
        first = owner.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def measure_file(path: Path, *, repo: Path = ROOT) -> FileMetrics:
    """Measure one syntactically valid Python file."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    docstrings = _docstring_lines(tree)
    comments: set[int] = set()
    token_lines: set[int] = set()
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type == tokenize.COMMENT:
                comments.add(token.start[0])
            elif token.type not in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.NEWLINE,
                tokenize.NL,
            }:
                token_lines.update(range(token.start[0], token.end[0] + 1))

    functions = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    classes = sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    test_functions = sum(node.name.startswith("test_") for node in functions)
    source_lines = token_lines - docstrings
    return FileMetrics(
        path=path.relative_to(repo).as_posix(),
        physical_lines=len(text.splitlines()),
        source_lines=len(source_lines),
        comment_lines=len(comments - docstrings),
        docstring_lines=len(docstrings),
        functions=len(functions),
        classes=classes,
        test_functions=test_functions,
    )


def build_inventory(
    repo: Path = ROOT,
    *,
    roots: Sequence[str] = DEFAULT_ROOTS,
    minimum_source_lines: int | None = None,
) -> Inventory:
    if minimum_source_lines is not None and minimum_source_lines < 0:
        raise ValueError("minimum_source_lines must be non-negative")
    entries = tuple(measure_file(path, repo=repo) for path in tracked_python_files(repo, roots))

    def total(field: str) -> int:
        return sum(int(getattr(entry, field)) for entry in entries)

    source_lines = total("source_lines")
    return Inventory(
        schema_version="code-quality-inventory/v1",
        files=len(entries),
        physical_lines=total("physical_lines"),
        source_lines=source_lines,
        comment_lines=total("comment_lines"),
        docstring_lines=total("docstring_lines"),
        functions=total("functions"),
        classes=total("classes"),
        test_functions=total("test_functions"),
        minimum_source_lines=minimum_source_lines,
        meets_minimum=None
        if minimum_source_lines is None
        else source_lines >= minimum_source_lines,
        entries=entries,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--root", action="append", dest="roots", help="tracked subtree to include")
    parser.add_argument(
        "--minimum", type=int, help="fail when semantic source lines are below this"
    )
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--summary-only", action="store_true", help="omit per-file entries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inventory = build_inventory(
        args.repo.resolve(),
        roots=tuple(args.roots or DEFAULT_ROOTS),
        minimum_source_lines=args.minimum,
    )
    payload = asdict(inventory)
    if args.summary_only:
        payload.pop("entries")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 1 if inventory.meets_minimum is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
