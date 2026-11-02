"""Tests for ``scripts/mutation_spotcheck.py``.

The spot-check harness is a hand-rolled AST mutator for the money paths: it
enumerates single-token mutations, applies exactly one at a time, runs a test
slice and records kill/survive. A mutant that *survives* is a hole in the
test suite, so the enumerator itself has to be trustworthy.

The properties pinned here:

* the enumerator finds the mutation it is asked for, and only that one, on a
  given source;
* the mutation is a real, minimal change to the money term (a flipped
  comparison, a swapped arithmetic operator, a dropped ``abs()``);
* operator sets filter what is enumerated — a ``cmp``-only target never
  produces an arithmetic mutant;
* ``enumerate_mutants()`` produces well-formed, uniquely numbered mutants for
  the curated target files.

``apply_mutant`` is deliberately **not** exercised here: it rewrites files
under ``src/``, and a unit test must never mutate the tree it runs against.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "mutation_spotcheck.py"
_spec = importlib.util.spec_from_file_location("gap_mutation_spotcheck", _SCRIPT_PATH)
assert _spec and _spec.loader, "mutation_spotcheck.py must be importable"
mod = importlib.util.module_from_spec(_spec)
sys.modules["gap_mutation_spotcheck"] = mod
_spec.loader.exec_module(mod)

SOURCE = """\
def f(a, b):
    c = a + b
    if a < b:
        d = abs(a)
    e = a
    e += b
    return c
"""


def _mutants(source: str, qualname: str, ops: set[str]):
    enum = mod._Enumerator(source, qualname, None, ops)
    enum.visit(ast.parse(source))
    return enum.mutants


def _kinds(source: str, ops: set[str]) -> dict[str, str]:
    return {mutant.kind: mutant.mutated for mutant in _mutants(source, "f", ops)}


# --------------------------------------------------------------------------- #
# Call-name resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("abs(x)", "abs"),
        ("np.sqrt(x)", "np.sqrt"),
        ("a.b.c(x)", "a.b.c"),
    ],
)
def test_call_names_are_resolved(expression: str, expected: str) -> None:
    node = ast.parse(expression).body[0]
    assert isinstance(node, ast.Expr)
    assert mod._call_name(node.value.func) == expected  # type: ignore[attr-defined]


def test_a_non_name_call_has_no_name() -> None:
    node = ast.parse("f()()").body[0]
    assert isinstance(node, ast.Expr)
    assert mod._call_name(node.value.func) is None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Offset conversion
# --------------------------------------------------------------------------- #


def test_offsets_are_converted_from_ast_columns() -> None:
    """AST columns are UTF-8 byte offsets, not character offsets."""
    lines = ["x = 1\n", "y = 2\n"]
    assert mod._offset(lines, 1, 0) == 0
    assert mod._offset(lines, 2, 0) == 6
    assert mod._offset(lines, 1, 4) == 4


def test_offsets_survive_multibyte_characters() -> None:
    lines = ["s = 'é'\n", "y = 2\n"]
    # The AST column of the second line is a byte offset; the conversion must
    # land on the character boundary anyway.
    assert mod._offset(lines, 2, 0) == len(lines[0])


# --------------------------------------------------------------------------- #
# Mutation operators
# --------------------------------------------------------------------------- #


def test_a_comparison_is_complemented() -> None:
    kinds = _kinds(SOURCE, {"cmp"})
    assert "cmp-complement" in kinds
    assert ">=" in kinds["cmp-complement"]


def test_arithmetic_operators_are_swapped() -> None:
    kinds = _kinds(SOURCE, {"arith"})
    assert "arith-add" in kinds
    assert "-" in kinds["arith-add"]


def test_a_call_is_dropped_to_its_argument() -> None:
    kinds = _kinds(SOURCE, {"call"})
    assert "call-drop" in kinds


def test_a_negation_is_dropped() -> None:
    source = "def f(a):\n    return -a\n"
    kinds = _kinds(source, {"neg"})
    assert "neg-drop" in kinds


def test_an_augmented_assignment_is_swapped() -> None:
    kinds = _kinds(SOURCE, {"aug"})
    assert "aug-swap" in kinds
    assert "-=" in kinds["aug-swap"]


def test_a_returned_expression_is_negated() -> None:
    kinds = _kinds(SOURCE, {"ret"})
    assert "ret-negate" in kinds
    assert kinds["ret-negate"].startswith("return -(")


def test_min_and_max_calls_are_swapped() -> None:
    source = "def f(a, b):\n    return min(a, b)\n"
    kinds = _kinds(source, {"call"})
    assert "call-swap" in kinds
    assert kinds["call-swap"].startswith("max")


def test_isfinite_is_inverted() -> None:
    source = "def f(a):\n    return np.isfinite(a)\n"
    kinds = _kinds(source, {"call"})
    assert "call-invert" in kinds
    assert kinds["call-invert"].startswith("(not ")


def test_np_sign_is_negated() -> None:
    source = "def f(a):\n    return np.sign(a)\n"
    kinds = _kinds(source, {"call"})
    assert "call-negate" in kinds


def test_the_operator_set_filters_the_enumeration() -> None:
    """A ``cmp``-only target never produces an arithmetic mutant."""
    kinds = _kinds(SOURCE, {"cmp"})
    assert set(kinds) == {"cmp-complement"}
    assert not any(kind.startswith("arith") for kind in kinds)


def test_a_qualname_outside_the_target_produces_nothing() -> None:
    enum = mod._Enumerator(SOURCE, "other", None, {"arith", "cmp", "call", "aug", "ret"})
    enum.visit(ast.parse(SOURCE))
    assert enum.mutants == []


def test_a_line_window_restricts_the_enumeration() -> None:
    """A target pinned to a line range only mutates inside that range."""
    enum = mod._Enumerator(SOURCE, "f", (1, 1), {"arith"})
    enum.visit(ast.parse(SOURCE))
    assert enum.mutants == []


def test_every_mutant_records_a_position_and_a_changed_expression() -> None:
    for mutant in _mutants(SOURCE, "f", {"cmp", "arith", "call", "aug", "ret"}):
        assert mutant.original
        assert mutant.mutated
        assert mutant.mutated != mutant.original
        assert mutant.span[0] >= 1


# --------------------------------------------------------------------------- #
# The curated target list
# --------------------------------------------------------------------------- #


def test_the_curated_targets_all_point_at_real_source_files() -> None:
    for file, _qualname, _window, _ops in mod.TARGETS:
        assert (mod.REPO / file).is_file(), file


def test_enumerate_mutants_produces_uniquely_numbered_mutants() -> None:
    mutants = mod.enumerate_mutants()
    assert mutants, "the curated target list must yield at least one mutant"
    ids = [mutant.mid for mutant in mutants]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert ids[0] == "M001"
    assert all(mutant.file in {file for file, _, _, _ in mod.TARGETS} for mutant in mutants)
    assert all(mutant.original != mutant.mutated for mutant in mutants)
