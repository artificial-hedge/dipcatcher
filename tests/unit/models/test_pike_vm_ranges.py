"""Probe: char-class ranges must use the last parsed char as the lower
bound. The buggy form popped an arbitrary set element (hash-seed order),
giving nondeterministic regex semantics."""

from __future__ import annotations

import re

from quant_fund.models.pike_vm import compile, run


def _matches(pat: str, text: str) -> bool:
    prog, ng = compile(pat)
    return run(prog, ng, text) is not None


def test_range_after_multi_char_class() -> None:
    """[za-b]: range is a-b (last parsed char is 'a'), plus literal 'z'.
    'b' must match; buggy pop can take 'z' -> empty range -> 'b' rejected."""
    assert _matches("[za-b]", "b")
    assert _matches("[za-b]", "a")
    assert _matches("[za-b]", "z")
    assert not _matches("[za-b]", "c")


def test_class_battery_against_re() -> None:
    for pat in ["[ab-d]", "[x0-9]+y", "[^a-c]+", "[a-c-e]+"]:
        for t in ["a", "b", "c", "d", "e", "x9y", "z", "-", "ac", "de"]:
            try:
                py = re.fullmatch(pat, t) is not None
            except re.error:
                continue
            assert _matches(pat, t) == py, f"{pat!r} vs {t!r}"


def test_dash_edge_cases() -> None:
    assert _matches("[a-]", "-")
    assert _matches("[-a]", "-")
    assert _matches("[a-z-9]", "-")
    assert _matches("[a-z-9]", "9")
    assert not _matches("[a-z-9]", "0")
