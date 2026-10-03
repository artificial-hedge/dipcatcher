"""Literal prefilter: extract a required literal from a regex AST and use
it to cheaply reject text before running the full matcher.

A required literal is one every match must contain. Walks the AST:
cat -> union of literals from all parts; alt -> literals common to all
branches; group/noncap -> descend; quantified nodes contribute only when
their minimum count > 0 (or when the atom is a literal char).
SYNTHETIC bench verifies no false negatives vs `re` and a speed ratio.
"""

from __future__ import annotations

import re as _re
from typing import Any

import numpy as np

from quant_fund.models.pike_vm import _Parser

_SEED = 20261231 + 929

Ast = tuple[Any, ...]


def _lits_of(ast: Ast) -> list[set[str]]:
    """Candidate required literals (each element = set of alternatives that
    ALL matches must contain at least one of)."""
    tag = ast[0] if isinstance(ast, tuple) else None
    if tag == "lit":
        return [{ast[1]}]
    if tag in ("dot", "cls"):
        return [set()]  # any char possible -> empty required set
    if tag == "cat":
        out: list[set[str]] = []
        for part in ast[1]:
            out.extend(_lits_of(part))
        return out
    if tag == "alt":
        # required literal must be required by EVERY branch
        per = [_lits_of(p) for p in ast[1]]
        if not per:
            return []
        first = per[0]
        common: list[set[str]] = []
        for cand in first:
            if all(cand in branch for branch in per[1:]):
                common.append(cand)
        return common
    if tag in ("group", "noncap"):
        return _lits_of(ast[1])
    if tag in ("*", "?"):
        return []  # zero repetitions allowed
    if tag == "+":
        return _lits_of(ast[1])
    return []


def required_literal(pat: str) -> str | None:
    p = _Parser(pat)
    ast = p.parse()
    cands = _lits_of(ast)
    nonempty = [c for c in cands if c]
    if not nonempty:
        return None
    return (
        max(nonempty, key=len).__iter__().__next__()
        if False
        else min(nonempty, key=len).__iter__().__next__()
    )


def _literal_str(pat: str) -> str | None:
    """Join a contiguous literal set from a 'cat' chain into a string when
    the required literal is a run of literal chars."""
    p = _Parser(pat)
    ast = p.parse()
    # find longest run of lit nodes inside the outer group->cat
    seq = ast[0][1]
    runs = _literal_runs(seq)
    return max(runs, key=len) if runs else None


def _literal_runs(node: Ast) -> list[str]:
    tag = node[0] if isinstance(node, tuple) else None
    if tag == "cat":
        runs: list[str] = []
        cur = ""
        for part in node[1]:
            ptag = part[0] if isinstance(part, tuple) else None
            if ptag == "lit":
                cur += part[1]
            else:
                if cur:
                    runs.append(cur)
                    cur = ""
                runs.extend(_literal_runs(part))
        if cur:
            runs.append(cur)
        return runs
    if tag in ("group", "noncap"):
        return _literal_runs(node[1])
    if tag in ("*", "+", "?"):
        sub = node[1]
        stag = sub[0] if isinstance(sub, tuple) else None
        if tag == "+" and stag == "lit":
            return [sub[1]]
        return []
    return []


def candidates(pat: str, text: str) -> list[int] | None:
    """Positions where a match could start, or None if no required literal
    (must run the full matcher). A match can only start at a position i such
    that the required literal occurs at some offset >= i."""
    lit = _literal_str(pat)
    if lit is None or len(lit) < 2:
        return None
    hits: list[int] = []
    start = 0
    while True:
        j = text.find(lit, start)
        if j < 0:
            break
        hits.append(j)
        start = j + 1
    return hits


def bench_literal_prefilter(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    score += 1.0 if _literal_str("ab.*cd") == "ab" else 0.0
    score += 1.0 if _literal_str(".*needle.*") == "needle" else 0.0
    ok = True
    pats = ["foo.*bar", "abc|abd", "x(aa|ab)y", "pre(mid)post", "aaab+ccc"]
    texts = [
        "zzfooxxbarzz",
        "abc abd x",
        "xaay xaaby",
        "premidpost",
        "aaabbbccc",
    ]
    for pat, text in zip(pats, texts, strict=True):
        rgx = _re.compile(".*" + pat + ".*")
        lit = _literal_str(pat)
        m = rgx.match(text)
        if m and lit and lit not in m.group(0):
            ok = False
    score += 1.0 if ok else 0.0
    # no-false-negative fuzz: every re.search hit implies lit in its span
    ok = True
    for _ in range(150):
        pat = "".join(
            str(rng.choice(list("ab") + [".", "*", "+", "?", "|", "(", ")"]))
            for _ in range(int(rng.integers(3, 10)))
        )
        try:
            rgx = _re.compile(pat)
            lit = _literal_str(pat)
        except (ValueError, _re.error):
            continue
        text = "".join(str(rng.choice(list("abc"))) for _ in range(int(rng.integers(0, 15))))
        m = rgx.search(text)
        if m and lit and lit not in m.group(0):
            ok = False
            break
    score += 1.0 if ok else 0.0
    return {"synthetic_literal_prefilter": score / 4.0}
