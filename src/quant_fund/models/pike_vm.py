"""Pike VM: Thompson NFA simulation with submatch capture slots.

Supports literals, '.', '*', '+', '?', '|', '()', character classes [..] and
[^..]. Compiled program alternates Char/Split/Save/Match instructions.
SYNTHETIC bench: capture groups agree with Python `re` on a battery.
"""

from __future__ import annotations

import re as _re
from typing import Any

import numpy as np

_SEED = 20261231 + 926

CHAR, DOT, CLS, MATCH, SPLIT, SAVE, JMP = range(7)

_Inst = tuple[int, Any, Any]


class _Parser:
    def __init__(self, pat: str) -> None:
        self.pat = pat
        self.i = 0
        self.ngroup = 0

    def parse(self) -> Any:
        node = self._alt()
        if self.i != len(self.pat):
            raise ValueError(f"trailing pattern {self.pat[self.i :]}")
        return [("group", node, 0)]

    def _alt(self) -> Any:
        parts = [self._seq()]
        while self.i < len(self.pat) and self.pat[self.i] == "|":
            self.i += 1
            parts.append(self._seq())
        return parts[0] if len(parts) == 1 else ("alt", parts)

    def _seq(self) -> Any:
        out: list[object] = []
        while self.i < len(self.pat) and self.pat[self.i] not in ")|":
            out.append(self._rep())
        return ("cat", out)

    def _rep(self) -> Any:
        a = self._atom()
        nq = 0
        while self.i < len(self.pat) and self.pat[self.i] in "*+?":
            nq += 1
            if nq > 1:
                raise ValueError("stacked quantifiers unsupported")
            a = (self.pat[self.i], a)
            self.i += 1
        return a

    def _atom(self) -> Any:
        c = self.pat[self.i]
        if c == "(":
            self.i += 1
            if self.pat[self.i : self.i + 2] == "?:":
                self.i += 2
                node = ("noncap", self._alt(), -1)
            else:
                self.ngroup += 1
                node = ("group", self._alt(), self.ngroup)
            if self.i >= len(self.pat) or self.pat[self.i] != ")":
                raise ValueError("unbalanced (")
            self.i += 1
            return node
        if c == "[":
            j = self.i + 1
            neg = j < len(self.pat) and self.pat[j] == "^"
            if neg:
                j += 1
            chars: set[str] = set()
            while j < len(self.pat) and self.pat[j] != "]":
                if (
                    self.pat[j] == "-"
                    and chars
                    and j + 1 < len(self.pat)
                    and self.pat[j + 1] != "]"
                ):
                    lo = chars.pop()
                    for ch in range(ord(lo), ord(self.pat[j + 1]) + 1):
                        chars.add(chr(ch))
                    j += 1
                else:
                    chars.add(self.pat[j])
                j += 1
            if j >= len(self.pat):
                raise ValueError("unbalanced [")
            self.i = j + 1
            return ("cls", frozenset(chars), neg)
        if c == ".":
            self.i += 1
            return ("dot",)
        self.i += 1
        return ("lit", c)


def _emit(ast: Any, prog: list[_Inst]) -> None:
    tag = ast[0] if isinstance(ast, tuple) else None
    if isinstance(ast, list):
        for a in ast:
            _emit(a, prog)
        return
    if tag == "lit":
        prog.append((CHAR, ast[1], None))
    elif tag == "dot":
        prog.append((DOT, None, None))
    elif tag == "cls":
        prog.append((CLS, ast[1], ast[2]))
    elif tag == "cat":
        for a in ast[1]:
            _emit(a, prog)
    elif tag == "noncap":
        _emit(ast[1], prog)
    elif tag == "group":
        n = int(ast[2])
        prog.append((SAVE, 2 * n, None))
        _emit(ast[1], prog)
        prog.append((SAVE, 2 * n + 1, None))
    elif tag == "alt":
        parts = ast[1]
        jmps: list[int] = []
        for k, part in enumerate(parts):
            if k + 1 < len(parts):
                sp = len(prog)
                prog.append((SPLIT, sp + 1, None))
                _emit(part, prog)
                jmps.append(len(prog))
                prog.append((JMP, None, None))
                prog[sp] = (SPLIT, sp + 1, len(prog))
            else:
                _emit(part, prog)
        end = len(prog)
        for jp in jmps:
            prog[jp] = (JMP, end, None)
    elif tag in ("*", "+", "?"):
        sub = ast[1]
        if tag == "?":
            sp = len(prog)
            prog.append((SPLIT, sp + 1, None))
            _emit(sub, prog)
            prog[sp] = (SPLIT, sp + 1, len(prog))
        elif tag == "*":
            sp = len(prog)
            prog.append((SPLIT, sp + 1, None))
            _emit(sub, prog)
            prog.append((JMP, sp, None))
            prog[sp] = (SPLIT, sp + 1, len(prog))
        else:  # +
            start = len(prog)
            _emit(sub, prog)
            prog.append((SPLIT, start, len(prog) + 1))
    else:
        raise ValueError(f"bad ast {ast}")


def compile(pat: str) -> tuple[list[_Inst], int]:
    p = _Parser(pat)
    ast = p.parse()
    prog: list[_Inst] = []
    _emit(ast, prog)
    prog.append((MATCH, None, None))
    return prog, p.ngroup


def run(prog: list[_Inst], ngroups: int, text: str) -> list[tuple[int, int] | None] | None:
    """Return capture spans [ (s,e) per group incl 0 ] or None."""
    nslots = 2 * (ngroups + 1)

    def add(
        cl: list[tuple[int, list[int]]],
        pc: int,
        slots: list[int],
        i: int,
        seen: set[int],
    ) -> None:
        if pc in seen:
            return
        seen.add(pc)
        op, a, b = prog[pc]
        if op == JMP:
            add(cl, int(a), slots, i, seen)
        elif op == SPLIT:
            add(cl, int(a), slots, i, seen)
            add(cl, int(b), slots, i, seen)
        elif op == SAVE:
            s2 = slots.copy()
            s2[int(a)] = i
            add(cl, pc + 1, s2, i, seen)
        else:
            cl.append((pc, slots))

    start = [-1] * nslots
    start[0] = 0
    clist: list[tuple[int, list[int]]] = []
    add(clist, 0, start, 0, set())

    best: list[int] | None = None
    n = len(text)
    for i in range(n + 1):
        nlist: list[tuple[int, list[int]]] = []
        for pc, slots in clist:
            op, a, b = prog[pc]
            if op == MATCH:
                # fullmatch: only a match ending at n counts; first in
                # clist order is the backtracker's pick
                if i == n:
                    best = slots
                    break
                continue
            ok = i < n and (
                (op == CHAR and text[i] == a)
                or (op == DOT)
                or (op == CLS and ((text[i] in a) != bool(b)))
            )
            if ok:
                add(nlist, pc + 1, slots, i + 1, set())
        clist = nlist
        if not clist:
            break
    if best is None:
        return None
    out: list[tuple[int, int] | None] = []
    for g in range(ngroups + 1):
        gs, ge = int(best[2 * g]), int(best[2 * g + 1])
        out.append(None if gs < 0 else (gs, ge))
    return out


def bench_pike_vm(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    cases = [
        (r"a(b|c)*d", ["ad", "abd", "acbcd", "axd"]),
        (r"(ab)+c", ["abc", "ababc", "c", "abab"]),
        (r"x[0-9]+y", ["x42y", "xy", "x007y"]),
        (r"a.?b", ["ab", "axb", "axxb", "acb"]),
        (r"(a|b)(c|d)", ["ac", "bd", "aa", "dc"]),
    ]
    ok = True
    for pat, texts in cases:
        prog, ng = compile(pat)
        for t in texts:
            got = run(prog, ng, t)
            m = _re.fullmatch(pat, t)
            truth_list: list[tuple[int, int] | None] | None = None
            if m:
                truth_list = [m.span(0)]
                truth_list += [
                    m.span(i) if m.group(i) is not None else None for i in range(1, ng + 1)
                ]
            g_span = [None if s is None else (s[0], s[1]) for s in got] if got else None
            t_span = truth_list
            if g_span != t_span:
                ok = False
    score += 1.0 if ok else 0.0
    # random patterns vs re
    alpha = "ab"
    ok = True
    for _ in range(200):
        pat = "".join(
            str(rng.choice(list(alpha) + ["*", "+", "?", "|", "(", ")", "."]))
            for _ in range(int(rng.integers(3, 10)))
        )
        try:
            prog, ng = compile(pat)
            rgx = _re.compile(pat)
        except (ValueError, _re.error):
            continue
        t = "".join(str(rng.choice(list(alpha + "c"))) for _ in range(int(rng.integers(0, 8))))
        got = run(prog, ng, t)
        matched = rgx.fullmatch(t) is not None
        if (got is not None) != matched:
            ok = False
            break
    score += 1.0 if ok else 0.0
    # capture semantics
    prog, ng = compile(r"(a*)(b+)")
    got = run(prog, ng, "aabbb")
    score += 1.0 if got and got[1] == (0, 2) and got[2] == (2, 5) else 0.0
    prog, ng = compile(r"(a|ab)(c|bcd)")
    got = run(prog, ng, "abcd")
    score += 1.0 if got is not None and got[0] == (0, 4) else 0.0
    return {"synthetic_pike_vm": score / 4.0}
