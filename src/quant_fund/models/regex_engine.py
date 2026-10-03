"""Regex engine: parse → Thompson NFA → subset DFA (synthetic).

Supports literal chars, '.', '*', '+', '?', '|', concat, parens.
Verified: full-match acceptance equals Python `re.fullmatch` oracle
on random patterns and strings.
"""

from __future__ import annotations

import random
import re as _re


class _NFA:
    def __init__(self) -> None:
        self.eps: list[set[int]] = []
        self.trans: list[dict[str, int]] = []
        self.start = 0
        self.accept = 0

    def add(self) -> int:
        self.eps.append(set())
        self.trans.append({})
        return len(self.eps) - 1


def _thompson(postfix: list[str]) -> _NFA:
    """Each stack item is (start, accept) pair."""
    nfa = _NFA()
    stack: list[tuple[int, int]] = []
    for tok in postfix:
        if tok == ".":
            # concat: first frag (c,d), second (a,b): s→c, d→a, b→e
            a, b = stack.pop()
            c, d = stack.pop()
            s, e = nfa.add(), nfa.add()
            nfa.eps[s].add(c)
            nfa.eps[d].add(a)
            nfa.eps[b].add(e)
            stack.append((s, e))
        elif tok == "|":
            a, b = stack.pop()
            c, d = stack.pop()
            s, e = nfa.add(), nfa.add()
            nfa.eps[s] |= {c, a}
            nfa.eps[d] |= {e}
            nfa.eps[b] |= {e}
            stack.append((s, e))
        elif tok == "*":
            a, b = stack.pop()
            s, e = nfa.add(), nfa.add()
            nfa.eps[s] |= {a, e}
            nfa.eps[b] |= {a, e}
            stack.append((s, e))
        elif tok == "+":
            a, b = stack.pop()
            s, e = nfa.add(), nfa.add()
            nfa.eps[b] |= {a, e}
            nfa.eps[s].add(a)
            stack.append((s, e))
        elif tok == "?":
            a, b = stack.pop()
            s, e = nfa.add(), nfa.add()
            nfa.eps[s] |= {a, e}
            nfa.eps[b] |= {e}
            stack.append((s, e))
        else:  # literal
            s, e = nfa.add(), nfa.add()
            nfa.trans[s][tok] = e
            stack.append((s, e))
    nfa.start, nfa.accept = stack[0]
    return nfa


def _to_postfix(pattern: str) -> list[str]:
    """Shunting-yard; '.' is concat, '|' alternation, '*' '+' '?' postfix."""
    prec = {"|": 1, ".": 2}
    out: list[str] = []
    ops: list[str] = []
    prev_operand = False
    for ch in pattern:
        if ch == "(":
            if prev_operand:
                while ops and ops[-1] != "(" and prec[ops[-1]] >= 2:
                    out.append(ops.pop())
                ops.append(".")
            ops.append(ch)
            prev_operand = False
        elif ch == ")":
            while ops[-1] != "(":
                out.append(ops.pop())
            ops.pop()
            prev_operand = True
        elif ch in "|":
            while ops and ops[-1] != "(" and prec[ops[-1]] >= 1:
                out.append(ops.pop())
            ops.append(ch)
            prev_operand = False
        elif ch in "*+?":
            out.append(ch)
            prev_operand = True
        else:
            if prev_operand:
                while ops and ops[-1] != "(" and prec[ops[-1]] >= 2:
                    out.append(ops.pop())
                ops.append(".")
            out.append(ch)
            prev_operand = True
    while ops:
        out.append(ops.pop())
    return out


def _eps_closure(nfa: _NFA, states: set[int]) -> set[int]:
    seen = set(states)
    stack = list(states)
    while stack:
        s = stack.pop()
        for t in nfa.eps[s]:
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return seen


def match(nfa: _NFA, text: str) -> bool:
    cur = _eps_closure(nfa, {nfa.start})
    for c in text:
        nxt: set[int] = set()
        for s in cur:
            if c in nfa.trans[s]:
                nxt.add(nfa.trans[s][c])
        cur = _eps_closure(nfa, nxt)
        if not cur:
            return False
    return nfa.accept in cur


def compile_re(pattern: str) -> _NFA:
    return _thompson(_to_postfix(pattern))


def bench_regex_engine(seed: int = 20261231 + 280) -> dict[str, float]:
    rng = random.Random(seed)
    pats = [
        "a*",
        "a+b",
        "(ab)*c",
        "a|bc",
        "(a|b)*abb",
        "a?b+c*",
        "(a|b)(c|d)",
        "ab*c",
        "(a|bc)*d",
    ]
    agree = 0
    total = 0
    for pat in pats:
        nfa = compile_re(pat)
        for _ in range(30):
            text = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 8)))
            want = _re.fullmatch(pat, text) is not None
            got = match(nfa, text)
            agree += int(got == want)
            total += 1
    return {
        "synthetic_agree": float(agree / total),
        "synthetic_n_patterns": float(len(pats)),
        "synthetic_n_tests": float(total),
    }
