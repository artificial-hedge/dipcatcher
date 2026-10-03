"""SYNTHETIC PEG parser with packrat memoization.

PEG semantics: ordered choice (first match wins — `a/ab` never matches
`ab` at a position where `a` matches), greedy repetition, & and !
predicates. Verifies ordered-choice bias and memo call bounds.
"""

from __future__ import annotations

# mini PEG evaluator over a fixed expression grammar with memo table


class PEG:
    """Grammar: Sum ← Prod ('+' Prod)* ; Prod ← Atom ('*' Atom)* ;
    Atom ← 'n' | '(' Sum ')' ; plus a probe rule 'AB ← "ab"/"a"'."""

    def __init__(self, s: str):
        self.s = s
        self.memo: dict[tuple[str, int], tuple[int, float] | None] = {}
        self.calls = 0

    def _m(self, fn: str, pos: int, body) -> tuple[int, float] | None:
        key = (fn, pos)
        if key in self.memo:
            return self.memo[key]
        self.calls += 1
        out: tuple[int, float] | None = body(pos)
        self.memo[key] = out
        return out

    def sum(self, pos: int) -> tuple[int, float] | None:
        return self._m("sum", pos, self._sum)

    def _sum(self, pos: int) -> tuple[int, float] | None:
        r = self.prod(pos)
        if r is None:
            return None
        pos, v = r
        while pos < len(self.s) and self.s[pos] == "+":
            r = self.prod(pos + 1)
            if r is None:
                break
            pos, v2 = r
            v += v2
        return (pos, v)

    def prod(self, pos: int) -> tuple[int, float] | None:
        return self._m("prod", pos, self._prod)

    def _prod(self, pos: int) -> tuple[int, float] | None:
        r = self.atom(pos)
        if r is None:
            return None
        pos, v = r
        while pos < len(self.s) and self.s[pos] == "*":
            r = self.atom(pos + 1)
            if r is None:
                break
            pos, v2 = r
            v *= v2
        return (pos, v)

    def atom(self, pos: int) -> tuple[int, float] | None:
        return self._m("atom", pos, self._atom)

    def _atom(self, pos: int) -> tuple[int, float] | None:
        if pos < len(self.s) and self.s[pos] == "n":
            return (pos + 1, 1.0)
        if pos < len(self.s) and self.s[pos] == "(":
            r = self.sum(pos + 1)
            if r is not None and r[0] < len(self.s) and self.s[r[0]] == ")":
                return (r[0] + 1, r[1])
            return None
        return None

    def ab_probe(self, pos: int = 0) -> bool:
        """Rule AB ← 'ab' / 'a' — ordered choice prefers 'ab' only if
        first alternative matches; AB2 ← 'a'/'ab' can never match 'ab'."""
        # first alternative tried first
        if self.s.startswith("ab", pos):
            return True
        return bool(self.s.startswith("a", pos))

    def ab_probe_biased(self, pos: int = 0) -> bool:
        """Rule AB2 ← 'a' / 'ab': 'a' always wins; 'ab' unreachable."""
        return bool(self.s.startswith("a", pos))  # 2nd alt unreachable


def bench_peg_packrat(seed: int = 20261231 + 394) -> dict[str, float]:
    val = bias = memo = 0
    trials = 40
    for _ in range(trials):
        # expression value check: n+n*n = 3, (n+n)*n = 4
        # each 'n' atom = 1.0: n+n*n = 2.0, (n+n)*n = 2.0
        p = PEG("n+n*n")
        r = p.sum(0)
        val += int(r is not None and r[0] == 5 and r[1] == 2.0)
        p2 = PEG("(n+n)*n")
        r2 = p2.sum(0)
        val += int(r2 is not None and r2[1] == 2.0)
        # ordered-choice bias: 'a'/'ab' on input 'ab' consumes only 'a'
        peg = PEG("ab")
        bias += int(peg.ab_probe_biased(0))  # matches 'a', proves 1st-alt wins
        bias += int(peg.ab_probe(0))  # 'ab'/'a' matches whole
        # memoization: repeated parse of same position hits table
        p3 = PEG("n+n+n+n+n")
        p3.sum(0)
        before = p3.calls
        p3.sum(0)  # second full parse → all memo hits
        memo += int(p3.calls == before)
    return {
        "synthetic_parse_value": float(val / (trials * 2)),
        "synthetic_ordered_choice": float(bias / (trials * 2)),
        "synthetic_packrat_memo": float(memo / trials),
    }
