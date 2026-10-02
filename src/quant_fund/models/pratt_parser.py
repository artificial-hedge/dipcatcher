"""SYNTHETIC Pratt (precedence-climbing) parser.

Handles left-assoc +,-,*,/ and right-assoc '^' (power). Verifies
precedence, right-associativity (2^3^2 = 512), left-assoc of '-'.
"""

from __future__ import annotations

import random

# binding powers: (left_bp, right_bp); right-assoc '^' has rbp < lbp
_BP = {"+": (10, 11), "-": (10, 11), "*": (20, 21), "/": (20, 21), "^": (31, 30)}


class Pratt:
    def __init__(self, s: str):
        for ch in "+-*/^()":
            s = s.replace(ch, f" {ch} ")
        self.toks = s.split()
        self.i = 0

    def _peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def parse(self) -> float:
        v = self._expr(0)
        if self.i != len(self.toks):
            raise ValueError("trailing")
        return v

    def _expr(self, min_bp: int) -> float:
        tok = self.toks[self.i]
        self.i += 1
        if tok == "(":
            lhs = self._expr(0)
            if self._peek() != ")":
                raise ValueError("missing )")
            self.i += 1
        elif tok == "-":
            lhs = -self._expr(25)
        else:
            lhs = float(tok)
        while self._peek() in _BP:
            op = self._peek() or ""
            lbp, rbp = _BP[op]
            if lbp < min_bp:
                break
            self.i += 1
            rhs = self._expr(rbp)
            lhs = {
                "+": lhs + rhs,
                "-": lhs - rhs,
                "*": lhs * rhs,
                "/": lhs / rhs,
                "^": lhs**rhs,
            }[op]
        return lhs


def bench_pratt_parser(seed: int = 20261231 + 391) -> dict[str, float]:
    rng = random.Random(seed)
    prec = right = assoc = 0
    trials = 60
    for _ in range(trials):
        prec += int(Pratt("2 + 3 * 4").parse() == 14.0)
        right += int(Pratt("2 ^ 3 ^ 2").parse() == 512.0)
        a, b, c = rng.randrange(1, 9), rng.randrange(1, 9), rng.randrange(1, 9)
        assoc += int(Pratt(f"{a} - {b} - {c}").parse() == (a - b) - c)
    return {
        "synthetic_precedence": float(prec / trials),
        "synthetic_right_assoc_power": float(right / trials),
        "synthetic_left_assoc_sub": float(assoc / trials),
    }
