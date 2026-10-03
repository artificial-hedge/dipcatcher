"""SYNTHETIC recursive-descent expression parser.

Grammar: E := T (('+'|'-') T)* ; T := F (('*'|'/') F)* ;
F := num | '(' E ')' | '-' F. Verified against Python's ast module
evaluation oracle (no eval()).
"""

from __future__ import annotations

import ast
import random


class RDParser:
    def __init__(self, s: str):
        self.toks = (
            s.replace("(", " ( ")
            .replace(")", " ) ")
            .replace("+", " + ")
            .replace("-", " - ")
            .replace("*", " * ")
            .replace("/", " / ")
            .split()
        )
        self.i = 0

    def _peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _eat(self, t: str | None = None) -> str:
        tok = self.toks[self.i]
        if t is not None and tok != t:
            raise ValueError(f"expected {t} got {tok}")
        self.i += 1
        return tok

    def parse(self) -> float:
        v = self.expr()
        if self.i != len(self.toks):
            raise ValueError("trailing tokens")
        return v

    def expr(self) -> float:
        v = self.term()
        while self._peek() in ("+", "-"):
            op = self._eat()
            rhs = self.term()
            v = v + rhs if op == "+" else v - rhs
        return v

    def term(self) -> float:
        v = self.factor()
        while self._peek() in ("*", "/"):
            op = self._eat()
            rhs = self.factor()
            v = v * rhs if op == "*" else v / rhs
        return v

    def factor(self) -> float:
        tok = self._peek()
        if tok == "-":
            self._eat("-")
            return -self.factor()
        if tok == "(":
            self._eat("(")
            v = self.expr()
            self._eat(")")
            return v
        return float(self._eat())


def _oracle(s: str) -> float:
    def go(n: ast.expr) -> float:
        if isinstance(n, ast.Constant):
            return float(n.value)  # type: ignore[arg-type]
        if isinstance(n, ast.BinOp):
            a, b = go(n.left), go(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -go(n.operand)
        raise ValueError("bad ast")

    return go(ast.parse(s, mode="eval").body)


def _gen_expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2 or rng.random() < 0.3:
        return str(rng.randrange(1, 20))
    a, b = _gen_expr(rng, depth + 1), _gen_expr(rng, depth + 1)
    op = rng.choice("+-*")
    return f"({a} {op} {b})"


def bench_recursive_descent(seed: int = 20261231 + 390) -> dict[str, float]:
    rng = random.Random(seed)
    match = rej = assoc = 0
    trials = 60
    for _ in range(trials):
        e = _gen_expr(rng)
        try:
            got = RDParser(e).parse()
            exp = _oracle(e)
            match += int(abs(got - exp) < 1e-9)
        except ValueError:
            pass
        # reject malformed
        try:
            RDParser("1 + * 2").parse()
            rej += 0
        except (ValueError, IndexError):
            rej += 1
        # precedence: 2+3*4 = 14 not 20
        assoc += int(RDParser("2 + 3 * 4").parse() == 14.0)
    return {
        "synthetic_matches_oracle": float(match / trials),
        "synthetic_rejects_bad": float(rej / trials),
        "synthetic_precedence": float(assoc / trials),
    }
