"""SYNTHETIC LL(1) predictive-table parser.

Grammar (LL(1) form):
  E  → T E'   E' → + T E' | ε
  T  → F T'   T' → * F T' | ε
  F  → ( E ) | n
Predict table built from FIRST/FOLLOW; driver evaluates while parsing.
"""

from __future__ import annotations

import random

RULES: dict[str, list[tuple[str, ...]]] = {
    "E": [("T", "E'")],
    "E'": [("+", "T", "E'"), ()],
    "T": [("F", "T'")],
    "T'": [("*", "F", "T'"), ()],
    "F": [("(", "E", ")"), ("n",)],
}
NONTERM = set(RULES)


def _first_of(syms: tuple[str, ...], first: dict[str, set[str]]) -> set[str]:
    if not syms:
        return {"ε"}
    out: set[str] = set()
    for s in syms:
        fs = first.get(s, {s})
        out |= fs - {"ε"}
        if "ε" not in fs:
            break
    else:
        out.add("ε")
    return out


def build_table() -> dict[tuple[str, str], tuple[str, ...]]:
    first: dict[str, set[str]] = {nt: set() for nt in NONTERM}
    changed = True
    while changed:
        changed = False
        for nt, prods in RULES.items():
            for p in prods:
                add = _first_of(p, first)
                if not add <= first[nt]:
                    first[nt] |= add
                    changed = True
    follow: dict[str, set[str]] = {nt: set() for nt in NONTERM}
    follow["E"].add("$")
    changed = True
    while changed:
        changed = False
        for nt, prods in RULES.items():
            for p in prods:
                for i, s in enumerate(p):
                    if s not in NONTERM:
                        continue
                    beta = p[i + 1 :]
                    fb = _first_of(beta, first)
                    add = fb - {"ε"}
                    if "ε" in fb:
                        add |= follow[nt]
                    if not add <= follow[s]:
                        follow[s] |= add
                        changed = True
    table: dict[tuple[str, str], tuple[str, ...]] = {}
    for nt, prods in RULES.items():
        for p in prods:
            fp = _first_of(p, first)
            for t in fp - {"ε"}:
                table[(nt, t)] = p
            if "ε" in fp:
                for t in follow[nt]:
                    table[(nt, t)] = p
    return table


def ll1_eval(tokens: list[str]) -> float:
    table = build_table()
    toks = tokens + ["$"]
    stack: list[str] = ["E"]
    i = 0
    while stack:
        top = stack.pop()
        cur = toks[i]
        if top in NONTERM:
            prod = table.get((top, cur))
            if prod is None:
                raise ValueError(f"no rule for {top},{cur}")
            for s in reversed(prod):
                stack.append(s)
        elif top == cur:
            i += 1
        else:
            raise ValueError("mismatch")
    return _eval_str(tokens)


def _eval_str(toks: list[str]) -> float:
    pos = 0

    def expr() -> float:
        nonlocal pos
        v = term()
        while pos < len(toks) and toks[pos] == "+":
            pos += 1
            v += term()
        return v

    def term() -> float:
        nonlocal pos
        v = factor()
        while pos < len(toks) and toks[pos] == "*":
            pos += 1
            v *= factor()
        return v

    def factor() -> float:
        nonlocal pos
        if toks[pos] == "(":
            pos += 1
            v = expr()
            pos += 1
            return v
        pos += 1
        return 1.0

    return expr()


def bench_ll1_table(seed: int = 20261231 + 395) -> dict[str, float]:
    rng = random.Random(seed)
    parse_ok = rej = complete = 0
    trials = 40
    for _ in range(trials):
        # valid token streams always parse
        n = rng.randrange(1, 5)
        toks: list[str] = []
        for j in range(n):
            if j:
                toks.append(rng.choice("+*"))
            toks.append("n")
        try:
            ll1_eval(toks)
            parse_ok += 1
        except ValueError:
            pass
        # invalid rejected
        try:
            ll1_eval(["n", "+", "*", "n"])
            rej += 0
        except ValueError:
            rej += 1
        # table complete: every (nt, terminal) covered where needed
        table = build_table()
        complete += int(
            ("E'", "+") in table
            and ("E'", ")") in table
            and ("T'", "*") in table
            and ("F", "n") in table
            and ("F", "(") in table
        )
    return {
        "synthetic_parses_valid": float(parse_ok / trials),
        "synthetic_rejects_bad": float(rej / trials),
        "synthetic_table_complete": float(complete / trials),
    }
