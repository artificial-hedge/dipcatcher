"""SYNTHETIC Earley recognizer for arbitrary CFGs.

Classic Earley chart: predict/scan/complete over (rule, dot, origin)
items. Tested on balanced-parens and expression grammars vs brute-force
membership oracles.
"""

from __future__ import annotations

import random

Rule = tuple[str, tuple[str, ...]]


def earley_accepts(grammar: list[Rule], start: str, tokens: list[str]) -> bool:
    rules_by_lhs: dict[str, list[Rule]] = {}
    for r in grammar:
        rules_by_lhs.setdefault(r[0], []).append(r)
    n = len(tokens)
    # chart[i] = set of (rule_index, dot, origin)
    chart: list[set[tuple[int, int, int]]] = [set() for _ in range(n + 1)]
    # seed: all rules expanding the start symbol
    for r in grammar:
        if r[0] == start:
            chart[0].add((grammar.index(r), 0, 0))
    for i in range(n + 1):
        agenda = list(chart[i])
        seen = set(chart[i])
        while agenda:
            item = agenda.pop()
            ri, dot, org = item
            lhs, rhs = grammar[ri]
            if dot < len(rhs):
                sym = rhs[dot]
                if sym in rules_by_lhs:  # nonterminal → predict
                    for r in rules_by_lhs[sym]:
                        ni = (grammar.index(r), 0, i)
                        if ni not in seen:
                            seen.add(ni)
                            chart[i].add(ni)
                            agenda.append(ni)
                else:  # terminal → scan
                    if i < n and tokens[i] == sym:
                        ni = (ri, dot + 1, org)
                        if ni not in chart[i + 1]:
                            chart[i + 1].add(ni)
            else:  # complete
                for rj, dj, oj in list(chart[org]):
                    lhs2, rhs2 = grammar[rj]
                    if dj < len(rhs2) and rhs2[dj] == lhs:
                        ni = (rj, dj + 1, oj)
                        if ni not in seen:
                            seen.add(ni)
                            chart[i].add(ni)
                            agenda.append(ni)
    return any(
        grammar[ri][0] == start and dot == len(grammar[ri][1]) and org == 0
        for (ri, dot, org) in chart[n]
    )


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        d += 1 if ch == "(" else -1
        if d < 0:
            return False
    return d == 0


def bench_earley_parser(seed: int = 20261231 + 392) -> dict[str, float]:
    rng = random.Random(seed)
    gram_bp: list[Rule] = [
        ("S", ("S", "S")),
        ("S", ("(", "S", ")")),
        ("S", ()),
    ]
    gram_num: list[Rule] = [
        ("E", ("E", "+", "E")),
        ("E", ("n",)),
    ]
    bp_match = num_match = rej = 0
    trials = 40
    for _ in range(trials):
        s = "".join(rng.choice("()") for _ in range(rng.randrange(0, 12)))
        got = earley_accepts(gram_bp, "S", list(s))
        bp_match += int(got == _balanced(s))
        # expression grammar: "n + n + ... + n" valid; "n + +" invalid
        toks = []
        for j in range(rng.randrange(1, 5)):
            if j:
                toks.append("+")
            toks.append("n")
        num_match += int(earley_accepts(gram_num, "E", toks))
        rej += int(not earley_accepts(gram_num, "E", ["n", "+", "+"]))
    return {
        "synthetic_parens_correct": float(bp_match / trials),
        "synthetic_accepts_valid": float(num_match / trials),
        "synthetic_rejects_invalid": float(rej / trials),
    }
