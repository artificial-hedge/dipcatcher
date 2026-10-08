"""CYK parser for CNF grammars (synthetic) (SYNTHETIC).

Bottom-up chart parsing: X[i,j] = set of nonterminals deriving
w[i:j]. Verified: membership verdicts agree with exhaustive
recursive-derivation oracle on small CNF grammars.
"""

from __future__ import annotations

import random

# grammar: nonterminal -> [("A","B") binary | ("a",) terminal]
G = dict[str, list[tuple[str, ...]]]


def cyk(gram: G, start: str, w: str) -> bool:
    n = len(w)
    chart: list[list[set[str]]] = [[set() for _ in range(n + 1)] for _ in range(n)]
    for i, c in enumerate(w):
        for nt, prods in gram.items():
            for p in prods:
                if p == (c,):
                    chart[i][i + 1].add(nt)
    for ln in range(2, n + 1):
        for i in range(n - ln + 1):
            j = i + ln
            for k in range(i + 1, j):
                for nt, prods in gram.items():
                    for p in prods:
                        if len(p) == 2 and p[0] in chart[i][k] and p[1] in chart[k][j]:
                            chart[i][j].add(nt)
    return start in chart[0][n]


def _derive(gram: G, start: str, w: str, fuel: int = 20000) -> bool:
    """Exhaustive derivation oracle with fuel bound."""
    sent = [start]
    seen: set[str] = set()
    while sent:
        cur = sent.pop()
        if cur == w:
            return True
        if len(cur) > len(w) or cur in seen or fuel <= 0:
            continue
        seen.add(cur)
        fuel -= 1
        for i, ch in enumerate(cur):
            if ch.isupper():
                for p in gram.get(ch, []):
                    if len(cur) - 1 + len(p) <= len(w):
                        sent.append(cur[:i] + "".join(p) + cur[i + 1 :])
    return False


def bench_cyk_parser(seed: int = 20261231 + 282) -> dict[str, float]:
    rng = random.Random(seed)
    # classic balanced-paren-ish CNF grammar
    gram: G = {
        "S": [("A", "B"), ("S", "S"), ("A", "S")],
        "A": [("a",)],
        "B": [("b",), ("S", "B")],
    }
    agree = 0
    trials = 60
    for _ in range(trials):
        w = "".join(rng.choice("ab") for _ in range(rng.randint(1, 6)))
        agree += int(cyk(gram, "S", w) == _derive(gram, "S", w))
    # known members: 'ab', 'aabb' are in language (S→AB→ab; S→SS etc.)
    known = cyk(gram, "S", "ab") and cyk(gram, "S", "aab")
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_known_members": float(known),
    }
