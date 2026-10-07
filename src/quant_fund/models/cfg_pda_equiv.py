"""CFG ↔ PDA equivalence: expand-top-of-stack PDA checked against CYK oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 553


def cyk(rules: dict[str, list[tuple[str, ...]]], start: str, word: str) -> bool:
    """CYK on CNF-like rules (A -> BC or A -> a)."""
    n = len(word)
    if n == 0:
        return ("eps",) in rules.get(start, []) or () in rules.get(start, [])
    T: list[list[set[str]]] = [[set() for _ in range(n)] for _ in range(n)]
    for i, ch in enumerate(word):
        for a, rhss in rules.items():
            for rhs in rhss:
                if rhs == (ch,):
                    T[i][i].add(a)
    for span in range(2, n + 1):
        for i in range(n - span + 1):
            j = i + span - 1
            for k in range(i, j):
                for a, rhss in rules.items():
                    for rhs in rhss:
                        if len(rhs) == 2 and rhs[0] in T[i][k] and rhs[1] in T[k + 1][j]:
                            T[i][j].add(a)
    return start in T[0][n - 1]


def pda_accept(
    rules: dict[str, list[tuple[str, ...]]], start: str, word: str, cap: int = 200_000
) -> bool:
    """BFS over (stack, pos): expand nonterminal at top, consume terminal."""
    terminals = {c for rhss in rules.values() for rhs in rhss for c in rhs if c.islower()}
    seen: set[tuple[tuple[str, ...], int]] = set()
    todo: list[tuple[tuple[str, ...], int]] = [((start,), 0)]
    steps = 0
    while todo and steps < cap:
        st, i = todo.pop()
        steps += 1
        if (st, i) in seen:
            continue
        seen.add((st, i))
        if not st:
            if i == len(word):
                return True
            continue
        top, rest = st[0], st[1:]
        if top in terminals:
            if i < len(word) and word[i] == top:
                todo.append((rest, i + 1))
            continue
        for rhs in rules.get(top, []):
            new: tuple[str, ...] = () if rhs == ("eps",) else tuple(rhs)
            nst = new + rest
            # stack can't exceed unmatched terminals + variables ≤ len(word)+2
            if len(nst) <= len(word) + 2:
                todo.append((nst, i))
    return False


def bench_cfg_pda_equiv(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # CNF grammar for a^n b^n (n≥1): S -> AB | AT; T -> SB; A -> a; B -> b
    rules: dict[str, list[tuple[str, ...]]] = {
        "S": [("A", "B"), ("A", "T")],
        "T": [("S", "B")],
        "A": [("a",)],
        "B": [("b",)],
    }
    n = 60
    eq = 0
    for _ in range(n):
        w = "".join(rng.choice(list("ab")) for _ in range(rng.randint(1, 7)))
        c = cyk(rules, "S", w)
        p = pda_accept(rules, "S", w)
        eq += int(c == p)
    return {"synthetic_pda_cyk_agree": float(eq / n)}
