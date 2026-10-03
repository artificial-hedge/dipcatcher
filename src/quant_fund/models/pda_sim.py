"""SYNTHETIC nondeterministic pushdown automaton simulator.

Configuration-graph BFS over (state, input position, stack): accepts a^n b^n
via counting PDA and even palindromes via nondeterministic middle guess.
Verified against brute-force language oracles.
"""

from __future__ import annotations

import random
from collections import deque


def run_pda(
    transitions: dict[tuple[str, str | None, str | None], list[tuple[str, str]]],
    s: str,
    start: str = "q0",
    accepts: frozenset[str] | None = None,
    cap: int = 20000,
) -> bool:
    """NPDA accept-or-reject via BFS over configurations.

    transition key: (state, input-char-or-None, stack-top-or-None)
    value list: (next_state, stack_push_string) — pop replaces top, "" = pop.
    """
    accepts = accepts or frozenset({"qf"})
    seen: set[tuple[str, int, tuple[str, ...]]] = set()
    q: deque[tuple[str, int, tuple[str, ...]]] = deque([(start, 0, ("Z",))])
    steps = 0
    while q and steps < cap:
        st, pos, stack = q.popleft()
        steps += 1
        if pos == len(s) and st in accepts:
            return True
        top = stack[-1] if stack else None
        keys = [(st, s[pos] if pos < len(s) else None, top), (st, None, top)]
        for key in keys:
            for nst, push in transitions.get(key, []):
                ns = stack[:-1] if top else stack
                ns = ns + tuple(push) if push else ns
                cfg = (nst, pos + (0 if key[1] is None else 1), ns)
                if cfg not in seen and len(cfg[2]) <= len(s) + 2:
                    seen.add(cfg)
                    q.append(cfg)
    return False


def _anbn_pda() -> dict[tuple[str, str | None, str | None], list[tuple[str, str]]]:
    tr: dict[tuple[str, str | None, str | None], list[tuple[str, str]]] = {}
    tr[("q0", "a", "Z")] = [("q0", "ZA")]
    tr[("q0", "a", "A")] = [("q0", "AA")]  # pop A, push AA (net +1)
    tr[("q0", "b", "A")] = [("q1", "")]
    tr[("q1", "b", "A")] = [("q1", "")]
    tr[("q1", None, "Z")] = [("qf", "")]
    tr[("q0", None, "Z")] = [("qf", "")]  # epsilon word
    return tr


def _pal_pda() -> dict[tuple[str, str | None, str | None], list[tuple[str, str]]]:
    # even palindrome ww^R over {a,b}: guess middle (eps transition)
    tr: dict[tuple[str, str | None, str | None], list[tuple[str, str]]] = {}
    for c in "ab":
        for top in ("Z", "A", "B"):
            tr.setdefault(("q0", c, top), []).append(("q0", top + c.upper()))
        tr.setdefault(("q0", None, "Z"), []).append(("q1", "Z"))
        tr.setdefault(("q0", None, "A"), []).append(("q1", "A"))
        tr.setdefault(("q0", None, "B"), []).append(("q1", "B"))
        tr[("q1", c, c.upper())] = [("q1", "")]
    tr[("q1", None, "Z")] = [("qf", "")]
    return tr


def bench_pda_sim(seed: int = 20261231 + 501) -> dict[str, float]:
    rng = random.Random(seed)
    ok_ab = 0
    n_ab = 50
    for _ in range(n_ab):
        n = rng.randrange(0, 6)
        s = (
            "a" * n + "b" * n
            if rng.random() < 0.5
            else "".join(rng.choice("ab") for _ in range(rng.randrange(0, 9)))
        )
        truth = s == "a" * s.count("a") + "b" * s.count("b") and s.count("a") == s.count("b")
        ok_ab += int(run_pda(_anbn_pda(), s) == truth)
    ok_pal = 0
    n_pal = 40
    for _ in range(n_pal):
        w = "".join(rng.choice("ab") for _ in range(rng.randrange(0, 5)))
        s = (
            w + w[::-1]
            if rng.random() < 0.6
            else "".join(rng.choice("ab") for _ in range(rng.randrange(0, 9)))
        )
        ok_pal += int(run_pda(_pal_pda(), s, cap=40000) == (s == s[::-1] and len(s) % 2 == 0))
    return {
        "synthetic_anbn_exact": ok_ab / n_ab,
        "synthetic_palindrome_exact": ok_pal / n_pal,
    }
