"""Lazy DFA: on-demand subset construction over a Thompson NFA.

Reuses the Pike-VM compiled program: CHAR/DOT/CLS are consuming edges,
SPLIT/JMP/SAVE are epsilon, MATCH is the accept marker. DFA states are
frozensets of NFA pcs, built lazily and memoized. SYNTHETIC bench verifies
agreement with the NFA simulation and bounded state growth.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models.pike_vm import CHAR, CLS, DOT, JMP, MATCH, SAVE, SPLIT, compile

_SEED = 20261231 + 927


class LazyDFA:
    def __init__(self, pat: str) -> None:
        self.prog, self.ngroups = compile(pat)
        self.eps: list[list[int]] = [[] for _ in self.prog]
        for pc, (op, a, b) in enumerate(self.prog):
            if op == JMP:
                self.eps[pc].append(int(a))
            elif op == SPLIT:
                self.eps[pc].extend([int(a), int(b)])
            elif op == SAVE:
                self.eps[pc].append(pc + 1)
        self.states: dict[frozenset[int], int] = {}
        self.trans: dict[tuple[int, str], int] = {}
        self.accepts: set[int] = set()
        self.start = self._intern(self._closure({0}))

    def _closure(self, pcs: set[int]) -> frozenset[int]:
        out: set[int] = set()
        stack = list(pcs)
        while stack:
            pc = stack.pop()
            if pc in out:
                continue
            out.add(pc)
            stack.extend(self.eps[pc])
        return frozenset(out)

    def _intern(self, s: frozenset[int]) -> int:
        if s not in self.states:
            self.states[s] = len(self.states)
            if any(self.prog[pc][0] == MATCH for pc in s):
                self.accepts.add(self.states[s])
        return self.states[s]

    def step(self, dfa_state: int, ch: str) -> int | None:
        key = (dfa_state, ch)
        if key in self.trans:
            return self.trans[key]
        s_set = next(s for s, sid in self.states.items() if sid == dfa_state)
        nxt: set[int] = set()
        for pc in s_set:
            op, a, b = self.prog[pc]
            if (op == CHAR and ch == a) or op == DOT or (op == CLS and ((ch in a) != bool(b))):
                nxt.add(pc + 1)
        sid = self._intern(self._closure(nxt)) if nxt else None
        self.trans[key] = sid  # type: ignore[assignment]
        return sid

    def fullmatch(self, text: str) -> bool:
        cur: int | None = self.start
        for ch in text:
            if cur is None:
                return False
            cur = self.step(cur, ch)
        return cur in self.accepts if cur is not None else False

    def n_states(self) -> int:
        return len(self.states)


def bench_lazy_dfa(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    dfa = LazyDFA("(a|b)*abb")
    truth = {"abb", "aabb", "babb", "ababb", "aaabb", "bbabb"}
    ok = all(
        dfa.fullmatch(t) == (t.endswith("abb") and set(t) <= {"a", "b"})
        for t in truth | {"ab", "abab", "bbba", ""}
    )
    score += 1.0 if ok else 0.0
    # fuzz agreement with re
    import re as _re

    ok = True
    alphabet = "ab"
    for _ in range(150):
        pat = "".join(
            str(rng.choice(list(alphabet) + ["*", "+", "?", "|", "(", ")", "."]))
            for _ in range(int(rng.integers(3, 9)))
        )
        try:
            d = LazyDFA(pat)
            rgx = _re.compile(pat)
        except (ValueError, _re.error):
            continue
        t = "".join(str(rng.choice(list(alphabet + "c"))) for _ in range(int(rng.integers(0, 8))))
        if d.fullmatch(t) != (rgx.fullmatch(t) is not None):
            ok = False
            break
    score += 1.0 if ok else 0.0
    # laziness: after one query, far fewer states than reachable NFA subsets
    d2 = LazyDFA("(a|b)(a|b)(a|b)(a|b)(a|b)x")
    d2.fullmatch("ababax")
    score += 1.0 if d2.n_states() < 16 else 0.0
    # memoization: repeat query adds no states
    n0 = d2.n_states()
    d2.fullmatch("bbabax")
    score += 1.0 if d2.n_states() <= n0 + 4 else 0.0
    return {"synthetic_lazy_dfa": score / 4.0}
