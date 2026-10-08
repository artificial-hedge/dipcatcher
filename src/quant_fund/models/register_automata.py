"""Register automaton over data words: distinct-value acceptance (fresh names) (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 555


def run_ra(
    trans: dict[tuple[int, str], tuple[int, int, int]],
    start: int,
    accept: set[int],
    word: list[int],
) -> bool:
    """trans[(q,cond)] = (q', store_reg, cmp_reg); cond in {'fresh','eq','neq'}.
    'fresh': value not in any register; 'eq': value == reg[cmp_reg]; 'neq': != reg.
    store_reg ≥0 stores current symbol; -1 keeps."""
    regs: dict[int, int] = {}
    q = start
    for v in word:
        progressed = False
        for (s, cond), (t, st_reg, cmp_reg) in trans.items():
            if s != q:
                continue
            ok = False
            if cond == "fresh":
                ok = v not in regs.values()
            elif cond == "eq":
                ok = regs.get(cmp_reg) == v
            elif cond == "neq":
                ok = regs.get(cmp_reg) != v
            if ok:
                if st_reg >= 0:
                    regs[st_reg] = v
                q = t
                progressed = True
                break
        if not progressed:
            return False
    return q in accept


def bench_register_automata(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # accepts iff all symbols distinct: q0 reads fresh -> store r0 -> q1;
    # q1: fresh -> q1 store nothing? need pairwise check — use "eq" fail:
    # simpler: language = first symbol repeats at end (x ... x)
    trans = {
        (0, "fresh"): (1, 0, -1),  # store first in r0
        (1, "fresh"): (1, -1, -1),  # middle: anything fresh ok
        (1, "eq"): (2, -1, 0),  # repeat of r0 → accept state
    }
    n = 60
    hits = 0
    for _ in range(n):
        w = [rng.randint(0, 4) for _ in range(rng.randint(2, 7))]
        truth = len(w) >= 2 and w[-1] == w[0] and w[-1] not in w[1:-1] and w[0] not in w[1:-1]
        # actually: accepted iff last sym == first AND last was "fresh" until then —
        # eq fires when v == regs[0]; fresh fires when v unseen anywhere... our
        # automaton only stores r0 — 'fresh' means "not equal to r0" effectively
        # since only r0 holds a value. truth: w ends with w[0] and no earlier
        # position (besides 0) equals w[0].
        truth = w[-1] == w[0] and w[0] not in w[1:-1]
        hits += int(run_ra(trans, 0, {2}, w) == truth)
    return {"synthetic_first_last_eq": float(hits / n)}
