"""Two-way DFA over input word with end-markers; crossing-sequence halt check (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 554


def run_2dfa(
    trans: dict[tuple[int, str], tuple[int, int]],
    start: int,
    accept: int,
    word: str,
    cap: int = 20_000,
) -> bool:
    """trans[(q,c)] = (q', move) with move in {-1,0,+1}; word wrapped in ⊢x⊣."""
    tape = "⊢" + word + "⊣"
    q, i = start, 1
    seen: set[tuple[int, int]] = set()
    for _ in range(cap):
        if q == accept:
            return True
        key = (q, i)
        if key in seen:
            return False
        seen.add(key)
        if (q, tape[i]) not in trans:
            return False
        q, mv = trans[(q, tape[i])]
        i += mv
        i = max(0, min(len(tape) - 1, i))
    return False


def bench_two_way_dfa(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    # 2DFA for strings containing "ab" (scan right looking for 'a', then peek)
    # states: 0 scan-right for 'a'; 1 saw a, check next is b; 2 accept; 3 reject
    trans: dict[tuple[int, str], tuple[int, int]] = {}
    for c in "ab⊣⊢":
        trans[(0, c)] = (0, 1)
    trans[(0, "a")] = (1, 1)
    trans[(0, "⊣")] = (3, 0)
    trans[(1, "b")] = (2, 0)
    trans[(1, "a")] = (1, 1)
    trans[(1, "⊣")] = (3, 0)
    n = 60
    hits = 0
    for _ in range(n):
        w = "".join(rng.choice(list("ab")) for _ in range(rng.randint(1, 8)))
        hits += int(run_2dfa(trans, 0, 2, w) == ("ab" in w))
    # palindrome checker via crossing: mark ends
    trans2: dict[tuple[int, str], tuple[int, int]] = {}
    # q0: at left, remember symbol in q_a/q_b and move right to ⊣; q_back: move left to ⊢ then forward past leftmark...
    # simpler language: first==last
    for c in "ab":
        trans2[(0, c)] = (10 if c == "a" else 11, 1)
        trans2[(10, c)] = (10, 1)
        trans2[(11, c)] = (11, 1)
    trans2[(10, "⊣")] = (20, -1)
    trans2[(11, "⊣")] = (21, -1)
    trans2[(20, "a")] = (2, 0)
    trans2[(20, "b")] = (3, 0)
    trans2[(21, "a")] = (3, 0)
    trans2[(21, "b")] = (2, 0)
    hits2 = 0
    for _ in range(n):
        w = "".join(rng.choice(list("ab")) for _ in range(rng.randint(2, 8)))
        hits2 += int(run_2dfa(trans2, 0, 2, w) == (w[0] == w[-1]))
    return {
        "synthetic_contains_ab": float(hits / n),
        "synthetic_first_eq_last": float(hits2 / n),
    }
