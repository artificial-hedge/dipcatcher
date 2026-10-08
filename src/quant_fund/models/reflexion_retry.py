"""Reflexion — self-critique + retry (Shinn et al. 2023) (SYNTHETIC).

A failed first attempt produces a critique signal (which step went
wrong); the retry steers around the failing branch — raising the
solve rate over single-shot greedy.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._agent_synth import N_ACTIONS, transition

TARGET = 42


def _greedy(s0: int, max_steps: int, banned: set) -> tuple[int, list[int]]:
    s = s0
    path = []
    for _t in range(max_steps):
        if s == TARGET:
            break
        cand = [(a, transition(s, a)) for a in range(N_ACTIONS) if (s, a) not in banned]
        if not cand:
            break
        a, ns = cand[int(np.argmin([abs(c[1] - TARGET) for c in cand]))]
        path.append(a)
        s = ns
    return s, path


def bench_reflexion_retry(
    seed: int = 389,
    n_tasks: int = 80,
    max_steps: int = 10,
    max_retries: int = 3,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # tasks where first greedy attempt fails
    hard = [s for s in range(100) if _greedy(s, max_steps, set())[0] != TARGET]
    starts = rng.choice(hard, min(n_tasks, len(hard)), replace=False)
    first = 0
    after = 0
    for s0 in starts:
        s0 = int(s0)
        s, path = _greedy(s0, max_steps, set())
        first += int(s == TARGET)
        # reflexion: ban the last failed action chain's decision point
        banned = set()
        done = s == TARGET
        tries = 1
        while not done and tries <= max_retries:
            # critique: ban each (state, action) on the failed path
            cur = s0
            for a in path:
                banned.add((cur, a))
                cur = transition(cur, a)
            s, path = _greedy(s0, max_steps, banned)
            done = s == TARGET
            tries += 1
        after += int(done)
    return {
        "synthetic_refl_first_solve": first / n_tasks,
        "synthetic_refl_retry_solve": after / n_tasks,
        "synthetic_refl_gain": (after - first) / n_tasks,
    }
