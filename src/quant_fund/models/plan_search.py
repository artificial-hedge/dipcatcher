"""Tree-of-thoughts / plan search (Yao et al. 2023, bis) (SYNTHETIC).

Beam search over the tool-graph finds solutions where greedy ReAct
fails — branching factor × depth beats single-path reasoning.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._agent_synth import N_ACTIONS, transition

TARGET = 42


def _greedy(s0: int, max_steps: int) -> bool:
    s = s0
    for _t in range(max_steps):
        if s == TARGET:
            return True
        cand = [transition(s, a) for a in range(N_ACTIONS)]
        s = cand[int(np.argmin([abs(c - TARGET) for c in cand]))]
    return s == TARGET


def _beam(s0: int, width: int, max_steps: int) -> bool:
    front = [(s0, 0)]
    for _t in range(max_steps):
        nxt = []
        for s, _ in front:
            for a in range(N_ACTIONS):
                ns = transition(s, a)
                if ns == TARGET:
                    return True
                nxt.append((ns, abs(ns - TARGET)))
        nxt.sort(key=lambda x: x[1])
        front = nxt[:width]
    return any(s == TARGET for s, _ in front)


def bench_plan_search(
    seed: int = 383,
    n_tasks: int = 80,
    width: int = 6,
    max_steps: int = 10,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # restrict to states where greedy fails (need escape moves)
    hard = [s for s in range(100) if not _greedy(s, max_steps)]
    starts = rng.choice(hard, min(n_tasks, len(hard)), replace=False)
    g = float(np.mean([_greedy(int(s), max_steps) for s in starts]))
    b = float(np.mean([_beam(int(s), width, max_steps) for s in starts]))

    def _bfs_target(s0):
        from collections import deque

        q = deque([s0])
        seen = {s0}
        for _d in range(max_steps):
            for _k in range(len(q)):
                v = q.popleft()
                if v == TARGET:
                    return True
                for a in range(N_ACTIONS):
                    ns = transition(v, a)
                    if ns not in seen:
                        seen.add(ns)
                        q.append(ns)
        return False

    oracle = float(np.mean([_bfs_target(int(s)) for s in starts]))
    return {
        "synthetic_plan_beam_solve": b,
        "synthetic_plan_greedy_solve": g,
        "synthetic_plan_oracle_solve": oracle,
        "synthetic_plan_gain": b - g,
    }
