"""ReAct — interleaved reasoning + acting (Yao et al. 2023) (SYNTHETIC).

An agent that observes state each step and picks the greedy-to-goal
action solves the tool-graph task far more often than a blind agent
following a fixed action plan — the observation loop is the value-add.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._agent_synth import (  # noqa: F401
    N_ACTIONS,
    bfs_solution,
    is_goal,
    transition,
)


def bench_react_loop(
    seed: int = 373,
    n_tasks: int = 60,
    max_steps: int = 12,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    solved_react = 0
    solved_blind = 0
    starts = rng.integers(0, 100, n_tasks)
    for s0 in starts:
        # ReAct-style: observe → pick action that lands closest to goal
        s = int(s0)
        for _t in range(max_steps):
            if is_goal(s) and s != int(s0):
                solved_react += 1
                break
            cand = [transition(s, a) for a in range(N_ACTIONS)]

            # greedy heuristic: min distance to a goal residue
            def dgoal(v):
                return min(v % 10, 10 - v % 10)

            s = cand[int(np.argmin([dgoal(c) for c in cand]))]
        else:
            solved_react += int(is_goal(s))
        # blind: precomputed plan executed without observation — replay
        # a solution that exists for a DIFFERENT random start
        other = int(rng.integers(0, 100))
        plan = bfs_solution(other, max_steps)
        s = int(s0)
        done = False
        if plan:
            for a in plan:
                s = transition(s, a)
                if is_goal(s):
                    done = True
                    break
            solved_blind += int(done or is_goal(s))
    return {
        "synthetic_react_solve": solved_react / n_tasks,
        "synthetic_react_blind_solve": solved_blind / n_tasks,
        "synthetic_react_gain": (solved_react - solved_blind) / n_tasks,
    }
