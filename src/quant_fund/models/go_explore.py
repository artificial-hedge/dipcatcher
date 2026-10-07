"""Go-Explore-lite (Ecoffet et al. 2021): archive of (state, traj); each
phase returns to a frontier state then explores. Deterministic env so
"return" replays actions. Metric = coverage/success vs random walk.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._ex_synth import GOAL, SIDE, WALLS, neighbors, reward, s2i


def bench_go_explore(seed: int = 2867, rounds: int = 300, tail: int = 8) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # archive: state -> (traj of actions from start, best score)
    arch: dict[int, list[int]] = {s2i((3, 0)): []}
    score: dict[int, float] = {s2i((3, 0)): 0.0}
    visited: set[int] = set()
    successes = 0

    def simulate(actions: list[int]) -> list[tuple[int, int]]:
        s = (3, 0)
        path = [s]
        for a in actions:
            legal = dict(neighbors(s))
            if a not in legal:
                break
            s = legal[a]
            path.append(s)
            if s == GOAL:
                break
        return path

    for _r in range(rounds):
        # pick archive state preferring unvisited-neighbor frontier
        keys = list(arch)
        pick = keys[int(rng.integers(len(keys)))]
        traj = list(arch[pick])
        # go (deterministic restore) + explore tail
        s = simulate(traj)[-1]
        for _ in range(tail):
            legal = neighbors(s)
            a = int(rng.choice([act for act, _ in legal]))
            s = dict(legal)[a]
            traj.append(a)
            i = s2i(s)
            visited.add(i)
            val = 1.0 if s == GOAL else (0.1 if reward(s) > 0 else 0.0)
            if s == GOAL:
                successes += 1
            if i not in arch or len(traj) < len(arch[i]):
                arch[i] = list(traj)
                score[i] = max(score.get(i, 0), val)
            if s == GOAL:
                break
    cover = len(visited) / (SIDE * SIDE - len(WALLS))
    # discovered path length to the goal vs optimal Manhattan distance
    goal_traj = arch.get(s2i(GOAL))
    path_len = float(len(goal_traj)) if goal_traj is not None else 0.0
    optimal = float((3 - GOAL[0]) + (GOAL[1] - 0))
    return {
        "synthetic_goexp_coverage": float(cover),
        "synthetic_goexp_goal_found": float(min(successes, 1)),
        "synthetic_goexp_path_len": path_len,
        "synthetic_optimal_path_len": optimal,
        "synthetic_torch_available": 0.0,
    }
