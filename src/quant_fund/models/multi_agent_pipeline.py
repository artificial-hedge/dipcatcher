"""Planner→executor→critic decomposition (multi-agent pipeline) (SYNTHETIC).

Planner proposes action sequences, executor simulates them, critic
verifies goal-reach and returns failures for replanning. The pipeline
solves more tasks than a monolithic single-pass policy.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._agent_synth import N_ACTIONS, is_goal, transition


def _plan(s0: int, depth: int, rng: np.random.Generator) -> list[int]:
    """Planner: random beam of action sequences, scored by residue."""
    best = None
    best_score = -1
    for _i in range(24):
        seq = rng.integers(0, N_ACTIONS, depth)
        s = s0
        for a in seq:
            s = transition(s, int(a))
        sc = -min(s % 10, 10 - s % 10)
        if best_score < sc:
            best_score = sc
            best = seq
    return list(map(int, best)) if best is not None else []


def _exec(s0: int, seq: list[int]) -> tuple[int, list[int]]:
    s = s0
    visited = []
    for a in seq:
        s = transition(s, a)
        visited.append(s)
        if is_goal(s):
            break
    return s, visited


def bench_multi_agent_pipeline(
    seed: int = 397,
    n_tasks: int = 60,
    depth: int = 10,
    n_replan: int = 3,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pipe = 0
    mono = 0
    for s0 in rng.integers(0, 100, n_tasks):
        s0 = int(s0)
        # monolith: single plan, no critic feedback
        seq = _plan(s0, depth, rng)
        s, _ = _exec(s0, seq)
        mono += int(is_goal(s))
        # pipeline: critic rejects failures → replan with feedback
        s_fin = s0
        solved = False
        banned_prefix = 0
        for _r in range(n_replan):
            seq = _plan(s0, depth - banned_prefix, rng)
            s_fin, vis = _exec(s0, seq)
            if is_goal(s_fin) and len(vis) > 0:
                solved = True
                break
            banned_prefix = min(banned_prefix + 2, depth - 1)
        pipe += int(solved)
    return {
        "synthetic_map_pipe_solve": pipe / n_tasks,
        "synthetic_map_mono_solve": mono / n_tasks,
        "synthetic_map_gain": (pipe - mono) / n_tasks,
    }
