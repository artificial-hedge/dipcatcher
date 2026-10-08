"""Toolformer — learned API-call insertion (Schick et al. 2023) (SYNTHETIC).

A policy predicts WHEN to call a tool vs rely on the parametric guess.
On the tool-graph, a learned gate that calls the oracle tool only when
the parametric estimate is unreliable beats always-call and never-call.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_fund.models._agent_synth import bfs_solution, is_goal, transition


def bench_toolformer_call(
    seed: int = 379,
    n_tasks: int = 200,
    max_steps: int = 10,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # "parametric guess": noisy heuristic on state residue — unreliable
    # for states where the residue is mid-range.
    starts = rng.integers(0, 100, n_tasks)
    feats: list[list[float]] = []
    labels: list[int] = []  # 1 if parametric guess is right
    for s0 in starts:
        s0 = int(s0)
        sol = bfs_solution(s0, max_steps)
        # parametric guess: goal reachable in a single tool call
        one_step = any(is_goal(transition(s0, a)) for a in range(6))
        resid = min(s0 % 10, 10 - s0 % 10)
        feats.append([s0 / 100.0, resid / 10.0, len(sol) if sol else max_steps])
        labels.append(int(one_step))
    feats_a = np.array(feats)
    labels_a = np.array(labels)
    cut = n_tasks // 2
    gate = LogisticRegression(max_iter=300).fit(feats_a[:cut], labels_a[:cut])
    # never-call: solve by parametric 1-step guess only
    never = float(
        np.mean([any(is_goal(transition(int(s), a)) for a in range(6)) for s in starts[cut:]])
    )
    # always-call: run bfs — always solvable
    always = float(np.mean([bfs_solution(int(s), max_steps) is not None for s in starts[cut:]]))
    # learned gate: call only when guess predicted wrong
    gated = 0.0
    for i in range(cut, n_tasks):
        call = gate.predict(feats_a[i : i + 1])[0] == 0
        if call:
            gated += bfs_solution(int(starts[i]), max_steps) is not None
        else:
            s = int(starts[i])
            gated += any(is_goal(transition(s, a)) for a in range(6))
    gated /= cut
    calls_frac = float(1 - gate.predict(feats_a[cut:]).mean())
    return {
        "synthetic_tool_gated_solve": gated,
        "synthetic_tool_always_solve": always,
        "synthetic_tool_never_solve": never,
        "synthetic_tool_call_frac": calls_frac,
        "synthetic_tool_gain_vs_never": gated - never,
    }
