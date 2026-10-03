"""Task-DAG executor: ready-queue scheduling with dependency counting."""

import numpy as np

_SEED = 20261231 + 644


def dag_exec(n_tasks: int, edge_prob: float, rng: np.random.RandomState) -> tuple[int, bool]:
    # random DAG via topological order
    order = rng.permutation(n_tasks)
    deps: dict[int, set[int]] = {i: set() for i in range(n_tasks)}
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if rng.rand() < edge_prob:
                deps[order[j]].add(order[i])
    done: set[int] = set()
    steps = 0
    indeg = {t: len(d) for t, d in deps.items()}
    while len(done) < n_tasks:
        ready = [t for t in range(n_tasks) if indeg[t] == 0 and t not in done]
        if not ready:
            return steps, False
        for t in ready:
            done.add(t)
            for u in range(n_tasks):
                if t in deps[u]:
                    indeg[u] -= 1
        steps += 1
    return steps, True


def bench_task_dag(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    trials = 40
    for _ in range(trials):
        _, good = dag_exec(rng.randint(5, 15), 0.15, rng)
        ok += good
    return {"synthetic_dag_scheduled": ok / trials}
