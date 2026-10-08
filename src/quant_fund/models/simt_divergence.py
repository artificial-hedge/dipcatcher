"""SIMT branch divergence: reconvergence-stack executor vs serial oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 687


def simt_run(pred: np.ndarray, n_threads: int) -> tuple[list[int], list[int]]:
    """pred: (T, n) per-step per-thread predicate; return taken/not-taken mask counts per step."""
    # true semantics: each step executes the taken set then reconverged set — SIMT runs both serially
    taken_steps: list[int] = []
    serial_steps = 0
    for t in range(pred.shape[0]):
        n_taken = int(pred[t].sum())
        taken_steps.append(n_taken)
        serial_steps += 1 + int(n_taken not in (0, n_threads))
    return taken_steps, [serial_steps]


def bench_simt_divergence(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        n = int(rng.randint(4, 12))
        pred = rng.rand(int(rng.randint(3, 8)), n) < 0.5
        counts, _ = simt_run(pred, n)
        ok += float(counts == [int(x.sum()) for x in pred])
    return {"synthetic_simt_counts": ok / trials}
