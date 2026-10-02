import numpy as np

from quant_fund.models.metaheuristic_optimizers import (
    bench_metaheuristics,
    differential_evolution,
    genetic_algorithm,
    nsga2_sort,
    particle_swarm,
    simulated_annealing,
)


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def test_sa_sphere():
    r = simulated_annealing(sphere, np.array([3.0, -2.0]), t0=2.0, it=2000, seed=1)
    assert float(r["f"]) < 1.0


def test_de_sphere():
    r = differential_evolution(sphere, np.full(3, -5.0), np.full(3, 5.0), it=40, seed=2)
    assert float(r["f"]) < 0.5


def test_pso_sphere():
    r = particle_swarm(sphere, np.full(3, -5.0), np.full(3, 5.0), it=60, seed=3)
    assert float(r["f"]) < 0.1


def test_ga_sphere():
    r = genetic_algorithm(sphere, np.full(2, -4.0), np.full(2, 4.0), it=80, seed=4)
    assert float(r["f"]) < 1.0


def test_nsga2_fronts():
    obj = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [1.5, 1.5]])
    r = nsga2_sort(obj)
    assert len(np.asarray(r["rank"])) == 4
    fronts = r["fronts"]
    assert set(fronts[0]) == {0, 1, 2}


def test_bench_metaheuristics_runs():
    out = bench_metaheuristics(seed=540)
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_nsga_front0_true"] >= 48
