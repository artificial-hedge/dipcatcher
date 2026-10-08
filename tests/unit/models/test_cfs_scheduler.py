"""CFS vruntime-spread invariant tests."""

from __future__ import annotations

from quant_fund.models.cfs_scheduler import NICE_0, bench_cfs_scheduler, run_cfs


def test_spread_bound_is_real():
    """Independent re-simulation: the runnable-set vruntime spread must
    stay within one tick-quantum of the lightest runnable weight —
    a property of scheduler STATE, not of the argmin choice (the old
    tautological ``picked == argmin`` self-check could never fail)."""
    tasks = [(2.0, 400), (1.0, 400), (3.0, 400)]
    tick = 4
    vrt = [0.0] * len(tasks)
    rem = [b for _w, b in tasks]
    t = 0
    while t < 600 and any(r > 0 for r in rem):
        run = [i for i in range(len(tasks)) if rem[i] > 0]
        pid = min(run, key=lambda i: vrt[i])
        step = min(tick, rem[pid])
        rem[pid] -= step
        vrt[pid] += step * NICE_0 / tasks[pid][0]
        bound = tick * NICE_0 / min(tasks[i][0] for i in run)
        spread = max(vrt[i] for i in run) - min(vrt[i] for i in run)
        assert spread <= bound + 1e-9
        t += step


def test_run_cfs_grants_all_work():
    tasks = [(1.0, 100), (2.0, 200)]
    g = run_cfs(tasks, 10_000)
    assert g == {0: 100.0, 1: 200.0}


def test_heavier_weight_gets_more_cpu():
    tasks = [(4.0, 100_000), (1.0, 100_000)]
    g = run_cfs(tasks, 4000)
    assert g[0] > 2.5 * g[1]


def test_bench():
    out = bench_cfs_scheduler()
    assert out["synthetic_fair_share"] == 1.0
    assert out["synthetic_proportional_weight"] == 1.0
    assert out["synthetic_vruntime_spread_ok"] == 1.0
    assert "synthetic_min_vruntime_picks" not in out
