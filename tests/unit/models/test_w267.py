"""Wave-267 GPU-architecture module tests."""

import numpy as np

from quant_fund.models.bank_conflict import conflicts
from quant_fund.models.mem_coalesce import transactions
from quant_fund.models.occupancy_calc import occupancy
from quant_fund.models.shared_mem_tile import tiled_matmul
from quant_fund.models.simt_divergence import simt_run
from quant_fund.models.warp_scheduler import schedule


def test_warp_sched_issues_everything() -> None:
    out = schedule([[1, 1, 1], [2, 2]])
    assert sum(len(x) for x in out) == 5
    assert all(len(x) <= 1 for x in out)


def test_warp_sched_respects_latency() -> None:
    # single warp with latency-3 instructions can't issue back-to-back
    out = schedule([[3, 3]])
    cycles = [i for i, x in enumerate(out) if x]
    assert cycles[-1] - cycles[0] >= 3


def test_simt_counts() -> None:
    pred = np.array([[1, 0, 1], [0, 1, 0]])
    counts, _ = simt_run(pred, 3)
    assert counts == [2, 1]


def test_bank_conflict_stride1() -> None:
    assert conflicts(np.arange(32)) == 1


def test_bank_conflict_stride32() -> None:
    assert conflicts(np.arange(32) * 32) == 32


def test_coalesce_contiguous() -> None:
    assert transactions(4 * np.arange(32)) <= 2


def test_coalesce_scatter() -> None:
    scattered = np.arange(0, 32 * 128, 128)
    assert transactions(scattered) == 32


def test_occupancy_bounds() -> None:
    assert occupancy(256, 32, 8192) == 1.0
    assert occupancy(1024, 128, 49152) < 1.0


def test_tiled_matmul_exact() -> None:
    rng = np.random.RandomState(0)
    a = rng.rand(13, 11)
    b = rng.rand(11, 9)
    assert np.allclose(tiled_matmul(a, b, 4), a @ b)
