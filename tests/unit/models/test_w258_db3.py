"""Wave-258 databases-3 canon tests."""

import numpy as np

from quant_fund.models.adaptive_qp import bench_adaptive_qp
from quant_fund.models.bitmap_index import BitmapIndex, bench_bitmap_index
from quant_fund.models.cascades_opt import _cost, bench_cascades_opt, optimize
from quant_fund.models.func_dep import _holds, bench_func_dep
from quant_fund.models.vectorized_exec import _vectorized, bench_vectorized_exec
from quant_fund.models.zone_map import _zone_scan, bench_zone_map


def test_cascades_opt():
    cards = {"a": 100.0, "b": 10.0, "c": 1000.0}
    plan = optimize(["a", "b", "c"], cards)
    assert _cost(plan, cards) > 0


def test_cascades_bench():
    assert bench_cascades_opt()["synthetic_cascades_optimal"] == 1.0


def test_vec_equiv():
    out = _vectorized(np.array([1.0, 5.0, 9.0]), 4.0)
    assert np.allclose(out, [11.0, 19.0])


def test_vec_bench():
    r = bench_vectorized_exec()
    assert r["synthetic_vec_equiv"] == 1.0 and r["synthetic_vec_call_reduction"] > 0.9


def test_zone_scan():
    got, scanned = _zone_scan(np.arange(100.0), 40.0, 60.0, 25)
    assert np.allclose(got, np.arange(40, 61))


def test_zone_bench():
    assert bench_zone_map()["synthetic_zone_equiv"] == 1.0


def test_fd_holds():
    rows = np.array([[0, 1], [1, 2], [0, 1]])
    assert _holds(rows, (0,), 1)


def test_fd_bench():
    assert bench_func_dep()["synthetic_fd_found"] == 1.0


def test_bitmap():
    idx = BitmapIndex(np.array([1, 2, 1, 3]))
    assert np.array_equal(idx.query([1]), [0, 2])


def test_bitmap_bench():
    assert bench_bitmap_index()["synthetic_bitmap_equiv"] == 1.0


def test_adaptive_bench():
    assert bench_adaptive_qp()["synthetic_adaptive_optimal"] == 1.0
