"""Wave-260 matching canon tests."""

import numpy as np

from quant_fund.models.gale_chu import bench_gale_chu, gale_chu
from quant_fund.models.gale_shapley import _is_stable, bench_gale_shapley, gale_shapley
from quant_fund.models.hopcroft_karp import bench_hopcroft_karp, hopcroft_karp
from quant_fund.models.konig_cover import bench_konig_cover, konig_cover
from quant_fund.models.kuhn_munkres import bench_kuhn_munkres, hungarian
from quant_fund.models.topo_layers import bench_topo_layers, kahn_layers


def test_gs_stable():
    men = np.array([[0, 1], [1, 0]])
    women = np.array([[1, 0], [0, 1]])
    m = gale_shapley(men, women)
    assert _is_stable(m, men, women)


def test_gs_bench():
    assert bench_gale_shapley()["synthetic_gs_stable"] == 1.0


def test_hk_match():
    m = hopcroft_karp([[0, 1], [0], [1]], 2)
    assert len(m) == 2


def test_hk_bench():
    assert bench_hopcroft_karp()["synthetic_hk_maximal"] == 1.0


def test_hungarian_basic():
    val, assign = hungarian(np.array([[1.0, 5.0], [4.0, 2.0]]))
    assert np.isclose(val, 3.0) and sorted(assign) == [0, 1]


def test_hungarian_bench():
    assert bench_kuhn_munkres()["synthetic_hungarian_optimal"] == 1.0


def test_konig():
    cl, cr = konig_cover([[0], [0]], 1)
    assert len(cl) + len(cr) == 1


def test_konig_bench():
    assert bench_konig_cover()["synthetic_konig_min_cover"] == 1.0


def test_gale_chu_quota():
    students = np.array([[0, 1], [0, 1], [1, 0]])
    hosp = np.array([[0, 1, 2], [0, 1, 2]])
    out = gale_chu(students, hosp, np.array([2, 1]))
    assert len(out[0]) <= 2


def test_gale_chu_bench():
    assert bench_gale_chu()["synthetic_gc_valid"] == 1.0


def test_topo():
    layers = kahn_layers([(0, 1), (0, 2), (1, 3)], 4)
    assert layers[0] == [0]


def test_topo_bench():
    assert bench_topo_layers()["synthetic_topo_correct"] == 1.0
