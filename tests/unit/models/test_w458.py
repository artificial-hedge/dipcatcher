from quant_fund.models.delta_matroid import bench_delta_matroid
from quant_fund.models.matroid_minor import bench_matroid_minor
from quant_fund.models.matroid_rep import bench_matroid_rep
from quant_fund.models.regular_mat import bench_regular_mat
from quant_fund.models.transversal_mat import bench_transversal_mat
from quant_fund.models.tutte_poly import bench_tutte_poly


def test_transversal_mat():
    assert bench_transversal_mat()["synthetic_transversal_mat"] == 1.0


def test_matroid_rep():
    assert bench_matroid_rep()["synthetic_matroid_rep"] == 1.0


def test_tutte_poly():
    assert bench_tutte_poly()["synthetic_tutte_poly"] == 1.0


def test_matroid_minor():
    assert bench_matroid_minor()["synthetic_matroid_minor"] == 1.0


def test_regular_mat():
    assert bench_regular_mat()["synthetic_regular_mat"] == 1.0


def test_delta_matroid():
    assert bench_delta_matroid()["synthetic_delta_matroid"] == 1.0
