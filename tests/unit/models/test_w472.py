from quant_fund.models.alg_cobordism import bench_alg_cobordism
from quant_fund.models.hermitian_k import bench_hermitian_k
from quant_fund.models.motivic_stem2 import bench_motivic_stem2
from quant_fund.models.oriented_coh import bench_oriented_coh
from quant_fund.models.rostmotive import bench_rostmotive
from quant_fund.models.slice_spec import bench_slice_spec


def test_alg_cobordism():
    assert bench_alg_cobordism()["synthetic_alg_cobordism"] == 1.0


def test_hermitian_k():
    assert bench_hermitian_k()["synthetic_hermitian_k"] == 1.0


def test_oriented_coh():
    assert bench_oriented_coh()["synthetic_oriented_coh"] == 1.0


def test_slice_spec():
    assert bench_slice_spec()["synthetic_slice_spec"] == 1.0


def test_motivic_stem2():
    assert bench_motivic_stem2()["synthetic_motivic_stem2"] == 1.0


def test_rostmotive():
    assert bench_rostmotive()["synthetic_rostmotive"] == 1.0
