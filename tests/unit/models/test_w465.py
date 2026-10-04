from quant_fund.models.affinoid_alg import bench_affinoid_alg
from quant_fund.models.dagger_groth import bench_dagger_groth
from quant_fund.models.gauss_point import bench_gauss_point
from quant_fund.models.kedlaya_renorm import bench_kedlaya_renorm
from quant_fund.models.raynaud_gen import bench_raynaud_gen
from quant_fund.models.weierstrass_prep import bench_weierstrass_prep


def test_kedlaya_renorm():
    assert bench_kedlaya_renorm()["synthetic_kedlaya_renorm"] == 1.0


def test_dagger_groth():
    assert bench_dagger_groth()["synthetic_dagger_groth"] == 1.0


def test_raynaud_gen():
    assert bench_raynaud_gen()["synthetic_raynaud_gen"] == 1.0


def test_weierstrass_prep():
    assert bench_weierstrass_prep()["synthetic_weierstrass_prep"] == 1.0


def test_gauss_point():
    assert bench_gauss_point()["synthetic_gauss_point"] == 1.0


def test_affinoid_alg():
    assert bench_affinoid_alg()["synthetic_affinoid_alg"] == 1.0
