from quant_fund.models.herbrand_model import bench_herbrand_model
from quant_fund.models.los_theorem import bench_los_theorem
from quant_fund.models.presburger import bench_presburger
from quant_fund.models.skolem_normal import bench_skolem_normal
from quant_fund.models.unification_fol import bench_unification_fol


def test_unification_fol():
    assert bench_unification_fol()["synthetic_unification_fol"] == 1.0


def test_skolem_normal():
    assert bench_skolem_normal()["synthetic_skolem_normal"] == 1.0


def test_herbrand_model():
    assert bench_herbrand_model()["synthetic_herbrand_model"] == 1.0


def test_presburger():
    assert bench_presburger()["synthetic_presburger"] == 1.0


def test_los_theorem():
    assert bench_los_theorem()["synthetic_los_theorem"] == 1.0
