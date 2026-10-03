from quant_fund.models.anisotropic_quad import (
    bench_anisotropic_quad,
)
from quant_fund.models.combination_technique import (
    bench_combination_technique,
)
from quant_fund.models.dimension_adaptive import (
    bench_dimension_adaptive,
)
from quant_fund.models.gerstner_griebel import (
    bench_gerstner_griebel,
)
from quant_fund.models.smolyak_grid import (
    bench_smolyak_grid,
)
from quant_fund.models.sparse_tensor import (
    bench_sparse_tensor,
)


def test_smolyak_grid():
    assert bench_smolyak_grid()["synthetic_smolyak_grid"] == 1.0


def test_sparse_tensor():
    assert bench_sparse_tensor()["synthetic_sparse_tensor"] == 1.0


def test_anisotropic_quad():
    assert bench_anisotropic_quad()["synthetic_anisotropic_quad"] == 1.0


def test_gerstner_griebel():
    assert bench_gerstner_griebel()["synthetic_gerstner_griebel"] == 1.0


def test_combination_technique():
    assert bench_combination_technique()["synthetic_combination_technique"] == 1.0


def test_dimension_adaptive():
    assert bench_dimension_adaptive()["synthetic_dimension_adaptive"] == 1.0
