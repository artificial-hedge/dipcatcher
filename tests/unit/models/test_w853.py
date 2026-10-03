from quant_fund.models.gll_nodes import (
    bench_gll_nodes,
)
from quant_fund.models.hp_refinement import (
    bench_hp_refinement,
)
from quant_fund.models.mortar_method import (
    bench_mortar_method,
)
from quant_fund.models.sem_grid import (
    bench_sem_grid,
)
from quant_fund.models.spectral_element import (
    bench_spectral_element,
)
from quant_fund.models.tensor_product_sem import (
    bench_tensor_product_sem,
)


def test_sem_grid():
    assert bench_sem_grid()["synthetic_sem_grid"] == 1.0


def test_gll_nodes():
    assert bench_gll_nodes()["synthetic_gll_nodes"] == 1.0


def test_spectral_element():
    assert bench_spectral_element()["synthetic_spectral_element"] == 1.0


def test_mortar_method():
    assert bench_mortar_method()["synthetic_mortar_method"] == 1.0


def test_tensor_product_sem():
    assert bench_tensor_product_sem()["synthetic_tensor_product_sem"] == 1.0


def test_hp_refinement():
    assert bench_hp_refinement()["synthetic_hp_refinement"] == 1.0
