from quant_fund.models.brave_new_ring import (
    bench_brave_new_ring,
)
from quant_fund.models.e_infty_space import (
    bench_e_infty_space,
)
from quant_fund.models.formal_moduli import (
    bench_formal_moduli,
)
from quant_fund.models.log_ring import bench_log_ring
from quant_fund.models.orient_cohom import bench_orient_cohom
from quant_fund.models.thom_constr import bench_thom_constr


def test_e_infty_space():
    assert bench_e_infty_space()["synthetic_e_infty_space"] == 1.0


def test_brave_new_ring():
    assert bench_brave_new_ring()["synthetic_brave_new_ring"] == 1.0


def test_thom_constr():
    assert bench_thom_constr()["synthetic_thom_constr"] == 1.0


def test_log_ring():
    assert bench_log_ring()["synthetic_log_ring"] == 1.0


def test_orient_cohom():
    assert bench_orient_cohom()["synthetic_orient_cohom"] == 1.0


def test_formal_moduli():
    assert bench_formal_moduli()["synthetic_formal_moduli"] == 1.0
