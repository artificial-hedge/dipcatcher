from quant_fund.models.cyclotomic_spec import (
    bench_cyclotomic_spec,
)
from quant_fund.models.negative_cyclic import (
    bench_negative_cyclic,
)
from quant_fund.models.periodic_cyclic import (
    bench_periodic_cyclic,
)
from quant_fund.models.tate_construction import (
    bench_tate_construction,
)
from quant_fund.models.tc_spec import bench_tc_spec
from quant_fund.models.tr_structure import bench_tr_structure


def test_cyclotomic_spec():
    assert bench_cyclotomic_spec()["synthetic_cyclotomic_spec"] == 1.0


def test_tr_structure():
    assert bench_tr_structure()["synthetic_tr_structure"] == 1.0


def test_tc_spec():
    assert bench_tc_spec()["synthetic_tc_spec"] == 1.0


def test_negative_cyclic():
    assert bench_negative_cyclic()["synthetic_negative_cyclic"] == 1.0


def test_periodic_cyclic():
    assert bench_periodic_cyclic()["synthetic_periodic_cyclic"] == 1.0


def test_tate_construction():
    assert bench_tate_construction()["synthetic_tate_construction"] == 1.0
