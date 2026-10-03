from quant_fund.models.argument_principle import bench_argument_principle
from quant_fund.models.cauchy_integral import bench_cauchy_integral
from quant_fund.models.conformal_map import bench_conformal_map
from quant_fund.models.laurent_series import bench_laurent_series
from quant_fund.models.liouville import bench_liouville
from quant_fund.models.residue_calc import bench_residue_calc


def test_cauchy_integral():
    assert bench_cauchy_integral()["synthetic_cauchy_integral"] == 1.0


def test_residue_calc():
    assert bench_residue_calc()["synthetic_residue_calc"] == 1.0


def test_laurent_series():
    assert bench_laurent_series()["synthetic_laurent_series"] == 1.0


def test_argument_principle():
    assert bench_argument_principle()["synthetic_argument_principle"] == 1.0


def test_conformal_map():
    assert bench_conformal_map()["synthetic_conformal_map"] == 1.0


def test_liouville():
    assert bench_liouville()["synthetic_liouville"] == 1.0
