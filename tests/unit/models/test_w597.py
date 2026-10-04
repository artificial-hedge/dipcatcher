from quant_fund.models.bousfield_kan import (
    bench_bousfield_kan,
)
from quant_fund.models.curtis_lower import bench_curtis_lower
from quant_fund.models.dror_smith import bench_dror_smith
from quant_fund.models.lannes_t import bench_lannes_t
from quant_fund.models.periodicity_thm import (
    bench_periodicity_thm,
)
from quant_fund.models.telescope_conj import (
    bench_telescope_conj,
)


def test_curtis_lower():
    assert bench_curtis_lower()["synthetic_curtis_lower"] == 1.0


def test_bousfield_kan():
    assert bench_bousfield_kan()["synthetic_bousfield_kan"] == 1.0


def test_lannes_t():
    assert bench_lannes_t()["synthetic_lannes_t"] == 1.0


def test_dror_smith():
    assert bench_dror_smith()["synthetic_dror_smith"] == 1.0


def test_telescope_conj():
    assert bench_telescope_conj()["synthetic_telescope_conj"] == 1.0


def test_periodicity_thm():
    assert bench_periodicity_thm()["synthetic_periodicity_thm"] == 1.0
