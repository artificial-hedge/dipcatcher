from quant_fund.models.calabi_yau_alg import bench_calabi_yau_alg
from quant_fund.models.connes_nc import bench_connes_nc
from quant_fund.models.cyclic_coh import bench_cyclic_coh
from quant_fund.models.ginzburg_dga import bench_ginzburg_dga
from quant_fund.models.hochschild_coh import bench_hochschild_coh
from quant_fund.models.nc_scheme import bench_nc_scheme


def test_hochschild_coh():
    assert bench_hochschild_coh()["synthetic_hochschild_coh"] == 1.0


def test_cyclic_coh():
    assert bench_cyclic_coh()["synthetic_cyclic_coh"] == 1.0


def test_nc_scheme():
    assert bench_nc_scheme()["synthetic_nc_scheme"] == 1.0


def test_calabi_yau_alg():
    assert bench_calabi_yau_alg()["synthetic_calabi_yau_alg"] == 1.0


def test_ginzburg_dga():
    assert bench_ginzburg_dga()["synthetic_ginzburg_dga"] == 1.0


def test_connes_nc():
    assert bench_connes_nc()["synthetic_connes_nc"] == 1.0
