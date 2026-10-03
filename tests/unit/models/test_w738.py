from quant_fund.models.abraham_bipartite import (
    bench_abraham_bipartite,
)
from quant_fund.models.bettinelli_jacob import (
    bench_bettinelli_jacob,
)
from quant_fund.models.chapuy_dolega import bench_chapuy_dolega
from quant_fund.models.curien_legall import bench_curien_legall
from quant_fund.models.le_gall_miermont import (
    bench_le_gall_miermont,
)
from quant_fund.models.marckert_mokkadem import (
    bench_marckert_mokkadem,
)


def test_marckert_mokkadem():
    assert bench_marckert_mokkadem()["synthetic_marckert_mokkadem"] == 1.0


def test_le_gall_miermont():
    assert bench_le_gall_miermont()["synthetic_le_gall_miermont"] == 1.0


def test_curien_legall():
    assert bench_curien_legall()["synthetic_curien_legall"] == 1.0


def test_abraham_bipartite():
    assert bench_abraham_bipartite()["synthetic_abraham_bipartite"] == 1.0


def test_bettinelli_jacob():
    assert bench_bettinelli_jacob()["synthetic_bettinelli_jacob"] == 1.0


def test_chapuy_dolega():
    assert bench_chapuy_dolega()["synthetic_chapuy_dolega"] == 1.0
