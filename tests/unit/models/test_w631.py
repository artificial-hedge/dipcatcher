from quant_fund.models.constructible_sh import (
    bench_constructible_sh,
)
from quant_fund.models.etale_cover3 import (
    bench_etale_cover3,
)
from quant_fund.models.etale_site3 import (
    bench_etale_site3,
)
from quant_fund.models.ql_sheaf import bench_ql_sheaf
from quant_fund.models.torsion_sheaf import (
    bench_torsion_sheaf,
)
from quant_fund.models.weil_sheaf import bench_weil_sheaf


def test_etale_cover3():
    assert bench_etale_cover3()["synthetic_etale_cover3"] == 1.0


def test_etale_site3():
    assert bench_etale_site3()["synthetic_etale_site3"] == 1.0


def test_constructible_sh():
    assert bench_constructible_sh()["synthetic_constructible_sh"] == 1.0


def test_weil_sheaf():
    assert bench_weil_sheaf()["synthetic_weil_sheaf"] == 1.0


def test_torsion_sheaf():
    assert bench_torsion_sheaf()["synthetic_torsion_sheaf"] == 1.0


def test_ql_sheaf():
    assert bench_ql_sheaf()["synthetic_ql_sheaf"] == 1.0
