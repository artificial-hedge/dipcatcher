from quant_fund.models.homeo_top import bench_homeo_top
from quant_fund.models.locally_compact import bench_locally_compact
from quant_fund.models.open_cover import bench_open_cover
from quant_fund.models.paracompact import bench_paracompact
from quant_fund.models.partition_unity import bench_partition_unity
from quant_fund.models.quotient_map import bench_quotient_map


def test_quotient_map():
    assert bench_quotient_map()["synthetic_quotient_map"] == 1.0


def test_open_cover():
    assert bench_open_cover()["synthetic_open_cover"] == 1.0


def test_locally_compact():
    assert bench_locally_compact()["synthetic_locally_compact"] == 1.0


def test_homeo_top():
    assert bench_homeo_top()["synthetic_homeo_top"] == 1.0


def test_paracompact():
    assert bench_paracompact()["synthetic_paracompact"] == 1.0


def test_partition_unity():
    assert bench_partition_unity()["synthetic_partition_unity"] == 1.0
