from quant_fund.models.closed_unbounded import bench_closed_unbounded
from quant_fund.models.club_set import bench_club_set
from quant_fund.models.mahlo_cardinal import bench_mahlo_cardinal
from quant_fund.models.partition_calc import bench_partition_calc
from quant_fund.models.stationary_set import bench_stationary_set
from quant_fund.models.ultrafilter_toy import bench_ultrafilter_toy


def test_club_set():
    assert bench_club_set()["synthetic_club_set"] == 1.0


def test_stationary_set():
    assert bench_stationary_set()["synthetic_stationary_set"] == 1.0


def test_ultrafilter_toy():
    assert bench_ultrafilter_toy()["synthetic_ultrafilter_toy"] == 1.0


def test_partition_calc():
    assert bench_partition_calc()["synthetic_partition_calc"] == 1.0


def test_closed_unbounded():
    assert bench_closed_unbounded()["synthetic_closed_unbounded"] == 1.0


def test_mahlo_cardinal():
    assert bench_mahlo_cardinal()["synthetic_mahlo_cardinal"] == 1.0
