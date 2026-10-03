from quant_fund.models.hall_algebra import bench_hall_algebra
from quant_fund.models.joyce_hall import bench_joyce_hall
from quant_fund.models.lusztig_hall import bench_lusztig_hall
from quant_fund.models.ringel_hall import bench_ringel_hall
from quant_fund.models.schiffmann_hall import (
    bench_schiffmann_hall,
)
from quant_fund.models.toen_hall import bench_toen_hall


def test_hall_algebra():
    assert bench_hall_algebra()["synthetic_hall_algebra"] == 1.0


def test_ringel_hall():
    assert bench_ringel_hall()["synthetic_ringel_hall"] == 1.0


def test_toen_hall():
    assert bench_toen_hall()["synthetic_toen_hall"] == 1.0


def test_lusztig_hall():
    assert bench_lusztig_hall()["synthetic_lusztig_hall"] == 1.0


def test_schiffmann_hall():
    assert bench_schiffmann_hall()["synthetic_schiffmann_hall"] == 1.0


def test_joyce_hall():
    assert bench_joyce_hall()["synthetic_joyce_hall"] == 1.0
