from quant_fund.models.bridgeland_hall import (
    bench_bridgeland_hall,
)
from quant_fund.models.calaque_hall import bench_calaque_hall
from quant_fund.models.green_hall import bench_green_hall
from quant_fund.models.kontsevich_soibelman import (
    bench_kontsevich_soibelman,
)
from quant_fund.models.morita_hall import bench_morita_hall
from quant_fund.models.mozgovoy_hall import (
    bench_mozgovoy_hall,
)


def test_green_hall():
    assert bench_green_hall()["synthetic_green_hall"] == 1.0


def test_bridgeland_hall():
    assert bench_bridgeland_hall()["synthetic_bridgeland_hall"] == 1.0


def test_kontsevich_soibelman():
    assert bench_kontsevich_soibelman()["synthetic_kontsevich_soibelman"] == 1.0


def test_mozgovoy_hall():
    assert bench_mozgovoy_hall()["synthetic_mozgovoy_hall"] == 1.0


def test_morita_hall():
    assert bench_morita_hall()["synthetic_morita_hall"] == 1.0


def test_calaque_hall():
    assert bench_calaque_hall()["synthetic_calaque_hall"] == 1.0
