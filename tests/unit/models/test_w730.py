from quant_fund.models.groth_tame import bench_groth_tame
from quant_fund.models.grothendieck_muw import (
    bench_grothendieck_muw,
)
from quant_fund.models.kato_swan import bench_kato_swan
from quant_fund.models.raynaud_pencil import (
    bench_raynaud_pencil,
)
from quant_fund.models.saito_epsilon import (
    bench_saito_epsilon,
)
from quant_fund.models.swan_conductor import (
    bench_swan_conductor,
)


def test_grothendieck_muw():
    assert bench_grothendieck_muw()["synthetic_grothendieck_muw"] == 1.0


def test_raynaud_pencil():
    assert bench_raynaud_pencil()["synthetic_raynaud_pencil"] == 1.0


def test_saito_epsilon():
    assert bench_saito_epsilon()["synthetic_saito_epsilon"] == 1.0


def test_swan_conductor():
    assert bench_swan_conductor()["synthetic_swan_conductor"] == 1.0


def test_groth_tame():
    assert bench_groth_tame()["synthetic_groth_tame"] == 1.0


def test_kato_swan():
    assert bench_kato_swan()["synthetic_kato_swan"] == 1.0
