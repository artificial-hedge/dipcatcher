from quant_fund.models.dk_motive import bench_dk_motive
from quant_fund.models.motivic_adem import (
    bench_motivic_adem,
)
from quant_fund.models.motivic_steenrod import (
    bench_motivic_steenrod,
)
from quant_fund.models.motivic_transfer import (
    bench_motivic_transfer,
)
from quant_fund.models.power_operations import (
    bench_power_operations,
)
from quant_fund.models.simplicial_motive import (
    bench_simplicial_motive,
)


def test_motivic_steenrod():
    assert bench_motivic_steenrod()["synthetic_motivic_steenrod"] == 1.0


def test_motivic_adem():
    assert bench_motivic_adem()["synthetic_motivic_adem"] == 1.0


def test_power_operations():
    assert bench_power_operations()["synthetic_power_operations"] == 1.0


def test_simplicial_motive():
    assert bench_simplicial_motive()["synthetic_simplicial_motive"] == 1.0


def test_dk_motive():
    assert bench_dk_motive()["synthetic_dk_motive"] == 1.0


def test_motivic_transfer():
    assert bench_motivic_transfer()["synthetic_motivic_transfer"] == 1.0
