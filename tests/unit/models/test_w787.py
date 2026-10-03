from quant_fund.models.gubinelli_sewing import (
    bench_gubinelli_sewing,
)
from quant_fund.models.ito_signature import (
    bench_ito_signature,
)
from quant_fund.models.lyons_extension import (
    bench_lyons_extension,
)
from quant_fund.models.step_signature import (
    bench_step_signature,
)
from quant_fund.models.tame_map import bench_tame_map
from quant_fund.models.young_integral import (
    bench_young_integral,
)


def test_ito_signature():
    assert bench_ito_signature()["synthetic_ito_signature"] == 1.0


def test_lyons_extension():
    assert bench_lyons_extension()["synthetic_lyons_extension"] == 1.0


def test_tame_map():
    assert bench_tame_map()["synthetic_tame_map"] == 1.0


def test_step_signature():
    assert bench_step_signature()["synthetic_step_signature"] == 1.0


def test_gubinelli_sewing():
    assert bench_gubinelli_sewing()["synthetic_gubinelli_sewing"] == 1.0


def test_young_integral():
    assert bench_young_integral()["synthetic_young_integral"] == 1.0
