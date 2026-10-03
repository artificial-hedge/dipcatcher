from quant_fund.models.bates_model import (
    bench_bates_model,
)
from quant_fund.models.heston_model import (
    bench_heston_model,
)
from quant_fund.models.rough_heston import (
    bench_rough_heston,
)
from quant_fund.models.sabr_model import (
    bench_sabr_model,
)
from quant_fund.models.scott_vol import (
    bench_scott_vol,
)
from quant_fund.models.three_two_vol import (
    bench_three_two_vol,
)


def test_heston_model():
    assert bench_heston_model()["synthetic_heston_model"] == 1.0


def test_bates_model():
    assert bench_bates_model()["synthetic_bates_model"] == 1.0


def test_rough_heston():
    assert bench_rough_heston()["synthetic_rough_heston"] == 1.0


def test_sabr_model():
    assert bench_sabr_model()["synthetic_sabr_model"] == 1.0


def test_three_two_vol():
    assert bench_three_two_vol()["synthetic_three_two_vol"] == 1.0


def test_scott_vol():
    assert bench_scott_vol()["synthetic_scott_vol"] == 1.0
