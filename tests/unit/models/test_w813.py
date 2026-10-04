from quant_fund.models.diffusion_semigroup import (
    bench_diffusion_semigroup,
)
from quant_fund.models.feller_boundary import (
    bench_feller_boundary,
)
from quant_fund.models.kreyn_resolvent import (
    bench_kreyn_resolvent,
)
from quant_fund.models.scale_measure import (
    bench_scale_measure,
)
from quant_fund.models.speed_measure import (
    bench_speed_measure,
)
from quant_fund.models.yosida_op import (
    bench_yosida_op,
)


def test_feller_boundary():
    assert bench_feller_boundary()["synthetic_feller_boundary"] == 1.0


def test_scale_measure():
    assert bench_scale_measure()["synthetic_scale_measure"] == 1.0


def test_speed_measure():
    assert bench_speed_measure()["synthetic_speed_measure"] == 1.0


def test_diffusion_semigroup():
    assert bench_diffusion_semigroup()["synthetic_diffusion_semigroup"] == 1.0


def test_yosida_op():
    assert bench_yosida_op()["synthetic_yosida_op"] == 1.0


def test_kreyn_resolvent():
    assert bench_kreyn_resolvent()["synthetic_kreyn_resolvent"] == 1.0
