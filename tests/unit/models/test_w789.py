from quant_fund.models.benamou_brenier import (
    bench_benamou_brenier,
)
from quant_fund.models.entropy_regular import (
    bench_entropy_regular,
)
from quant_fund.models.fokker_planck2 import (
    bench_fokker_planck2,
)
from quant_fund.models.gradient_flow import (
    bench_gradient_flow,
)
from quant_fund.models.jko_step import bench_jko_step
from quant_fund.models.wasserstein_grad import (
    bench_wasserstein_grad,
)


def test_wasserstein_grad():
    assert bench_wasserstein_grad()["synthetic_wasserstein_grad"] == 1.0


def test_jko_step():
    assert bench_jko_step()["synthetic_jko_step"] == 1.0


def test_benamou_brenier():
    assert bench_benamou_brenier()["synthetic_benamou_brenier"] == 1.0


def test_entropy_regular():
    assert bench_entropy_regular()["synthetic_entropy_regular"] == 1.0


def test_fokker_planck2():
    assert bench_fokker_planck2()["synthetic_fokker_planck2"] == 1.0


def test_gradient_flow():
    assert bench_gradient_flow()["synthetic_gradient_flow"] == 1.0
