from quant_fund.models.kac_theorem import (
    bench_kac_theorem,
)
from quant_fund.models.mckean_vlasov import (
    bench_mckean_vlasov,
)
from quant_fund.models.mean_field_game2 import (
    bench_mean_field_game2,
)
from quant_fund.models.nonlinear_markov import (
    bench_nonlinear_markov,
)
from quant_fund.models.propagation_chaos import (
    bench_propagation_chaos,
)
from quant_fund.models.self_stabilizing import (
    bench_self_stabilizing,
)


def test_mckean_vlasov():
    assert bench_mckean_vlasov()["synthetic_mckean_vlasov"] == 1.0


def test_mean_field_game2():
    assert bench_mean_field_game2()["synthetic_mean_field_game2"] == 1.0


def test_propagation_chaos():
    assert bench_propagation_chaos()["synthetic_propagation_chaos"] == 1.0


def test_kac_theorem():
    assert bench_kac_theorem()["synthetic_kac_theorem"] == 1.0


def test_nonlinear_markov():
    assert bench_nonlinear_markov()["synthetic_nonlinear_markov"] == 1.0


def test_self_stabilizing():
    assert bench_self_stabilizing()["synthetic_self_stabilizing"] == 1.0
