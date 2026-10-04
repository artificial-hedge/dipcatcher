from quant_fund.models.cutoff_phenomenon import (
    bench_cutoff_phenomenon,
)
from quant_fund.models.doeblin_coupling import (
    bench_doeblin_coupling,
)
from quant_fund.models.drift_lyapunov import (
    bench_drift_lyapunov,
)
from quant_fund.models.ergodic_markov import (
    bench_ergodic_markov,
)
from quant_fund.models.harris_recurrent import (
    bench_harris_recurrent,
)
from quant_fund.models.mixing_time import (
    bench_mixing_time,
)


def test_doeblin_coupling():
    assert bench_doeblin_coupling()["synthetic_doeblin_coupling"] == 1.0


def test_harris_recurrent():
    assert bench_harris_recurrent()["synthetic_harris_recurrent"] == 1.0


def test_ergodic_markov():
    assert bench_ergodic_markov()["synthetic_ergodic_markov"] == 1.0


def test_mixing_time():
    assert bench_mixing_time()["synthetic_mixing_time"] == 1.0


def test_drift_lyapunov():
    assert bench_drift_lyapunov()["synthetic_drift_lyapunov"] == 1.0


def test_cutoff_phenomenon():
    assert bench_cutoff_phenomenon()["synthetic_cutoff_phenomenon"] == 1.0
