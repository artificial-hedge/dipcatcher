from quant_fund.models.latent_sde import (
    bench_latent_sde,
)
from quant_fund.models.logsig_rde import (
    bench_logsig_rde,
)
from quant_fund.models.neural_cde import (
    bench_neural_cde,
)
from quant_fund.models.neural_rde import (
    bench_neural_rde,
)
from quant_fund.models.sde_gan import (
    bench_sde_gan,
)
from quant_fund.models.sde_matching import (
    bench_sde_matching,
)


def test_latent_sde():
    assert bench_latent_sde()["synthetic_latent_sde"] == 1.0


def test_neural_cde():
    assert bench_neural_cde()["synthetic_neural_cde"] == 1.0


def test_neural_rde():
    assert bench_neural_rde()["synthetic_neural_rde"] == 1.0


def test_sde_gan():
    assert bench_sde_gan()["synthetic_sde_gan"] == 1.0


def test_sde_matching():
    assert bench_sde_matching()["synthetic_sde_matching"] == 1.0


def test_logsig_rde():
    assert bench_logsig_rde()["synthetic_logsig_rde"] == 1.0
