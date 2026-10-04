from quant_fund.models.compound_poisson import (
    bench_compound_poisson,
)
from quant_fund.models.excursion_theory import (
    bench_excursion_theory,
)
from quant_fund.models.jump_diffusion import (
    bench_jump_diffusion,
)
from quant_fund.models.kou_model import (
    bench_kou_model,
)
from quant_fund.models.marked_hawkes import (
    bench_marked_hawkes,
)
from quant_fund.models.merton_jump import (
    bench_merton_jump,
)


def test_jump_diffusion():
    assert bench_jump_diffusion()["synthetic_jump_diffusion"] == 1.0


def test_merton_jump():
    assert bench_merton_jump()["synthetic_merton_jump"] == 1.0


def test_kou_model():
    assert bench_kou_model()["synthetic_kou_model"] == 1.0


def test_compound_poisson():
    assert bench_compound_poisson()["synthetic_compound_poisson"] == 1.0


def test_excursion_theory():
    assert bench_excursion_theory()["synthetic_excursion_theory"] == 1.0


def test_marked_hawkes():
    assert bench_marked_hawkes()["synthetic_marked_hawkes"] == 1.0
