from quant_fund.models.caratheodory_thm import (
    bench_caratheodory_thm,
)
from quant_fund.models.farkas_lemma import (
    bench_farkas_lemma,
)
from quant_fund.models.lattice_point import (
    bench_lattice_point,
)
from quant_fund.models.radon_theorem import (
    bench_radon_theorem,
)
from quant_fund.models.separation_thm import (
    bench_separation_thm,
)
from quant_fund.models.tverberg_thm import (
    bench_tverberg_thm,
)


def test_radon_theorem():
    assert bench_radon_theorem()["synthetic_radon_theorem"] == 1.0


def test_caratheodory_thm():
    assert bench_caratheodory_thm()["synthetic_caratheodory_thm"] == 1.0


def test_farkas_lemma():
    assert bench_farkas_lemma()["synthetic_farkas_lemma"] == 1.0


def test_separation_thm():
    assert bench_separation_thm()["synthetic_separation_thm"] == 1.0


def test_lattice_point():
    assert bench_lattice_point()["synthetic_lattice_point"] == 1.0


def test_tverberg_thm():
    assert bench_tverberg_thm()["synthetic_tverberg_thm"] == 1.0
