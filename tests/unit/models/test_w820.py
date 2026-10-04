from quant_fund.models.canonical_decomp import (
    bench_canonical_decomp,
)
from quant_fund.models.doom_decomp import (
    bench_doom_decomp,
)
from quant_fund.models.pcdt import (
    bench_pcdt,
)
from quant_fund.models.sem_loc_char import (
    bench_sem_loc_char,
)
from quant_fund.models.special_sem import (
    bench_special_sem,
)
from quant_fund.models.triplet_char import (
    bench_triplet_char,
)


def test_doom_decomp():
    assert bench_doom_decomp()["synthetic_doom_decomp"] == 1.0


def test_pcdt():
    assert bench_pcdt()["synthetic_pcdt"] == 1.0


def test_special_sem():
    assert bench_special_sem()["synthetic_special_sem"] == 1.0


def test_canonical_decomp():
    assert bench_canonical_decomp()["synthetic_canonical_decomp"] == 1.0


def test_sem_loc_char():
    assert bench_sem_loc_char()["synthetic_sem_loc_char"] == 1.0


def test_triplet_char():
    assert bench_triplet_char()["synthetic_triplet_char"] == 1.0
