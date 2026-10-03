"""Wave-270 adapter bench tests."""

from quant_fund.research.benches_w270 import (
    bench_dmc_solver_family,
    bench_fdtd_wave_family,
    bench_ising_metro_family,
    bench_lattice_boltzmann_family,
    bench_lj_md_family,
    bench_pic_plasma_family,
)

FAMS = [
    bench_lj_md_family,
    bench_fdtd_wave_family,
    bench_lattice_boltzmann_family,
    bench_ising_metro_family,
    bench_pic_plasma_family,
    bench_dmc_solver_family,
]


def test_wave270_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave270_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
