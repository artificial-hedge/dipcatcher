"""Wave-281 adapter bench tests."""

from quant_fund.research.benches_w281 import (
    bench_galois_field_family,
    bench_group_table_family,
    bench_ideal_member_family,
    bench_matrix_grp_family,
    bench_perm_group_family,
    bench_poly_ring_family,
)

FAMS = [
    bench_group_table_family,
    bench_perm_group_family,
    bench_galois_field_family,
    bench_poly_ring_family,
    bench_ideal_member_family,
    bench_matrix_grp_family,
]


def test_wave281_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave281_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
