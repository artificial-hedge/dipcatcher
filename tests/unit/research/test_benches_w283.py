"""Wave-283 adapter bench tests."""

from quant_fund.research.benches_w283 import (
    bench_bezier_curve_family,
    bench_fk_dh_family,
    bench_ik_jac_family,
    bench_odom_comp_family,
    bench_pot_field_family,
    bench_ray_lidar_family,
)

FAMS = [
    bench_fk_dh_family,
    bench_ik_jac_family,
    bench_ray_lidar_family,
    bench_pot_field_family,
    bench_bezier_curve_family,
    bench_odom_comp_family,
]


def test_wave283_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave283_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
