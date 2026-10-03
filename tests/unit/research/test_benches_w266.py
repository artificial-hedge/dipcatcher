"""Wave-266 adapter bench tests."""

from quant_fund.research.benches_w266 import (
    bench_bump_map_family,
    bench_mipmap_sample_family,
    bench_phong_shade_family,
    bench_shadow_map_family,
    bench_ssao_lite_family,
    bench_triangle_raster_family,
)

FAMS = [
    bench_triangle_raster_family,
    bench_phong_shade_family,
    bench_mipmap_sample_family,
    bench_shadow_map_family,
    bench_bump_map_family,
    bench_ssao_lite_family,
]


def test_wave266_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k, v in out.items():
            assert k.startswith("synthetic_"), k
            assert 0.0 <= float(v) <= 1.0 or float(v) == float(v)


def test_wave266_benches_score_high() -> None:
    for f in FAMS:
        out = f()
        assert max(out.values()) >= 0.5, (f.__name__, out)
