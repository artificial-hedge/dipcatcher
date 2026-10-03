"""Wave-269 adapter bench tests."""

from quant_fund.research.benches_w269 import (
    bench_epipolar_8pt_family,
    bench_homography_4pt_family,
    bench_lk_flow_family,
    bench_orb_feature_family,
    bench_ransac_plane_family,
    bench_stereo_disparity_family,
)

FAMS = [
    bench_lk_flow_family,
    bench_orb_feature_family,
    bench_homography_4pt_family,
    bench_ransac_plane_family,
    bench_epipolar_8pt_family,
    bench_stereo_disparity_family,
]


def test_wave269_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave269_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
