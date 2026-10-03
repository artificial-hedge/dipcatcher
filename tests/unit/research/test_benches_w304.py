"""Adapter tests for wave-304 computer-vision-2 canon benches."""

from quant_fund.research.benches_w304 import (
    bench_grabcut_lite_family,
    bench_harris_corner_family,
    bench_hough_lines_family,
    bench_integral_image_family,
    bench_meanshift_track_family,
    bench_seam_carving_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_harris_corner_family,
        bench_hough_lines_family,
        bench_integral_image_family,
        bench_seam_carving_family,
        bench_grabcut_lite_family,
        bench_meanshift_track_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
