import pytest

from quant_fund.research import benches_w1313


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abs_scan_studies_family",
        "bench_activation_cluster_studies_family",
        "bench_fine_pruning_studies_family",
        "bench_sleepless_studies_family",
        "bench_strip_defense_studies_family",
        "bench_watermark_studies_family",
    ],
)
def test_benches_w1313(fam):
    out = getattr(benches_w1313, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
