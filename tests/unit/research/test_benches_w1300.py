import pytest

from quant_fund.research import benches_w1300


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adversarial_eval_studies_family",
        "bench_autoattack_studies_family",
        "bench_corruption_studies_family",
        "bench_imagenet_c_studies_family",
        "bench_imagenet_r_studies_family",
        "bench_robust_bench_studies_family",
    ],
)
def test_benches_w1300(fam):
    out = getattr(benches_w1300, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
