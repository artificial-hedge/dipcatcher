import pytest

from quant_fund.research import benches_w1348


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anli_r1_studies_family",
        "bench_anli_r2_studies_family",
        "bench_anli_r3_studies_family",
        "bench_mnli_match_studies_family",
        "bench_scitail_lite_studies_family",
        "bench_snli_lite_studies_family",
    ],
)
def test_benches_w1348(fam):
    out = getattr(benches_w1348, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
