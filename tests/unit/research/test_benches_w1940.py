import pytest

from quant_fund.research import benches_w1940


@pytest.mark.parametrize(
    "fam",
    [
        "bench_magami_qa_studies_family",
        "bench_mahagiri_qa_studies_family",
        "bench_min_kyawzwa_qa_studies_family",
        "bench_shwe_nabay_qa_studies_family",
        "bench_taungmagyi_qa_studies_family",
        "bench_thagya_min_qa_studies_family",
    ],
)
def test_benches_w1940(fam):
    out = getattr(benches_w1940, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
