import pytest

from quant_fund.research import benches_w1474


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bittern_qa_studies_family",
        "bench_cormorant_qa_studies_family",
        "bench_curlew_qa_studies_family",
        "bench_ibis_qa_studies_family",
        "bench_kingfisher_qa_studies_family",
        "bench_loon_qa_studies_family",
    ],
)
def test_benches_w1474(fam):
    out = getattr(benches_w1474, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
