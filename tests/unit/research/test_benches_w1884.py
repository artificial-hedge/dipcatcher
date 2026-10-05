import pytest

from quant_fund.research import benches_w1884


@pytest.mark.parametrize(
    "fam",
    [
        "bench_argemm_qa_studies_family",
        "bench_arzew_qa_studies_family",
        "bench_cilteni_qa_studies_family",
        "bench_essuf_qa_studies_family",
        "bench_medghassen_qa_studies_family",
        "bench_tanezruft_qa_studies_family",
    ],
)
def test_benches_w1884(fam):
    out = getattr(benches_w1884, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
