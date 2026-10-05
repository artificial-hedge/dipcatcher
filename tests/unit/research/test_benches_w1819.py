import pytest

from quant_fund.research import benches_w1819


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bragi2_qa_studies_family",
        "bench_forseti2_qa_studies_family",
        "bench_heimdall2_qa_studies_family",
        "bench_norna2_qa_studies_family",
        "bench_ve2_qa_studies_family",
        "bench_vili2_qa_studies_family",
    ],
)
def test_benches_w1819(fam):
    out = getattr(benches_w1819, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
