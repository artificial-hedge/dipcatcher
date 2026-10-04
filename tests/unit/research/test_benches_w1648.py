import pytest

from quant_fund.research import benches_w1648


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aitvaras_qa_studies_family",
        "bench_bilwis_qa_studies_family",
        "bench_indus_qa_studies_family",
        "bench_kudlak_qa_studies_family",
        "bench_viy_qa_studies_family",
        "bench_zilant_qa_studies_family",
    ],
)
def test_benches_w1648(fam):
    out = getattr(benches_w1648, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
