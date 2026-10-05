import pytest

from quant_fund.research import benches_w1953


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amy_qa_studies_family",
        "bench_andrealphus_qa_studies_family",
        "bench_gremory_qa_studies_family",
        "bench_murmur_qa_studies_family",
        "bench_orobas_qa_studies_family",
        "bench_ose_qa_studies_family",
    ],
)
def test_benches_w1953(fam):
    out = getattr(benches_w1953, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
