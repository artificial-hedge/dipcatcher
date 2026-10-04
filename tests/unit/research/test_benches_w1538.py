import pytest

from quant_fund.research import benches_w1538


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crowned_crane_qa_studies_family",
        "bench_demoiselle_qa_studies_family",
        "bench_finfoot_qa_studies_family",
        "bench_limpkin_qa_studies_family",
        "bench_trumpeter_qa_studies_family",
        "bench_whooping_qa_studies_family",
    ],
)
def test_benches_w1538(fam):
    out = getattr(benches_w1538, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
