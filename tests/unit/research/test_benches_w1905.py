import pytest

from quant_fund.research import benches_w1905


@pytest.mark.parametrize(
    "fam",
    [
        "bench_betobeto_qa_studies_family",
        "bench_buruburu_qa_studies_family",
        "bench_hyakume_qa_studies_family",
        "bench_shachihoko_qa_studies_family",
        "bench_uwan_qa_studies_family",
        "bench_waira_qa_studies_family",
    ],
)
def test_benches_w1905(fam):
    out = getattr(benches_w1905, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
