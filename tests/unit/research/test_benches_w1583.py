import pytest

from quant_fund.research import benches_w1583


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bowhead_qa_studies_family",
        "bench_fin_whale_qa_studies_family",
        "bench_humpback_qa_studies_family",
        "bench_minke_qa_studies_family",
        "bench_pilot_whale_qa_studies_family",
        "bench_sperm_whale_qa_studies_family",
    ],
)
def test_benches_w1583(fam):
    out = getattr(benches_w1583, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
