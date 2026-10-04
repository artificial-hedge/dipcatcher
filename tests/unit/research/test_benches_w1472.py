import pytest

from quant_fund.research import benches_w1472


@pytest.mark.parametrize(
    "fam",
    [
        "bench_basil_qa_studies_family",
        "bench_cardamom_qa_studies_family",
        "bench_chervil_qa_studies_family",
        "bench_cinnamon_qa_studies_family",
        "bench_coriander_qa_studies_family",
        "bench_cumin_qa_studies_family",
    ],
)
def test_benches_w1472(fam):
    out = getattr(benches_w1472, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
