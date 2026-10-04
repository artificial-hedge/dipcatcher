import pytest

from quant_fund.research import benches_w1522


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agave_qa_studies_family",
        "bench_aloe_qa_studies_family",
        "bench_echeveria_qa_studies_family",
        "bench_haworthia_qa_studies_family",
        "bench_lithops_qa_studies_family",
        "bench_sedum_qa_studies_family",
    ],
)
def test_benches_w1522(fam):
    out = getattr(benches_w1522, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
