import pytest

from quant_fund.research import benches_w1441


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crane_qa_studies_family",
        "bench_eagle_qa_studies_family",
        "bench_falcon_qa_studies_family",
        "bench_owl_qa_studies_family",
        "bench_raven_qa_studies_family",
        "bench_swan_qa_studies_family",
    ],
)
def test_benches_w1441(fam):
    out = getattr(benches_w1441, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
