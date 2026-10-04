import pytest

from quant_fund.research import benches_w1526


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bracken_qa_studies_family",
        "bench_horsetail_qa_studies_family",
        "bench_maidenhair_qa_studies_family",
        "bench_staghorn_qa_studies_family",
        "bench_swordfern_qa_studies_family",
        "bench_treefern_qa_studies_family",
    ],
)
def test_benches_w1526(fam):
    out = getattr(benches_w1526, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
