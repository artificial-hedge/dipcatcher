import pytest

from quant_fund.research import benches_w1537


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coot_qa_studies_family",
        "bench_crake_qa_studies_family",
        "bench_dabchick_qa_studies_family",
        "bench_gallinule_qa_studies_family",
        "bench_rail_qa_studies_family",
        "bench_waterhen_qa_studies_family",
    ],
)
def test_benches_w1537(fam):
    out = getattr(benches_w1537, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
