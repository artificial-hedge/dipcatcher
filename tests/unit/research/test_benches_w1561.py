import pytest

from quant_fund.research import benches_w1561


@pytest.mark.parametrize(
    "fam",
    [
        "bench_angelfish_qa_studies_family",
        "bench_blenny_qa_studies_family",
        "bench_goby_qa_studies_family",
        "bench_lionfish_qa_studies_family",
        "bench_surgeonfish_qa_studies_family",
        "bench_triggerfish_qa_studies_family",
    ],
)
def test_benches_w1561(fam):
    out = getattr(benches_w1561, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
