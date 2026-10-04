import pytest

from quant_fund.research import benches_w1616


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amphipod_qa_studies_family",
        "bench_barnacle_qa_studies_family",
        "bench_copepod_qa_studies_family",
        "bench_isopod_qa_studies_family",
        "bench_krill_qa_studies_family",
        "bench_sandhopper_qa_studies_family",
    ],
)
def test_benches_w1616(fam):
    out = getattr(benches_w1616, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
