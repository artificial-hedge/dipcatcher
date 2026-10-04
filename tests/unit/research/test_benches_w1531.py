import pytest

from quant_fund.research import benches_w1531


@pytest.mark.parametrize(
    "fam",
    [
        "bench_downy_qa_studies_family",
        "bench_flicker_qa_studies_family",
        "bench_pileated_qa_studies_family",
        "bench_sapsucker_qa_studies_family",
        "bench_woodpecker_qa_studies_family",
        "bench_wryneck_qa_studies_family",
    ],
)
def test_benches_w1531(fam):
    out = getattr(benches_w1531, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
