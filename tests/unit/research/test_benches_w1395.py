import pytest

from quant_fund.research import benches_w1395


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bamboogle_studies_family",
        "bench_fine_qa_studies_family",
        "bench_hotpot2_studies_family",
        "bench_kwik_qa_studies_family",
        "bench_quest_qa_studies_family",
        "bench_tatqa2_studies_family",
    ],
)
def test_benches_w1395(fam):
    out = getattr(benches_w1395, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
