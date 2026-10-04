import pytest

from quant_fund.research import benches_w1517


@pytest.mark.parametrize(
    "fam",
    [
        "bench_clubtail_qa_studies_family",
        "bench_damselfly_qa_studies_family",
        "bench_darner_qa_studies_family",
        "bench_forktail_qa_studies_family",
        "bench_hawker_qa_studies_family",
        "bench_spreadwing_qa_studies_family",
    ],
)
def test_benches_w1517(fam):
    out = getattr(benches_w1517, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
