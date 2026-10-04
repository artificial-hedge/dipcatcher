import pytest

from quant_fund.research import benches_w1496


@pytest.mark.parametrize(
    "fam",
    [
        "bench_jacana_qa_studies_family",
        "bench_lapwing_qa_studies_family",
        "bench_moorhen_qa_studies_family",
        "bench_railbird_qa_studies_family",
        "bench_snipe_qa_studies_family",
        "bench_turnstone_qa_studies_family",
    ],
)
def test_benches_w1496(fam):
    out = getattr(benches_w1496, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
