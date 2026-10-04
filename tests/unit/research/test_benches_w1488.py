import pytest

from quant_fund.research import benches_w1488


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bobcat_qa_studies_family",
        "bench_dingo_qa_studies_family",
        "bench_kodkod_qa_studies_family",
        "bench_oncilla_qa_studies_family",
        "bench_panther_qa_studies_family",
        "bench_tiger_qa_studies_family",
    ],
)
def test_benches_w1488(fam):
    out = getattr(benches_w1488, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
