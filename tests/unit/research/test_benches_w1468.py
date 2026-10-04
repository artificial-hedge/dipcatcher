import pytest

from quant_fund.research import benches_w1468


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dale_qa_studies_family",
        "bench_fen_qa_studies_family",
        "bench_glen_qa_studies_family",
        "bench_heath_qa_studies_family",
        "bench_knoll_qa_studies_family",
        "bench_moor_qa_studies_family",
    ],
)
def test_benches_w1468(fam):
    out = getattr(benches_w1468, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
