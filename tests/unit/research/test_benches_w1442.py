import pytest

from quant_fund.research import benches_w1442


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ant_qa_studies_family",
        "bench_bee_qa_studies_family",
        "bench_beetle_qa_studies_family",
        "bench_butterfly_qa_studies_family",
        "bench_cricket_qa_studies_family",
        "bench_moth_qa_studies_family",
    ],
)
def test_benches_w1442(fam):
    out = getattr(benches_w1442, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
