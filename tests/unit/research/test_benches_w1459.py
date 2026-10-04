import pytest

from quant_fund.research import benches_w1459


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arctic_fox_qa_studies_family",
        "bench_caribou_qa_studies_family",
        "bench_musk_ox_qa_studies_family",
        "bench_penguin_qa_studies_family",
        "bench_polar_bear_qa_studies_family",
        "bench_reindeer_qa_studies_family",
    ],
)
def test_benches_w1459(fam):
    out = getattr(benches_w1459, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
