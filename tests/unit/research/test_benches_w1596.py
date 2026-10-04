import pytest

from quant_fund.research import benches_w1596


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bottlenose_qa_studies_family",
        "bench_dusky_dolphin_qa_studies_family",
        "bench_false_killer_qa_studies_family",
        "bench_melon_head_qa_studies_family",
        "bench_pygmy_whale_qa_studies_family",
        "bench_sea_lion_qa_studies_family",
    ],
)
def test_benches_w1596(fam):
    out = getattr(benches_w1596, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
