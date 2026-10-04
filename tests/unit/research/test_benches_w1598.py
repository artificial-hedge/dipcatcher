import pytest

from quant_fund.research import benches_w1598


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bontebok_qa_studies_family",
        "bench_bushbuck_qa_studies_family",
        "bench_greater_kudu_qa_studies_family",
        "bench_lesser_kudu_qa_studies_family",
        "bench_mountain_nyala_qa_studies_family",
        "bench_sitatunga_qa_studies_family",
    ],
)
def test_benches_w1598(fam):
    out = getattr(benches_w1598, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
