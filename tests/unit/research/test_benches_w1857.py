import pytest

from quant_fund.research import benches_w1857


@pytest.mark.parametrize(
    "fam",
    [
        "bench_belatucadrus_qa_studies_family",
        "bench_cocidius_qa_studies_family",
        "bench_maponus_qa_studies_family",
        "bench_nemetona_qa_studies_family",
        "bench_rigisamus_qa_studies_family",
        "bench_sulis_qa_studies_family",
    ],
)
def test_benches_w1857(fam):
    out = getattr(benches_w1857, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
