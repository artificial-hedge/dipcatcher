import pytest

from quant_fund.research import benches_w1749


@pytest.mark.parametrize(
    "fam",
    [
        "bench_belobog_qa_studies_family",
        "bench_chernobog_qa_studies_family",
        "bench_dazhbog_qa_studies_family",
        "bench_hors_qa_studies_family",
        "bench_semargl_qa_studies_family",
        "bench_stribog_qa_studies_family",
    ],
)
def test_benches_w1749(fam):
    out = getattr(benches_w1749, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
