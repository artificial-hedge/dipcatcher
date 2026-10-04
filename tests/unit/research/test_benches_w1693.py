import pytest

from quant_fund.research import benches_w1693


@pytest.mark.parametrize(
    "fam",
    [
        "bench_asag_qa_studies_family",
        "bench_edimmu_qa_studies_family",
        "bench_galla_qa_studies_family",
        "bench_lamassu_qa_studies_family",
        "bench_shedu_qa_studies_family",
        "bench_utukku_qa_studies_family",
    ],
)
def test_benches_w1693(fam):
    out = getattr(benches_w1693, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
