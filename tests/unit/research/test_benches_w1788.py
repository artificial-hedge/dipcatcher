import pytest

from quant_fund.research import benches_w1788


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ceres_qa_studies_family",
        "bench_flora_qa_studies_family",
        "bench_janus_qa_studies_family",
        "bench_pomona_qa_studies_family",
        "bench_silvanus_qa_studies_family",
        "bench_solinvictus_qa_studies_family",
    ],
)
def test_benches_w1788(fam):
    out = getattr(benches_w1788, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
