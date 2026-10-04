import pytest

from quant_fund.research import benches_w1856


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cernunnos_qa_studies_family",
        "bench_epona_qa_studies_family",
        "bench_esus_qa_studies_family",
        "bench_rosmerta_qa_studies_family",
        "bench_taranis_qa_studies_family",
        "bench_teutates_qa_studies_family",
    ],
)
def test_benches_w1856(fam):
    out = getattr(benches_w1856, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
