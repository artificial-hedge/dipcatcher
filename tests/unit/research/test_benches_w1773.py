import pytest

from quant_fund.research import benches_w1773


@pytest.mark.parametrize(
    "fam",
    [
        "bench_balder_qa_studies_family",
        "bench_frigg_qa_studies_family",
        "bench_loki_qa_studies_family",
        "bench_sif_qa_studies_family",
        "bench_vali_qa_studies_family",
        "bench_vitharr_qa_studies_family",
    ],
)
def test_benches_w1773(fam):
    out = getattr(benches_w1773, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
