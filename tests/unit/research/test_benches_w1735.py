import pytest

from quant_fund.research import benches_w1735


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dievas_qa_studies_family",
        "bench_gabija_qa_studies_family",
        "bench_medeina_qa_studies_family",
        "bench_ragana_qa_studies_family",
        "bench_saulute_qa_studies_family",
        "bench_velnias_qa_studies_family",
    ],
)
def test_benches_w1735(fam):
    out = getattr(benches_w1735, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
