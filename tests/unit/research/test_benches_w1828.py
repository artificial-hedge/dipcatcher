import pytest

from quant_fund.research import benches_w1828


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ajatar2_qa_studies_family",
        "bench_ilmatar2_qa_studies_family",
        "bench_jumala2_qa_studies_family",
        "bench_metsanhiisi2_qa_studies_family",
        "bench_otso2_qa_studies_family",
        "bench_peikko2_qa_studies_family",
    ],
)
def test_benches_w1828(fam):
    out = getattr(benches_w1828, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
