import pytest

from quant_fund.research import benches_w1950


@pytest.mark.parametrize(
    "fam",
    [
        "bench_beleth_qa_studies_family",
        "bench_botis_qa_studies_family",
        "bench_eligos_qa_studies_family",
        "bench_leraje_qa_studies_family",
        "bench_sitri_qa_studies_family",
        "bench_zepar_qa_studies_family",
    ],
)
def test_benches_w1950(fam):
    out = getattr(benches_w1950, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
