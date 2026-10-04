import pytest

from quant_fund.research import benches_w1766


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cabrakan_qa_studies_family",
        "bench_camazotz_qa_studies_family",
        "bench_hunab_qa_studies_family",
        "bench_itzamna_qa_studies_family",
        "bench_ixmucane_qa_studies_family",
        "bench_zipacna_qa_studies_family",
    ],
)
def test_benches_w1766(fam):
    out = getattr(benches_w1766, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
