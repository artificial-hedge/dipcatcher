import pytest

from quant_fund.research import benches_w1661


@pytest.mark.parametrize(
    "fam",
    [
        "bench_draugr_qa_studies_family",
        "bench_fenrir_qa_studies_family",
        "bench_gullinbursti_qa_studies_family",
        "bench_hraesvelgr_qa_studies_family",
        "bench_huginn_qa_studies_family",
        "bench_muninn_qa_studies_family",
    ],
)
def test_benches_w1661(fam):
    out = getattr(benches_w1661, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
