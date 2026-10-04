import pytest

from quant_fund.research import benches_w1747


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bragi_qa_studies_family",
        "bench_forseti_qa_studies_family",
        "bench_idun_qa_studies_family",
        "bench_sif_qa_studies_family",
        "bench_ullr_qa_studies_family",
        "bench_vidar_qa_studies_family",
    ],
)
def test_benches_w1747(fam):
    out = getattr(benches_w1747, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
