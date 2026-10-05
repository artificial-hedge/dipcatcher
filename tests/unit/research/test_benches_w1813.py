import pytest

from quant_fund.research import benches_w1813


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baldr2_qa_studies_family",
        "bench_frigg2_qa_studies_family",
        "bench_idun2_qa_studies_family",
        "bench_njord2_qa_studies_family",
        "bench_sif2_qa_studies_family",
        "bench_tyr2_qa_studies_family",
    ],
)
def test_benches_w1813(fam):
    out = getattr(benches_w1813, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
