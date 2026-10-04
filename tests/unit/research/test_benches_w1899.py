import pytest

from quant_fund.research import benches_w1899


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aghasura_qa_studies_family",
        "bench_bakasura_qa_studies_family",
        "bench_daitya_qa_studies_family",
        "bench_diti_qa_studies_family",
        "bench_pishacha_qa_studies_family",
        "bench_putana_qa_studies_family",
    ],
)
def test_benches_w1899(fam):
    out = getattr(benches_w1899, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
