import pytest

from quant_fund.research import benches_w1769


@pytest.mark.parametrize(
    "fam",
    [
        "bench_angra_qa_studies_family",
        "bench_arash_qa_studies_family",
        "bench_haoma_qa_studies_family",
        "bench_simurgh_qa_studies_family",
        "bench_spenta_qa_studies_family",
        "bench_zal_qa_studies_family",
    ],
)
def test_benches_w1769(fam):
    out = getattr(benches_w1769, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
