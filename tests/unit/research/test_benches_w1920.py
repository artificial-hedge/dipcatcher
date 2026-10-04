import pytest

from quant_fund.research import benches_w1920


@pytest.mark.parametrize(
    "fam",
    [
        "bench_iele_qa_studies_family",
        "bench_moroi_qa_studies_family",
        "bench_pricolici_qa_studies_family",
        "bench_samca_qa_studies_family",
        "bench_strigoi_qa_studies_family",
        "bench_varcolac_qa_studies_family",
    ],
)
def test_benches_w1920(fam):
    out = getattr(benches_w1920, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
