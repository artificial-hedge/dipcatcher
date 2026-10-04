import pytest

from quant_fund.research import benches_w1829


@pytest.mark.parametrize(
    "fam",
    [
        "bench_iljang2_qa_studies_family",
        "bench_nemlert2_qa_studies_family",
        "bench_numgum2_qa_studies_family",
        "bench_otysi2_qa_studies_family",
        "bench_parnae2_qa_studies_family",
        "bench_xiberi2_qa_studies_family",
    ],
)
def test_benches_w1829(fam):
    out = getattr(benches_w1829, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
