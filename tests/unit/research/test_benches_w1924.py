import pytest

from quant_fund.research import benches_w1924


@pytest.mark.parametrize(
    "fam",
    [
        "bench_achiyay_qa_studies_family",
        "bench_cayt_qa_studies_family",
        "bench_emegen_qa_studies_family",
        "bench_maymene_qa_studies_family",
        "bench_ubir_qa_studies_family",
        "bench_uor_qa_studies_family",
    ],
)
def test_benches_w1924(fam):
    out = getattr(benches_w1924, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
