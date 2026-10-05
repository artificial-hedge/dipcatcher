import pytest

from quant_fund.research import benches_w1943


@pytest.mark.parametrize(
    "fam",
    [
        "bench_genderuwo_qa_studies_family",
        "bench_jenglot_qa_studies_family",
        "bench_kuntilanak_qa_studies_family",
        "bench_leyak_qa_studies_family",
        "bench_pocong_qa_studies_family",
        "bench_tuyul_qa_studies_family",
    ],
)
def test_benches_w1943(fam):
    out = getattr(benches_w1943, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
