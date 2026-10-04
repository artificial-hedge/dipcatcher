import pytest

from quant_fund.research import benches_w1668


@pytest.mark.parametrize(
    "fam",
    [
        "bench_berserkr_qa_studies_family",
        "bench_fafnir_qa_studies_family",
        "bench_jotun_qa_studies_family",
        "bench_regin_qa_studies_family",
        "bench_ulfhednar_qa_studies_family",
        "bench_vargr_qa_studies_family",
    ],
)
def test_benches_w1668(fam):
    out = getattr(benches_w1668, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
