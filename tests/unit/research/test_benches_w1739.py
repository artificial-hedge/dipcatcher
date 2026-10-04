import pytest

from quant_fund.research import benches_w1739


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ahura_mazda_qa_studies_family",
        "bench_anahita_qa_studies_family",
        "bench_angra_mainyu_qa_studies_family",
        "bench_mithra_qa_studies_family",
        "bench_verethragna_qa_studies_family",
        "bench_zahhak_qa_studies_family",
    ],
)
def test_benches_w1739(fam):
    out = getattr(benches_w1739, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
