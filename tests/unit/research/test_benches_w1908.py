import pytest

from quant_fund.research import benches_w1908


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anhanga_qa_studies_family",
        "bench_bolotnik_qa_studies_family",
        "bench_dvorovoy_qa_studies_family",
        "bench_jurupari_qa_studies_family",
        "bench_lobisomem_qa_studies_family",
        "bench_mula_sem_cabeca_qa_studies_family",
    ],
)
def test_benches_w1908(fam):
    out = getattr(benches_w1908, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
