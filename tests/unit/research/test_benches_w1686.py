import pytest

from quant_fund.research import benches_w1686


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abada_qa_studies_family",
        "bench_adze_qa_studies_family",
        "bench_ilomba_qa_studies_family",
        "bench_nbanda_qa_studies_family",
        "bench_ninki_qa_studies_family",
        "bench_okubi_qa_studies_family",
    ],
)
def test_benches_w1686(fam):
    out = getattr(benches_w1686, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
