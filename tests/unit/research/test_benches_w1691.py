import pytest

from quant_fund.research import benches_w1691


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abzu_qa_studies_family",
        "bench_enki_qa_studies_family",
        "bench_enlil_qa_studies_family",
        "bench_nanna_qa_studies_family",
        "bench_tiamat_qa_studies_family",
        "bench_utu_qa_studies_family",
    ],
)
def test_benches_w1691(fam):
    out = getattr(benches_w1691, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
