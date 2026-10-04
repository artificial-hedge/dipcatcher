import pytest

from quant_fund.research import benches_w1637


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abura_sumashi_qa_studies_family",
        "bench_azukiarai_qa_studies_family",
        "bench_betobeto_2_qa_studies_family",
        "bench_futakuchi_qa_studies_family",
        "bench_rokurokubi_qa_studies_family",
        "bench_shirime_qa_studies_family",
    ],
)
def test_benches_w1637(fam):
    out = getattr(benches_w1637, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
