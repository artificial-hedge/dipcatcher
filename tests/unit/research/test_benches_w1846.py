import pytest

from quant_fund.research import benches_w1846


@pytest.mark.parametrize(
    "fam",
    [
        "bench_almaqah_qa_studies_family",
        "bench_anbay_qa_studies_family",
        "bench_aranyada_qa_studies_family",
        "bench_athtar_qa_studies_family",
        "bench_haubas_qa_studies_family",
        "bench_nasr2_qa_studies_family",
    ],
)
def test_benches_w1846(fam):
    out = getattr(benches_w1846, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
