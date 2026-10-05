import pytest

from quant_fund.research import benches_w1824


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boszorka2_qa_studies_family",
        "bench_liderec2_qa_studies_family",
        "bench_remete2_qa_studies_family",
        "bench_sarkany2_qa_studies_family",
        "bench_tatros2_qa_studies_family",
        "bench_turul2_qa_studies_family",
    ],
)
def test_benches_w1824(fam):
    out = getattr(benches_w1824, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
