import pytest

from quant_fund.research import benches_w1597


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bharal_qa_studies_family",
        "bench_chamois_qa_studies_family",
        "bench_goral_qa_studies_family",
        "bench_ibex_qa_studies_family",
        "bench_serow_qa_studies_family",
        "bench_tahr_qa_studies_family",
    ],
)
def test_benches_w1597(fam):
    out = getattr(benches_w1597, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
