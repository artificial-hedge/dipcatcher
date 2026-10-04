import pytest

from quant_fund.research import benches_w1520


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agaric_qa_studies_family",
        "bench_bolete_qa_studies_family",
        "bench_chanterelle_qa_studies_family",
        "bench_inkcap_qa_studies_family",
        "bench_morel_qa_studies_family",
        "bench_puffball_qa_studies_family",
    ],
)
def test_benches_w1520(fam):
    out = getattr(benches_w1520, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
