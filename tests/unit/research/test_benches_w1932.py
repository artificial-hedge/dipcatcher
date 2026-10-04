import pytest

from quant_fund.research import benches_w1932


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ah_puch_qa_studies_family",
        "bench_alux_qa_studies_family",
        "bench_cizin_qa_studies_family",
        "bench_nahualli_qa_studies_family",
        "bench_vucub_qa_studies_family",
        "bench_xtabay_qa_studies_family",
    ],
)
def test_benches_w1932(fam):
    out = getattr(benches_w1932, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
