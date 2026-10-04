import pytest

from quant_fund.research import benches_w1580


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chital_qa_studies_family",
        "bench_fallow_qa_studies_family",
        "bench_muntjac_qa_studies_family",
        "bench_pudu_qa_studies_family",
        "bench_roe_qa_studies_family",
        "bench_sika_qa_studies_family",
    ],
)
def test_benches_w1580(fam):
    out = getattr(benches_w1580, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
