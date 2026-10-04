import pytest

from quant_fund.research import benches_w1347


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abductive_nli_studies_family",
        "bench_conseq_log_studies_family",
        "bench_logiqa_log_studies_family",
        "bench_lsat_log_studies_family",
        "bench_reason_mc_studies_family",
        "bench_recli_log_studies_family",
    ],
)
def test_benches_w1347(fam):
    out = getattr(benches_w1347, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
