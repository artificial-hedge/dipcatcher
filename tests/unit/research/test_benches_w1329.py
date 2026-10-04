import pytest

from quant_fund.research import benches_w1329


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gov_report_studies_family",
        "bench_looogle_studies_family",
        "bench_lost_middle_studies_family",
        "bench_marathon_eval_studies_family",
        "bench_niah_v2_studies_family",
        "bench_passkey_retrieval_studies_family",
    ],
)
def test_benches_w1329(fam):
    out = getattr(benches_w1329, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
