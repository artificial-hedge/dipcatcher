import pytest

from quant_fund.research import benches_w1312


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bio_risk_eval_studies_family",
        "bench_chem_risk_eval_studies_family",
        "bench_cyber_sec_eval_studies_family",
        "bench_lab_bench_studies_family",
        "bench_malicious_instruct_studies_family",
        "bench_wmdp_studies_family",
    ],
)
def test_benches_w1312(fam):
    out = getattr(benches_w1312, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
