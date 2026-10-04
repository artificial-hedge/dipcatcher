import pytest

from quant_fund.research import benches_w1361


@pytest.mark.parametrize(
    "fam",
    [
        "bench_covid_lies_studies_family",
        "bench_evidence_inf_studies_family",
        "bench_hoax_detect_studies_family",
        "bench_liar_lite_studies_family",
        "bench_rumor_eval_studies_family",
        "bench_scidtb_lite_studies_family",
    ],
)
def test_benches_w1361(fam):
    out = getattr(benches_w1361, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
