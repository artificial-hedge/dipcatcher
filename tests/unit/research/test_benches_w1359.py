import pytest

from quant_fund.research import benches_w1359


@pytest.mark.parametrize(
    "fam",
    [
        "bench_argu_ana_studies_family",
        "bench_babi_lite_studies_family",
        "bench_curious_qa_studies_family",
        "bench_qasper_lite_studies_family",
        "bench_scifact_lite_studies_family",
        "bench_web_questions_studies_family",
    ],
)
def test_benches_w1359(fam):
    out = getattr(benches_w1359, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
