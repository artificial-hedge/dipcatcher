import pytest

from quant_fund.research import benches_w1400


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cronqa_lite_studies_family",
        "bench_cwq_lite_studies_family",
        "bench_grailqa_studies_family",
        "bench_kgqa_lite_studies_family",
        "bench_pweb_qa_studies_family",
        "bench_qald_lite_studies_family",
    ],
)
def test_benches_w1400(fam):
    out = getattr(benches_w1400, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
