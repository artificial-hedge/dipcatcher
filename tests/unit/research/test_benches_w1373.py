import pytest

from quant_fund.research import benches_w1373


@pytest.mark.parametrize(
    "fam",
    [
        "bench_commonsense_lite_studies_family",
        "bench_logi_qa_studies_family",
        "bench_mr_lite_studies_family",
        "bench_muin_lite_studies_family",
        "bench_qasc_sci2_studies_family",
        "bench_winogrande_lite_studies_family",
    ],
)
def test_benches_w1373(fam):
    out = getattr(benches_w1373, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
