import pytest

from quant_fund.research import benches_w1378


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aqua_lite_studies_family",
        "bench_fin_qa_studies_family",
        "bench_math_qa_studies_family",
        "bench_num_glue_studies_family",
        "bench_tab_fact_studies_family",
        "bench_tat_qa_studies_family",
    ],
)
def test_benches_w1378(fam):
    out = getattr(benches_w1378, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
