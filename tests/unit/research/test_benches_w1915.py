import pytest

from quant_fund.research import benches_w1915


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dybbuk_qa_studies_family",
        "bench_ibbur_qa_studies_family",
        "bench_lilim_qa_studies_family",
        "bench_mazzik_qa_studies_family",
        "bench_seirim_qa_studies_family",
        "bench_shedim_qa_studies_family",
    ],
)
def test_benches_w1915(fam):
    out = getattr(benches_w1915, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
