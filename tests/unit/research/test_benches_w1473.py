import pytest

from quant_fund.research import benches_w1473


@pytest.mark.parametrize(
    "fam",
    [
        "bench_clove_qa_studies_family",
        "bench_dill_qa_studies_family",
        "bench_fennel_qa_studies_family",
        "bench_lemongrass_qa_studies_family",
        "bench_mint_qa_studies_family",
        "bench_nutmeg_qa_studies_family",
    ],
)
def test_benches_w1473(fam):
    out = getattr(benches_w1473, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
