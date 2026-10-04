import pytest

from quant_fund.research import benches_w1353


@pytest.mark.parametrize(
    "fam",
    [
        "bench_creak_lite_studies_family",
        "bench_entailment_bn_studies_family",
        "bench_hans_lite_studies_family",
        "bench_prove_it_studies_family",
        "bench_strategy_qa_studies_family",
        "bench_sup_nli_studies_family",
    ],
)
def test_benches_w1353(fam):
    out = getattr(benches_w1353, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
