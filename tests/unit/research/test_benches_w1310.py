import pytest

from quant_fund.research import benches_w1310


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alpaca_eval_studies_family",
        "bench_attribution_eval_studies_family",
        "bench_citation_eval_studies_family",
        "bench_diversity_eval_studies_family",
        "bench_factscore_studies_family",
        "bench_self_bleu_studies_family",
    ],
)
def test_benches_w1310(fam):
    out = getattr(benches_w1310, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
