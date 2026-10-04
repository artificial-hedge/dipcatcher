import pytest

from quant_fund.research import benches_w1342


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fairness_eval_studies_family",
        "bench_gender_bias_studies_family",
        "bench_jigsaw_tox_studies_family",
        "bench_nlp_bias_studies_family",
        "bench_pronoun_bias_studies_family",
        "bench_regard_metric_studies_family",
    ],
)
def test_benches_w1342(fam):
    out = getattr(benches_w1342, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
