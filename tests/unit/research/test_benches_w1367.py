import pytest

from quant_fund.research import benches_w1367


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bert_score_studies_family",
        "bench_bleu_rouge_studies_family",
        "bench_bleurt_lite_studies_family",
        "bench_comet_mt_studies_family",
        "bench_meteor_lite_studies_family",
        "bench_rouge_lite_studies_family",
    ],
)
def test_benches_w1367(fam):
    out = getattr(benches_w1367, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
