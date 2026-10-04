import pytest

from quant_fund.research import benches_w1341


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bold_eval_studies_family",
        "bench_crow_s_pairs_studies_family",
        "bench_hate_speech_eval_studies_family",
        "bench_holo_bias_studies_family",
        "bench_real_toxicity_studies_family",
        "bench_stereo_set_studies_family",
    ],
)
def test_benches_w1341(fam):
    out = getattr(benches_w1341, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
