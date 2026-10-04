import pytest

from quant_fund.research import benches_w1368


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chr_f_studies_family",
        "bench_mover_score_studies_family",
        "bench_nist_metric_studies_family",
        "bench_prism_mt_studies_family",
        "bench_sacrebleu_lite_studies_family",
        "bench_ter_lite_studies_family",
    ],
)
def test_benches_w1368(fam):
    out = getattr(benches_w1368, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
