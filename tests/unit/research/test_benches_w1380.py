import pytest

from quant_fund.research import benches_w1380


@pytest.mark.parametrize(
    "fam",
    [
        "bench_art_nli_studies_family",
        "bench_para_paws_studies_family",
        "bench_recast_lite_studies_family",
        "bench_snips_lite_studies_family",
        "bench_social_lite_studies_family",
        "bench_subj_lite_studies_family",
    ],
)
def test_benches_w1380(fam):
    out = getattr(benches_w1380, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
