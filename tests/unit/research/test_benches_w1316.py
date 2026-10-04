import pytest

from quant_fund.research import benches_w1316


@pytest.mark.parametrize(
    "fam",
    [
        "bench_backgrounds_studies_family",
        "bench_cue_conflict_studies_family",
        "bench_geirhos_studies_family",
        "bench_imagenet_bg_studies_family",
        "bench_shape_bias_studies_family",
        "bench_texture_bias_studies_family",
    ],
)
def test_benches_w1316(fam):
    out = getattr(benches_w1316, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
