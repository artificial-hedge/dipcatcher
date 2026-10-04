import pytest

from quant_fund.research import benches_w1337


@pytest.mark.parametrize(
    "fam",
    [
        "bench_android_env_studies_family",
        "bench_api_bank_studies_family",
        "bench_gaia_level_studies_family",
        "bench_video_game_studies_family",
        "bench_voyager_minecraft_studies_family",
        "bench_web_shopping_studies_family",
    ],
)
def test_benches_w1337(fam):
    out = getattr(benches_w1337, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
