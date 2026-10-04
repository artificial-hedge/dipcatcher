import pytest

from quant_fund.research import benches_w1381


@pytest.mark.parametrize(
    "fam",
    [
        "bench_begins_lite_studies_family",
        "bench_diamonds_lite_studies_family",
        "bench_faithful_dial_studies_family",
        "bench_multi_woz_studies_family",
        "bench_top_dialog_studies_family",
        "bench_wow_lite_studies_family",
    ],
)
def test_benches_w1381(fam):
    out = getattr(benches_w1381, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
