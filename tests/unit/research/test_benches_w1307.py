import pytest

from quant_fund.research import benches_w1307


@pytest.mark.parametrize(
    "fam",
    [
        "bench_lambada_studies_family",
        "bench_record_studies_family",
        "bench_story_cloze_studies_family",
        "bench_winogender_studies_family",
        "bench_winograd_studies_family",
        "bench_wsc_studies_family",
    ],
)
def test_benches_w1307(fam):
    out = getattr(benches_w1307, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
