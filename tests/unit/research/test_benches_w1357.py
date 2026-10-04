import pytest

from quant_fund.research import benches_w1357


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hendrycks_test_studies_family",
        "bench_hotpot_lite_studies_family",
        "bench_multirc_lite_studies_family",
        "bench_quoref_lite_studies_family",
        "bench_record_lite_studies_family",
        "bench_squad_lite2_studies_family",
    ],
)
def test_benches_w1357(fam):
    out = getattr(benches_w1357, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
