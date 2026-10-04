import pytest

from quant_fund.research import benches_w1350


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dyck_lang_studies_family",
        "bench_hops_add_studies_family",
        "bench_lcmc_lite_studies_family",
        "bench_mco_lite_studies_family",
        "bench_scan_cfsp_studies_family",
        "bench_shuffle_expr_studies_family",
    ],
)
def test_benches_w1350(fam):
    out = getattr(benches_w1350, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
