import pytest

from quant_fund.research import benches_w1873


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ar_marzh_qa_studies_family",
        "bench_darkman_qa_studies_family",
        "bench_kaier_qa_studies_family",
        "bench_noz_vat_qa_studies_family",
        "bench_paotr_bugel_qa_studies_family",
        "bench_santez_nonna_qa_studies_family",
    ],
)
def test_benches_w1873(fam):
    out = getattr(benches_w1873, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
