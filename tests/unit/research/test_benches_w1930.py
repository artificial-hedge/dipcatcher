import pytest

from quant_fund.research import benches_w1930


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bdud_qa_studies_family",
        "bench_bgegs_qa_studies_family",
        "bench_btsan_qa_studies_family",
        "bench_gdon_qa_studies_family",
        "bench_gnod_sbyin_qa_studies_family",
        "bench_srin_po_qa_studies_family",
    ],
)
def test_benches_w1930(fam):
    out = getattr(benches_w1930, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
