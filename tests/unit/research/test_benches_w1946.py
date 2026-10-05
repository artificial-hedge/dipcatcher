import pytest

from quant_fund.research import benches_w1946


@pytest.mark.parametrize(
    "fam",
    [
        "bench_egui_qa_studies_family",
        "bench_heibai_qa_studies_family",
        "bench_meng_po_qa_studies_family",
        "bench_niutou_qa_studies_family",
        "bench_wangliang_qa_studies_family",
        "bench_yanwang_qa_studies_family",
    ],
)
def test_benches_w1946(fam):
    out = getattr(benches_w1946, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
