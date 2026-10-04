import pytest

from quant_fund.research import benches_w1659


@pytest.mark.parametrize(
    "fam",
    [
        "bench_manananggal_qa_studies_family",
        "bench_minokawa_qa_studies_family",
        "bench_nuno_qa_studies_family",
        "bench_siyokoy_qa_studies_family",
        "bench_tiyanak_qa_studies_family",
        "bench_wakwak_qa_studies_family",
    ],
)
def test_benches_w1659(fam):
    out = getattr(benches_w1659, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
