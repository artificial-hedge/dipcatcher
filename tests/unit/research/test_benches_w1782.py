import pytest

from quant_fund.research import benches_w1782


@pytest.mark.parametrize(
    "fam",
    [
        "bench_geb_qa_studies_family",
        "bench_horus_qa_studies_family",
        "bench_isis_qa_studies_family",
        "bench_osiris_qa_studies_family",
        "bench_set_qa_studies_family",
        "bench_shu_qa_studies_family",
    ],
)
def test_benches_w1782(fam):
    out = getattr(benches_w1782, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
