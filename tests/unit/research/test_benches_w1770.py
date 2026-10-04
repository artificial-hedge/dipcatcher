import pytest

from quant_fund.research import benches_w1770


@pytest.mark.parametrize(
    "fam",
    [
        "bench_enkidu_qa_studies_family",
        "bench_etana_qa_studies_family",
        "bench_gilgamesh_qa_studies_family",
        "bench_kingu_qa_studies_family",
        "bench_nabu_qa_studies_family",
        "bench_tiamat_qa_studies_family",
    ],
)
def test_benches_w1770(fam):
    out = getattr(benches_w1770, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
