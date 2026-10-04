import pytest

from quant_fund.research import benches_w1501


@pytest.mark.parametrize(
    "fam",
    [
        "bench_acacia_qa_studies_family",
        "bench_alder_qa_studies_family",
        "bench_baobab_qa_studies_family",
        "bench_olive_qa_studies_family",
        "bench_palm_qa_studies_family",
        "bench_sycamore_qa_studies_family",
    ],
)
def test_benches_w1501(fam):
    out = getattr(benches_w1501, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
