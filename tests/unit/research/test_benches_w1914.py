import pytest

from quant_fund.research import benches_w1914


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ghul_qa_studies_family",
        "bench_ifrit_qa_studies_family",
        "bench_jann_qa_studies_family",
        "bench_marid_qa_studies_family",
        "bench_nasnas_qa_studies_family",
        "bench_shaitan_qa_studies_family",
    ],
)
def test_benches_w1914(fam):
    out = getattr(benches_w1914, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
