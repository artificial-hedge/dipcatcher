import pytest

from quant_fund.research import benches_w1682


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alkonost_qa_studies_family",
        "bench_gamayun_qa_studies_family",
        "bench_sirin_qa_studies_family",
        "bench_veles_qa_studies_family",
        "bench_zhaba_qa_studies_family",
        "bench_zmei_qa_studies_family",
    ],
)
def test_benches_w1682(fam):
    out = getattr(benches_w1682, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
