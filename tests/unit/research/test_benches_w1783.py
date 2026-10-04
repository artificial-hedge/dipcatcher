import pytest

from quant_fund.research import benches_w1783


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baldur_qa_studies_family",
        "bench_freyr_qa_studies_family",
        "bench_hermodr_qa_studies_family",
        "bench_hodr_qa_studies_family",
        "bench_njord_qa_studies_family",
        "bench_skadi_qa_studies_family",
    ],
)
def test_benches_w1783(fam):
    out = getattr(benches_w1783, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
