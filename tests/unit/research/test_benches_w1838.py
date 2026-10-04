import pytest

from quant_fund.research import benches_w1838


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aglibol2_qa_studies_family",
        "bench_astarte2_qa_studies_family",
        "bench_baalshamin2_qa_studies_family",
        "bench_bel2_qa_studies_family",
        "bench_malakbel2_qa_studies_family",
        "bench_yarhibol2_qa_studies_family",
    ],
)
def test_benches_w1838(fam):
    out = getattr(benches_w1838, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
