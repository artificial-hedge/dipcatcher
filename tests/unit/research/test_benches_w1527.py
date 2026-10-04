import pytest

from quant_fund.research import benches_w1527


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crustose_qa_studies_family",
        "bench_foliose_qa_studies_family",
        "bench_fruticose_qa_studies_family",
        "bench_oakmoss_qa_studies_family",
        "bench_usnea_qa_studies_family",
        "bench_xanthoria_qa_studies_family",
    ],
)
def test_benches_w1527(fam):
    out = getattr(benches_w1527, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
