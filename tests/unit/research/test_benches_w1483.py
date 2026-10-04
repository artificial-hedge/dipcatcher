import pytest

from quant_fund.research import benches_w1483


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coyote_qa_studies_family",
        "bench_ferret_qa_studies_family",
        "bench_jackal_qa_studies_family",
        "bench_marmot_qa_studies_family",
        "bench_moose_qa_studies_family",
        "bench_raccoon_qa_studies_family",
    ],
)
def test_benches_w1483(fam):
    out = getattr(benches_w1483, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
