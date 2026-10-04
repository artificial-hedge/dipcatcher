import pytest

from quant_fund.research import benches_w1486


@pytest.mark.parametrize(
    "fam",
    [
        "bench_flamingo_qa_studies_family",
        "bench_godwit_qa_studies_family",
        "bench_grebe_qa_studies_family",
        "bench_pelican_qa_studies_family",
        "bench_spoonbill_qa_studies_family",
        "bench_stork_qa_studies_family",
    ],
)
def test_benches_w1486(fam):
    out = getattr(benches_w1486, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
