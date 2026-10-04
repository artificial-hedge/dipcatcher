import pytest

from quant_fund.research import benches_w1672


@pytest.mark.parametrize(
    "fam",
    [
        "bench_domovoi_qa_studies_family",
        "bench_kikimora_qa_studies_family",
        "bench_leshy_qa_studies_family",
        "bench_polevik_qa_studies_family",
        "bench_rusalka_qa_studies_family",
        "bench_vodianoi_qa_studies_family",
    ],
)
def test_benches_w1672(fam):
    out = getattr(benches_w1672, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
