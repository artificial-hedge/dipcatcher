import pytest

from quant_fund.research import benches_w1495


@pytest.mark.parametrize(
    "fam",
    [
        "bench_grison_qa_studies_family",
        "bench_sable_qa_studies_family",
        "bench_stoat_qa_studies_family",
        "bench_tayra_qa_studies_family",
        "bench_weasel_qa_studies_family",
        "bench_zorilla_qa_studies_family",
    ],
)
def test_benches_w1495(fam):
    out = getattr(benches_w1495, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
