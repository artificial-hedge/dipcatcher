import pytest

from quant_fund.research import benches_w1763


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hapi_qa_studies_family",
        "bench_khnum_qa_studies_family",
        "bench_menhit_qa_studies_family",
        "bench_nephthys_qa_studies_family",
        "bench_serqet_qa_studies_family",
        "bench_tefnut_qa_studies_family",
    ],
)
def test_benches_w1763(fam):
    out = getattr(benches_w1763, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
