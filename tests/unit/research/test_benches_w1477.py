import pytest

from quant_fund.research import benches_w1477


@pytest.mark.parametrize(
    "fam",
    [
        "bench_albatross_qa_studies_family",
        "bench_gannet_qa_studies_family",
        "bench_petrel_qa_studies_family",
        "bench_puffin_qa_studies_family",
        "bench_shearwater_qa_studies_family",
        "bench_skua_qa_studies_family",
    ],
)
def test_benches_w1477(fam):
    out = getattr(benches_w1477, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
