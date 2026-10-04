import pytest

from quant_fund.research import benches_w1439


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cliff_qa_studies_family",
        "bench_crater_qa_studies_family",
        "bench_dune_qa_studies_family",
        "bench_fjord_qa_studies_family",
        "bench_gorge_qa_studies_family",
        "bench_mesa_qa_studies_family",
    ],
)
def test_benches_w1439(fam):
    out = getattr(benches_w1439, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
