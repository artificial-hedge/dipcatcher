import pytest

from quant_fund.research import benches_w1510


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chough_qa_studies_family",
        "bench_crow_qa_studies_family",
        "bench_jackdaw_qa_studies_family",
        "bench_jay_qa_studies_family",
        "bench_magpie_qa_studies_family",
        "bench_rook_qa_studies_family",
    ],
)
def test_benches_w1510(fam):
    out = getattr(benches_w1510, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
