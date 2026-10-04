import pytest

from quant_fund.research import benches_w1490


@pytest.mark.parametrize(
    "fam",
    [
        "bench_camellia_qa_studies_family",
        "bench_dahlia_qa_studies_family",
        "bench_sage_qa_studies_family",
        "bench_thyme_qa_studies_family",
        "bench_violet_qa_studies_family",
        "bench_zinnia_qa_studies_family",
    ],
)
def test_benches_w1490(fam):
    out = getattr(benches_w1490, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
