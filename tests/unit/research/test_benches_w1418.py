import pytest

from quant_fund.research import benches_w1418


@pytest.mark.parametrize(
    "fam",
    [
        "bench_almanac_qa_studies_family",
        "bench_atlas_qa_studies_family",
        "bench_idiom_qa_studies_family",
        "bench_jeopardy_qa_studies_family",
        "bench_misc_qa_studies_family",
        "bench_myth_qa_studies_family",
    ],
)
def test_benches_w1418(fam):
    out = getattr(benches_w1418, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
