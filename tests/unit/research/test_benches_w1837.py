import pytest

from quant_fund.research import benches_w1837


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apollopatara2_qa_studies_family",
        "bench_eliyana2_qa_studies_family",
        "bench_erbbina2_qa_studies_family",
        "bench_leto2_qa_studies_family",
        "bench_trqqiz2_qa_studies_family",
        "bench_xssentimi2_qa_studies_family",
    ],
)
def test_benches_w1837(fam):
    out = getattr(benches_w1837, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
