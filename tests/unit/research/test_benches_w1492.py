import pytest

from quant_fund.research import benches_w1492


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bilby_qa_studies_family",
        "bench_echidna_qa_studies_family",
        "bench_platypus_qa_studies_family",
        "bench_possum_qa_studies_family",
        "bench_quoll_qa_studies_family",
        "bench_thylacine_qa_studies_family",
    ],
)
def test_benches_w1492(fam):
    out = getattr(benches_w1492, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
