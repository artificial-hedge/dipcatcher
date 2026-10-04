import pytest

from quant_fund.research import benches_w1664


@pytest.mark.parametrize(
    "fam",
    [
        "bench_kraken_qa_studies_family",
        "bench_krampus_qa_studies_family",
        "bench_roc_qa_studies_family",
        "bench_simurgh_qa_studies_family",
        "bench_siren_qa_studies_family",
        "bench_wyvern_qa_studies_family",
    ],
)
def test_benches_w1664(fam):
    out = getattr(benches_w1664, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
