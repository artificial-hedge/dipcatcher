import pytest

from quant_fund.research import benches_w1629


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chupacabra_qa_studies_family",
        "bench_jersey_devil_qa_studies_family",
        "bench_kraken_2_qa_studies_family",
        "bench_mothman_qa_studies_family",
        "bench_thunderbird_qa_studies_family",
        "bench_yeti_2_qa_studies_family",
    ],
)
def test_benches_w1629(fam):
    out = getattr(benches_w1629, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
