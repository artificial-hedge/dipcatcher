import pytest

from quant_fund.research import benches_w1732


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dziewanna_qa_studies_family",
        "bench_marzanna_qa_studies_family",
        "bench_mokosz_qa_studies_family",
        "bench_nija_qa_studies_family",
        "bench_swarozyc_qa_studies_family",
        "bench_zywie_qa_studies_family",
    ],
)
def test_benches_w1732(fam):
    out = getattr(benches_w1732, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
