import pytest

from quant_fund.research import benches_w1785


@pytest.mark.parametrize(
    "fam",
    [
        "bench_benzaiten_qa_studies_family",
        "bench_hoori_qa_studies_family",
        "bench_jurojin_qa_studies_family",
        "bench_kushinadahime_qa_studies_family",
        "bench_toyotamahime_qa_studies_family",
        "bench_yamatotakeru_qa_studies_family",
    ],
)
def test_benches_w1785(fam):
    out = getattr(benches_w1785, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
