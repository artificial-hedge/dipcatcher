import pytest

from quant_fund.research import benches_w1457


@pytest.mark.parametrize(
    "fam",
    [
        "bench_badger_qa_studies_family",
        "bench_beaver_qa_studies_family",
        "bench_bison_qa_studies_family",
        "bench_cougar_qa_studies_family",
        "bench_elk_qa_studies_family",
        "bench_lynx_qa_studies_family",
    ],
)
def test_benches_w1457(fam):
    out = getattr(benches_w1457, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
