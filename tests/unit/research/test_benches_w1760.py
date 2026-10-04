import pytest

from quant_fund.research import benches_w1760


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ebisu_qa_studies_family",
        "bench_hiruko_qa_studies_family",
        "bench_kikuzuki_qa_studies_family",
        "bench_kisshoten_qa_studies_family",
        "bench_morinaga_qa_studies_family",
        "bench_senju_qa_studies_family",
    ],
)
def test_benches_w1760(fam):
    out = getattr(benches_w1760, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
