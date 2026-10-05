import pytest

from quant_fund.research import benches_w1907


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aigamuxa_qa_studies_family",
        "bench_dodo_spirit_qa_studies_family",
        "bench_emere_qa_studies_family",
        "bench_kishi_demon_qa_studies_family",
        "bench_obayifo_qa_studies_family",
        "bench_ogboni_qa_studies_family",
    ],
)
def test_benches_w1907(fam):
    out = getattr(benches_w1907, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
