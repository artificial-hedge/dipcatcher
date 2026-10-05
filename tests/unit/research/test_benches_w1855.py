import pytest

from quant_fund.research import benches_w1855


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ataecina_qa_studies_family",
        "bench_bandua_qa_studies_family",
        "bench_cariocecus_qa_studies_family",
        "bench_endovellicus_qa_studies_family",
        "bench_nabia_qa_studies_family",
        "bench_trebaruna_qa_studies_family",
    ],
)
def test_benches_w1855(fam):
    out = getattr(benches_w1855, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
