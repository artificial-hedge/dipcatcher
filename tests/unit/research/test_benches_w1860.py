import pytest

from quant_fund.research import benches_w1860


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arthur_qa_studies_family",
        "bench_elaine_qa_studies_family",
        "bench_gorlois_qa_studies_family",
        "bench_igraine_qa_studies_family",
        "bench_morgan_qa_studies_family",
        "bench_vivien_qa_studies_family",
    ],
)
def test_benches_w1860(fam):
    out = getattr(benches_w1860, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
