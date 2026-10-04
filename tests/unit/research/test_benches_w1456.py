import pytest

from quant_fund.research import benches_w1456


@pytest.mark.parametrize(
    "fam",
    [
        "bench_barracuda_qa_studies_family",
        "bench_catfish_qa_studies_family",
        "bench_cod_qa_studies_family",
        "bench_piranha_qa_studies_family",
        "bench_salmon_qa_studies_family",
        "bench_tuna_qa_studies_family",
    ],
)
def test_benches_w1456(fam):
    out = getattr(benches_w1456, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
