import pytest

from quant_fund.research import benches_w1448


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cobra_qa_studies_family",
        "bench_frog_qa_studies_family",
        "bench_gecko_qa_studies_family",
        "bench_iguana_qa_studies_family",
        "bench_python_qa_studies_family",
        "bench_viper_qa_studies_family",
    ],
)
def test_benches_w1448(fam):
    out = getattr(benches_w1448, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
