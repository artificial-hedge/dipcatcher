import pytest

from quant_fund.research import benches_w1936


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aoao_qa_studies_family",
        "bench_jasy_jatere_qa_studies_family",
        "bench_kurupi_qa_studies_family",
        "bench_luison_qa_studies_family",
        "bench_mboitui_qa_studies_family",
        "bench_pombero_qa_studies_family",
    ],
)
def test_benches_w1936(fam):
    out = getattr(benches_w1936, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
