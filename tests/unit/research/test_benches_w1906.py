import pytest

from quant_fund.research import benches_w1906


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amefuri_kozo_qa_studies_family",
        "bench_dosanjin_qa_studies_family",
        "bench_fuon_qa_studies_family",
        "bench_kejoro_qa_studies_family",
        "bench_nakisawame_qa_studies_family",
        "bench_yama_waro_qa_studies_family",
    ],
)
def test_benches_w1906(fam):
    out = getattr(benches_w1906, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
