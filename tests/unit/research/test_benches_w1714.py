import pytest

from quant_fund.research import benches_w1714


@pytest.mark.parametrize(
    "fam",
    [
        "bench_argimpasa_qa_studies_family",
        "bench_arimasp_qa_studies_family",
        "bench_papaios_qa_studies_family",
        "bench_tabiti_qa_studies_family",
        "bench_tavrita_qa_studies_family",
        "bench_thagimasadas_qa_studies_family",
    ],
)
def test_benches_w1714(fam):
    out = getattr(benches_w1714, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
