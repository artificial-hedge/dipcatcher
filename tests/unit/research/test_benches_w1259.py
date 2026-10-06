import pytest

from quant_fund.research import benches_w1259


@pytest.mark.parametrize(
    "fam",
    [
        "bench_qsar_studies_family",
        "bench_docking_studies_family",
        "bench_admet_studies_family",
        "bench_lead_optimization_studies_family",
        "bench_virtual_screening_studies_family",
        "bench_de_novo_design_studies_family",
    ],
)
def test_benches_w1259(fam):
    out = getattr(benches_w1259, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
