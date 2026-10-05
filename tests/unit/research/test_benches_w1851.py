import pytest

from quant_fund.research import benches_w1851


@pytest.mark.parametrize(
    "fam",
    [
        "bench_achaman_qa_studies_family",
        "bench_achuguayo_qa_studies_family",
        "bench_chaxiraxi_qa_studies_family",
        "bench_guayota_qa_studies_family",
        "bench_magec_qa_studies_family",
        "bench_tibicena_qa_studies_family",
    ],
)
def test_benches_w1851(fam):
    out = getattr(benches_w1851, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
