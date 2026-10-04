import pytest

from quant_fund.research import benches_w1631


@pytest.mark.parametrize(
    "fam",
    [
        "bench_basilisk_qa_studies_family",
        "bench_chimera_qa_studies_family",
        "bench_gorgon_qa_studies_family",
        "bench_griffin_2_qa_studies_family",
        "bench_hydra_2_qa_studies_family",
        "bench_manticore_qa_studies_family",
    ],
)
def test_benches_w1631(fam):
    out = getattr(benches_w1631, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
