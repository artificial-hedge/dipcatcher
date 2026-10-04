import pytest

from quant_fund.research import benches_w1569


@pytest.mark.parametrize(
    "fam",
    [
        "bench_eagle_ray_qa_studies_family",
        "bench_guitarfish_qa_studies_family",
        "bench_manta_qa_studies_family",
        "bench_sawfish_qa_studies_family",
        "bench_thornback_qa_studies_family",
        "bench_torpedo_ray_qa_studies_family",
    ],
)
def test_benches_w1569(fam):
    out = getattr(benches_w1569, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
