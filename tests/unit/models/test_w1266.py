import pytest


@pytest.mark.parametrize(
    "name",
    [
        "diffusion_lm_studies",
        "kv_compression_studies",
        "medusa_speculation_studies",
        "moe_shared_expert_studies",
        "rope_scaling_studies",
        "sparse_attention_studies",
    ],
)
def test_w1266_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
