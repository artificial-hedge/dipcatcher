import pytest


@pytest.mark.parametrize(
    "name",
    [
        "anli_lite_studies",
        "paws_lite_studies",
        "mrpc_lite_studies",
        "quora_dup_studies",
        "rte_lite_studies",
        "mnli_lite_studies",
    ],
)
def test_w1364_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
