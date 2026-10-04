import pytest


@pytest.mark.parametrize(
    "name",
    [
        "glue_studies",
        "super_glue_studies",
        "mnli_studies",
        "qnli_studies",
        "rte_studies",
        "wnli_studies",
    ],
)
def test_w1306_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
