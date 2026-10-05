import pytest


@pytest.mark.parametrize(
    "name",
    [
        "achaman_qa_studies",
        "chaxiraxi_qa_studies",
        "magec_qa_studies",
        "guayota_qa_studies",
        "tibicena_qa_studies",
        "achuguayo_qa_studies",
    ],
)
def test_w1851_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
