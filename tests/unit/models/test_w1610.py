import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ratel_qa_studies",
        "mara_qa_studies",
        "porcupine_qa_studies",
        "dhole_qa_studies",
        "coypu_qa_studies",
        "cavy_qa_studies",
    ],
)
def test_w1610_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
