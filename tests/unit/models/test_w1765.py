import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nuada_qa_studies",
        "lugh_qa_studies",
        "morrigan_qa_studies",
        "cerridwen_qa_studies",
        "rhiannon_qa_studies",
        "arianrhod_qa_studies",
    ],
)
def test_w1765_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
