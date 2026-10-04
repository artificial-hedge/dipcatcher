import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cerberus_qa_studies",
        "minotaur_qa_studies",
        "pegasus_qa_studies",
        "unicorn_2_qa_studies",
        "dragon_2_qa_studies",
        "phoenix_2_qa_studies",
    ],
)
def test_w1632_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
