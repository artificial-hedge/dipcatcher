import pytest


@pytest.mark.parametrize(
    "name",
    [
        "catoblepas_qa_studies",
        "jasconius_qa_studies",
        "zaratan_qa_studies",
        "peluda_qa_studies",
        "pard_qa_studies",
        "alion_qa_studies",
    ],
)
def test_w1653_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
