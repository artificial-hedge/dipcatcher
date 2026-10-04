import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ogboni_qa_studies",
        "obayifo_qa_studies",
        "aigamuxa_qa_studies",
        "emere_qa_studies",
        "dodo_spirit_qa_studies",
        "kishi_demon_qa_studies",
    ],
)
def test_w1907_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
