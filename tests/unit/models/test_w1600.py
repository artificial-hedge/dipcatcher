import pytest


@pytest.mark.parametrize(
    "name",
    [
        "limpet_qa_studies",
        "chiton_qa_studies",
        "periwinkle_qa_studies",
        "cockle_qa_studies",
        "abalone_qa_studies",
        "cowrie_qa_studies",
    ],
)
def test_w1600_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
