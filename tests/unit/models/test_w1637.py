import pytest


@pytest.mark.parametrize(
    "name",
    [
        "rokurokubi_qa_studies",
        "futakuchi_qa_studies",
        "shirime_qa_studies",
        "abura_sumashi_qa_studies",
        "azukiarai_qa_studies",
        "betobeto_2_qa_studies",
    ],
)
def test_w1637_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
