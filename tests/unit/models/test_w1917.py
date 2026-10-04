import pytest


@pytest.mark.parametrize(
    "name",
    [
        "boggart_qa_studies",
        "sluagh_qa_studies",
        "bean_nighe_qa_studies",
        "baobhan_sith_qa_studies",
        "fear_durach_qa_studies",
        "glaistig_qa_studies",
    ],
)
def test_w1917_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
