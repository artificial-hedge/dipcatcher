import pytest


@pytest.mark.parametrize(
    "name",
    [
        "vainamoinen_qa_studies",
        "ilmarinen_qa_studies",
        "lemminkainen_qa_studies",
        "joukahainen_qa_studies",
        "marjatta_qa_studies",
        "tuoni_qa_studies",
    ],
)
def test_w1736_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
