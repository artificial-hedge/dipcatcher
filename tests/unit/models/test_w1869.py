import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ankou_qa_studies",
        "korrigan_qa_studies",
        "yeun_elez_qa_studies",
        "mari_morgen_qa_studies",
        "tangi_qa_studies",
        "gwalarn_qa_studies",
    ],
)
def test_w1869_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
