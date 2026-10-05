import pytest


@pytest.mark.parametrize(
    "name",
    [
        "varuna_qa_studies",
        "vritra_qa_studies",
        "indra_hindu_qa_studies",
        "dakini_qa_studies",
        "zurvan_qa_studies",
        "jamshid_qa_studies",
    ],
)
def test_w1887_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
