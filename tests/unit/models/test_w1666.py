import pytest


@pytest.mark.parametrize(
    "name",
    [
        "agta_qa_studies",
        "berberoka_qa_studies",
        "bungisngis_qa_studies",
        "dalaketnon_qa_studies",
        "ekek_qa_studies",
        "engkanto_qa_studies",
    ],
)
def test_w1666_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
