import pytest


@pytest.mark.parametrize(
    "name",
    [
        "amaterasu2_qa_studies",
        "susanoo2_qa_studies",
        "tsukuyomi2_qa_studies",
        "kaguya2_qa_studies",
        "sarutahiko2_qa_studies",
        "hachiman2_qa_studies",
    ],
)
def test_w1799_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
