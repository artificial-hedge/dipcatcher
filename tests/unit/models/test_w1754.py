import pytest


@pytest.mark.parametrize(
    "name",
    [
        "hachiman_qa_studies",
        "inari_qa_studies",
        "uzume_qa_studies",
        "fujin_qa_studies",
        "raijin_qa_studies",
        "sarutahiko_qa_studies",
    ],
)
def test_w1754_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
