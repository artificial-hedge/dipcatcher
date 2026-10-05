import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kel_essuf_qa_studies",
        "tanit_lok_qa_studies",
        "tin_hinan_qa_studies",
        "djinnet_qa_studies",
        "afriye_qa_studies",
        "almajira_qa_studies",
    ],
)
def test_w1879_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
