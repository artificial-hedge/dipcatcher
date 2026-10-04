import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tane_qa_studies",
        "tangaroa_qa_studies",
        "tumatauenga_qa_studies",
        "rongo_qa_studies",
        "haumia_qa_studies",
        "tawhirimatea_qa_studies",
    ],
)
def test_w1742_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
