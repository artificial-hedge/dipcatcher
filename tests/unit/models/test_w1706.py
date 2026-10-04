import pytest


@pytest.mark.parametrize(
    "name",
    [
        "tengri_qa_studies",
        "ulgen_qa_studies",
        "erlik_qa_studies",
        "kayra_qa_studies",
        "akana_qa_studies",
        "perysh_qa_studies",
    ],
)
def test_w1706_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
