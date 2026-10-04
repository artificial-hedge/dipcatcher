import pytest


@pytest.mark.parametrize(
    "name",
    [
        "basiliskcock_qa_studies",
        "cocatrix_qa_studies",
        "gryps_qa_studies",
        "calygreyhound_qa_studies",
        "opinicus_qa_studies",
        "mantygre_qa_studies",
    ],
)
def test_w1655_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
