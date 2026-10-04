import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ishtar_qa_studies",
        "shamash_qa_studies",
        "sin_qa_studies",
        "nabu_qa_studies",
        "ashur_qa_studies",
        "adad_qa_studies",
    ],
)
def test_w1704_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
