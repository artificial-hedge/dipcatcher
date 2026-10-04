import pytest


@pytest.mark.parametrize(
    "name",
    [
        "design_qa_studies",
        "layout_qa_studies",
        "format_qa_studies",
        "pattern_qa_studies",
        "schema_qa_studies",
        "blueprint_qa_studies",
    ],
)
def test_w1424_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
