import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cihuateteo_qa_studies",
        "tzitzimitl_qa_studies",
        "nagual_qa_studies",
        "tlalocan_qa_studies",
        "chaneque_qa_studies",
        "xiuhcoatl_qa_studies",
    ],
)
def test_w1675_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
