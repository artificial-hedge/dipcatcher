import pytest


@pytest.mark.parametrize(
    "name",
    [
        "naberius_qa_studies",
        "glasya_labolas_qa_studies",
        "bune_qa_studies",
        "ronove_qa_studies",
        "berith_qa_studies",
        "forneus_qa_studies",
    ],
)
def test_w1951_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
