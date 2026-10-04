import pytest


@pytest.mark.parametrize(
    "name",
    [
        "pale_titi_qa_studies",
        "owl_monkey_qa_studies",
        "bamboo_lemur_qa_studies",
        "bearded_saki_qa_studies",
        "uakari_2_qa_studies",
        "woolly_lemur_qa_studies",
    ],
)
def test_w1607_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
