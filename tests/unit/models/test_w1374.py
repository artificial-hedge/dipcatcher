import pytest


@pytest.mark.parametrize(
    "name",
    [
        "deduc_lite_studies",
        "logic_nli_studies",
        "folio_lite_studies",
        "proof_writer_studies",
        "rule_taker_studies",
        "entail_bank_studies",
    ],
)
def test_w1374_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
