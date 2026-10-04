from quant_fund.models.expected_sig import (
    bench_expected_sig,
)
from quant_fund.models.pde_signature import (
    bench_pde_signature,
)
from quant_fund.models.sig_inversion import (
    bench_sig_inversion,
)
from quant_fund.models.signature_gan2 import (
    bench_signature_gan2,
)
from quant_fund.models.signature_kernel import (
    bench_signature_kernel,
)
from quant_fund.models.truncated_sig import (
    bench_truncated_sig,
)


def test_signature_kernel():
    assert bench_signature_kernel()["synthetic_signature_kernel"] == 1.0


def test_pde_signature():
    assert bench_pde_signature()["synthetic_pde_signature"] == 1.0


def test_truncated_sig():
    assert bench_truncated_sig()["synthetic_truncated_sig"] == 1.0


def test_signature_gan2():
    assert bench_signature_gan2()["synthetic_signature_gan2"] == 1.0


def test_expected_sig():
    assert bench_expected_sig()["synthetic_expected_sig"] == 1.0


def test_sig_inversion():
    assert bench_sig_inversion()["synthetic_sig_inversion"] == 1.0
