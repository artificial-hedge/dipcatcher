"""Tests for adversarial EBM losses (models/adversarial_ebm.py)."""

import numpy as np


def _net(z):
    return z[:, :1]


def _ident(z):
    return z


def test_critic_loss_prefers_low_energy_on_real():
    from quant_fund.models.adversarial_ebm import _critic_loss

    xr_lo, xf_hi = np.zeros((4, 2)), np.full((4, 2), 3.0)
    xr_hi, xf_lo = np.full((4, 2), 3.0), np.zeros((4, 2))
    good = _critic_loss(_net, xr_lo, xf_hi)
    bad = _critic_loss(_net, xr_hi, xf_lo)
    assert good < bad


def test_gen_loss_seeks_low_energy():
    from quant_fund.models.adversarial_ebm import _gen_loss

    z_lo = np.zeros((4, 2))
    z_hi = np.full((4, 2), 3.0)
    assert _gen_loss(_net, _ident, z_lo) < _gen_loss(_net, _ident, z_hi)
