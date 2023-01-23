# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from numpy import pi, exp, log

from numba import njit

from . import const
from . import opac_vp17 as opac



@njit
def grazing_angle_cg97(R):
    """
    Chiang & Goldreich 1997
    https://ui.adsabs.harvard.edu/abs/1997ApJ...490..368C
    """

    return 0.005/(R/const.AU) + 0.05*(R/const.AU)**(2.0/7.0)



@njit
def grazing_angle_dzn02(sigma, z, T_star, R_star, R):
    """
    Dullemond, van Zadelhoff & Natta 2002
    https://ui.adsabs.harvard.edu/abs/2002A%26A...389..464D
    """

    ## Grazing angle function
    fn_beta = lambda z : (0.4*R_star + (2.0/7.0)*z) / R
    ## Opacity at the stellar effective temperature [cm2 g-1]
    kappaP_star = opac.fn_kappaP(T_star)
    ## Here the optical depth is measured from the external bound
    tau = kappaP_star * (sigma[-1] - sigma)
    ## Height above the midplane where 63% (i.e. 1-exp(-1)) of the stellar
    ## radiation has been absorbed
    H_s = np.interp(exp(-1.0), exp(-tau/fn_beta(z)), z)

    return fn_beta(H_s)



@njit
def heat_star(sigma, T_star, R_star, R, beta):

    kappaP_star = opac.fn_kappaP(T_star)
    ## Here the optical depth is measured from the external bound
    if sigma[-1] > sigma[0]:
        tau = kappaP_star * (sigma[-1] - sigma)
    else:
        tau = kappaP_star * sigma
    L_star = 4*pi*R_star**2 * const.sigma_SB*T_star**4
    J_star = L_star/(4*pi*R**2) / (4*pi)
    F_star = J_star * exp(-tau/beta)

    ## Source function for the UV radiation [erg g-1 s-1]
    S = 4*pi * kappaP_star*F_star

    return S



@njit
def heat_isr(sigma, T_isr):
    """
    Parameters
    ----------
    sigma : array_like
        Surface density grid [g cm-2].
    T_isr : float
        Temperature of interstellar radiation [K].

    Returns
    -------
    ndarray
        Local heat rate [erg g-1 s-1].
    """

    ## Opacity at the interstellar radiation temperature [cm2 g-1]
    kappaP_isr = opac.fn_kappaP(T_isr)
    ## Optical depth
    if sigma[-1] > sigma[0]:
        tau = kappaP_isr * (sigma[-1] - sigma)
    else:
        tau = kappaP_isr * sigma
    ## Dilution of the UV field
    D_isr = 1e-14
    ## Mean intensity [erg cm-2 s-1]
    #J_isr = D_isr * const.c*const.a_rad/(4*pi) * T_isr**4
    J_isr = D_isr * const.sigma_SB*T_isr**4
    ## Flux [erg cm-2 s-1]
    F_isr = J_isr * exp(-tau/0.5)

    ## Source function for the UV radiation [erg g-1 s-1]
    S = 4*pi * kappaP_isr*F_isr

    return S



@njit
def heat_accr(sigma_max, M_star, dotM_star, R):
    """
    Returns
    -------
    S : float
        Local heat rate [erg g-1 s-1].
    """

    return const.G*M_star*dotM_star / (4.0*pi*R**3) / sigma_max
