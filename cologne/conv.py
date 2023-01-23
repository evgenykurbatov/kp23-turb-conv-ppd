# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from numpy import sqrt, log

from numba import njit

from . import const
from . import opac_vp17 as opac
from . import util



@njit
def convection(z, rho, T, g, mu_mol, gamma, kappaR, ell):
    """
    Parameters
    ----------
    z : array of floats
        Nodes of the z-grid.
    rho : array of floats
        Density at the nodes of the z-grid. Shape is `(len(z)-1)`.
    T : array of floats
        Temperature at the nodes.
    g : array of floats
        Projection of gravitational acceleration onto z-axis. Shape is `len(z)`.
    mu_mol : float
        Mean molocular weight of the gas.
    gamma : float
        Adiabatic index of the gas.
    kappaR : array of floats
        Rosseland mean opacity [cm2 g-1].
    ell : float
        Mixing length [cm].

    Returns
    -------
    """

    ##
    ## Basic flux and heat source

    m = mu_mol*const.m_H
    C_p = gamma/(gamma-1.0) * const.RR_gas/mu_mol
    C_v = 1.0/(gamma-1.0)   * const.RR_gas/mu_mol

    ## Flux
    #F_conv, effconv = flux_c92(z, rho, T, g, m, C_p, C_v, kappaR, ell)
    F_conv, effconv = flux_hk94(z, rho, T, g, m, C_p, C_v, kappaR, ell)

    ## Heat sources
    ## - (C_B/2)*Bzz
    dlogP_dz = util.grads(log(rho*T), z)
    C_B = 0.6
    S_conv = 0.5*C_B * 2.0*(gamma-1.0)/gamma * F_conv * dlogP_dz / rho
    ## - d(F_conv)/dz
    S_conv -= util.grad(F_conv, z) / rho

    dSconv_dT = np.zeros_like(z)

    return F_conv, effconv, S_conv, dSconv_dT



@njit
def convection_var(z, rho, T, g, mu_mol, gamma, kappaR, ell):
    """
    Parameters
    ----------
    z : array of floats
        Nodes of the z-grid.
    rho : array of floats
        Density at the nodes of the z-grid. Shape is `(len(z)-1)`.
    T : array of floats
        Temperature at the nodes.
    g : array of floats
        Projection of gravitational acceleration onto z-axis. Shape is `len(z)`.
    mu_mol : float
        Mean molocular weight of the gas.
    gamma : float
        Adiabatic index of the gas.
    kappaR : array of floats
        Rosseland mean opacity [cm2 g-1].
    ell : float
        Mixing length [cm].

    Returns
    -------
    """

    ##
    ## Basic flux and heat source

    m = mu_mol*const.m_H
    C_p = gamma/(gamma-1.0)*const.RR_gas/mu_mol
    C_v = 1.0/(gamma-1.0)*const.RR_gas/mu_mol
    ## Flux
    #F_conv, effconv = flux_c92(z, rho, T, g, m, C_p, C_v, kappaR, ell)
    F_conv, effconv = flux_hk94(z, rho, T, g, m, C_p, C_v, kappaR, ell)

    ## Heat sources
    ## - (C_B/2)*Bzz
    dlogP_dz = util.grads(log(rho*T), z)
    C_B = 0.6
    S_conv = 0.5*C_B * 2.0*(gamma-1.0)/gamma * F_conv * dlogP_dz / rho
    ## - d(F_conv)/dz
    S_conv -= util.grad(F_conv, z) / rho

    ##
    ## Variations of the flux and heat

    dT = 1.0
    dFconv_dT = np.empty_like(F_conv)

    ## Even
    T_var = T.copy()
    T_var[0::2] += dT
    gradT_var = util.grads(T_var, z)
    kappaR_var = opac.fn_kappaR(T_var)
    #F_conv_var, _ = flux_c92(z, rho, T_var, g, m, C_p, C_v, kappaR_var, ell)
    F_conv_var, _ = flux_hk94(z, rho, T_var, g, m, C_p, C_v, kappaR_var, ell)
    dFconv_dT[0::2] = (F_conv_var - F_conv)[0::2] / dT

    ## Odd
    T_var = T.copy()
    T_var[1::2] += dT
    gradT_var = util.grads(T_var, z)
    kappaR_var = opac.fn_kappaR(T_var)
    #F_conv_var, _ = flux_c92(z, rho, T_var, g, m, C_p, C_v, kappaR_var, ell)
    F_conv_var, _ = flux_hk94(z, rho, T_var, g, m, C_p, C_v, kappaR_var, ell)
    dFconv_dT[1::2] = (F_conv_var - F_conv)[1::2] / dT

    ##
    dSconv_dT = 0.5*C_B * 2.0*(gamma-1.0)/gamma * dFconv_dT * dlogP_dz / rho
    dSconv_dT -= util.grad(dFconv_dT, z) / rho

    return F_conv, effconv, S_conv, dSconv_dT



@njit
def flux_c92(z, rho, T, g, m, C_p, C_v, kappaR, ell):
    """
    Canuto (1992)
    """

    ## Temperature gradient excess
    dbeta = - ( util.grads(T, z) - g/C_p )
    dbeta[dbeta < 0] = 0

    ## Radiative heat conductivity
    varkappa_rad = 4*const.c*const.a_rad*T**3 / (3*rho * kappaR)
    ## Neutral-neutral (H_2) collision section [cm2]
    sigma_nn = 3e-16
    ## Thermal velocity [cm s-1]
    v_th = sqrt(3*const.k_B/m * T)
    ## Molecular heat conductivity [erg cm-1 s-1 K-1]
    varkappa_mol = (1/3) * m*C_v * v_th/sigma_nn
    ## Laminar heat conductivity [erg cm-1 s-1 K-1]
    varkappa_lam = varkappa_rad + varkappa_mol
    ## Laminar thermometric conductivity [cm2 s-1]
    nu_lam = varkappa_lam / (rho*C_p)
    ## Convection efficiency
    Y = ell**4 / (nu_lam**2 * T) * (- g * dbeta)

    Psi = np.where(Y > 0.01, (sqrt(1+Y) - 1)**3/Y, Y**2/8 - 3*Y**3/32)
    Fconv = varkappa_lam * Psi * dbeta
    Fconv[~(Y > 0)] = 0

    return Fconv, Y



@njit
def flux_hk94(z, rho, T, g, m, C_p, C_v, kappaR, ell):
    """
    Hansen & Kawaler (1994)
    """

    ## Temperature gradient excess
    dbeta = - ( util.grads(T, z) - g/C_p )
    dbeta[dbeta < 0] = 0

    ## Radiative heat conductivity
    varkappa_rad = 4*const.c*const.a_rad*T**3 / (3*rho * kappaR)
    ## Neutral-neutral (H_2) collision section [cm2]
    sigma_nn = 3e-16
    ## Thermal velocity [cm s-1]
    v_th = sqrt(3*const.k_B/m * T)
    ## Molecular heat conductivity [erg cm-1 s-1 K-1]
    varkappa_mol = (1/3) * m*C_v * v_th/sigma_nn
    ## Laminar heat conductivity [erg cm-1 s-1 K-1]
    varkappa_lam = varkappa_rad + varkappa_mol
    ## Laminar thermometric conductivity [cm2 s-1]
    nu_lam = varkappa_lam / (rho*C_p)
    ## Convection efficiency
    Y = ell**4 / (nu_lam**2 * T) * (- g * dbeta)

    ## Relative increment, see (5.42)
    X = np.where(Y > 0.01, (sqrt(4*Y+1) - 1)/(2*sqrt(Y)), sqrt(Y)*(1 - Y + 2*Y**2))
    ## Velocity of the blob [cm s-1]
    vconv = X * sqrt(np.abs(g) * dbeta/T) * ell
    ## Temperature difference between the blob and the environment [K]
    dT = X**2 * ell * dbeta
    Fconv = vconv * rho*C_p * dT
    Fconv[~(Y > 0)] = 0

    return Fconv, Y
