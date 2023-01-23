# -*- coding: utf-8 -*-
"""
Model for 1-D radiative transfer and thermal balance
"""

import numpy as np

from numba import njit

from . import const
from . import util



@njit
def advance_lin(sigma, rho, T, T_0, E, E_0, S, dS_dT, C_v, kappaP, kappaR, dt):
    """
    NB: Midplane is always on `sigma[0]` and the outer boundary is always on `sigma[-1]`.
    """

    ## Surface density may increase both from the midplane and outside
    s = 1.0  if sigma[-1] > sigma[0]  else -1.0

    Tcmb = 2.73
    Ecmb = const.a_rad*Tcmb**4

    aT4 = const.a_rad*T**4
    ckappaP = const.c*kappaP

    dsigma  = s * (sigma[1:] - sigma[:-1])

    dsigma_ = np.empty_like(sigma)
    dsigma_[0]    = dsigma[0]
    dsigma_[1:-1] = 0.5*(dsigma[:-1] + dsigma[1:])
    dsigma_[-1]   = dsigma[-1]

    kappaR_ = 0.5*(kappaR[:-1] + kappaR[1:])

    phi = 1.0/dsigma_
    psi = const.c/(3.0*kappaR_*dsigma)

    ## Lower diagonal
    A = - phi[1:] * psi
    A[-1] *= 2.0
    ## Upper diagonal
    C = - phi[:-1] * psi
    C[0] *= 2.0
    ## Diagonal
    B0 = np.empty_like(phi)
    B0[0]    = 2.0*phi[0] * psi[0]
    B0[1:-1] = phi[1:-1] * (psi[:-1] + psi[1:])
    B0[-1]   = phi[-1] * (2.0*psi[-1] + const.c)
    B0 += ckappaP
    ## Source
    D0 = - 3.0*ckappaP*aT4
    D0[-1] += phi[-1] * const.c*Ecmb

    while True:
        ## Components for temperature operator
        AA = 4.0*ckappaP*aT4 - dS_dT*T
        BB = C_v*T/dt + AA
        CC = C_v*T_0/dt + AA - ckappaP*aT4 + S

        B = B0 + 1.0/(rho*dt) - 4.0*ckappaP*aT4 * ckappaP / BB
        D = D0 + E_0/(rho*dt) + 4.0*ckappaP*aT4 * CC / BB

        ## Solve
        E = util.tdma(A, B, C, D)

        ## Adjust the time step if necessary
        f = 1.1
        while np.any((E < 0.0) | (E > f*E_0) | (E < E_0/f)):
            dt *= 0.5
            BB = C_v*T/dt + AA
            CC = C_v*T_0/dt + AA - ckappaP*aT4 + S
            B  = B0 + 1.0/(rho*dt) - 4.0*ckappaP*aT4 * ckappaP / BB
            D  = D0 + E_0/(rho*dt) + 4.0*ckappaP*aT4 * CC / BB
            E  = util.tdma(A, B, C, D)

        ## Calculate the temperature
        CC_ = CC + ckappaP*E
        T_new = CC_ / BB * T

        ## Adjust the time step if necessary
        if np.any( (BB < 0.0) | (CC_ < 0.0) | (CC_ > f*BB) | (CC_ < BB/f) ):
            dt *= 0.5
            continue

        break

    T = T_new

    ## Calculate flux for diagnostic purposes
    F = - s * const.c/(3.0*kappaR) * util.grads(E, sigma)

    return T, E, F, dt



@njit
def advance2_lin(sigma, rho, T, T_0, E, E_0, C_v, heating_callback, kappaP, kappaR, dt):
    """
    NB: Midplane is always on `sigma[0]` and the outer boundary is always on `sigma[-1]`.
    """

    ## Surface density may increase both from the midplane and outside
    s = 1.0  if sigma[-1] > sigma[0]  else -1.0

    Tcmb = 2.73
    Ecmb = const.a_rad*Tcmb**4

    aT4 = const.a_rad*T**4
    ckappaP = const.c*kappaP

    dsigma  = s * (sigma[1:] - sigma[:-1])

    dsigma_ = np.empty_like(sigma)
    dsigma_[0]    = dsigma[0]
    dsigma_[1:-1] = 0.5*(dsigma[:-1] + dsigma[1:])
    dsigma_[-1]   = dsigma[-1]

    kappaR_ = 0.5*(kappaR[:-1] + kappaR[1:])

    phi = 1.0/dsigma_
    psi = const.c/(3.0*kappaR_*dsigma)

    ## Lower diagonal
    A = - phi[1:] * psi
    A[-1] *= 2.0
    ## Upper diagonal
    C = - phi[:-1] * psi
    C[0] *= 2.0
    ## Diagonal
    B0 = np.empty_like(phi)
    B0[0]    = 2.0*phi[0] * psi[0]
    B0[1:-1] = phi[1:-1] * (psi[:-1] + psi[1:])
    B0[-1]   = phi[-1] * (2.0*psi[-1] + const.c)
    B0 += ckappaP
    ## Source
    D0 = - 3.0*ckappaP*aT4
    D0[-1] += phi[-1] * const.c*Ecmb

    while True:
        ## Heating
        S, dS_dT = heating_callback(T)

        ## Components for temperature operator
        AA = 4.0*ckappaP*aT4 - dS_dT*T
        BB = C_v*T/dt + AA
        CC = C_v*T_0/dt + AA - ckappaP*aT4 + S

        B = B0 + 1.0/(rho*dt) - 4.0*ckappaP*aT4 * ckappaP / BB
        D = D0 + E_0/(rho*dt) + 4.0*ckappaP*aT4 * CC / BB

        ## Solve
        E = util.tdma(A, B, C, D)

        ## Adjust the time step if necessary
        f = 1.1
        while np.any((E < 0.0) | (E > f*E_0) | (E < E_0/f)):
            dt *= 0.5
            BB = C_v*T/dt + AA
            CC = C_v*T_0/dt + AA - ckappaP*aT4 + S
            B  = B0 + 1.0/(rho*dt) - 4.0*ckappaP*aT4 * ckappaP / BB
            D  = D0 + E_0/(rho*dt) + 4.0*ckappaP*aT4 * CC / BB
            E  = util.tdma(A, B, C, D)

        ## Calculate the temperature
        CC_ = CC + ckappaP*E
        T_new = CC_ / BB * T

        ## Adjust the time step if necessary
        if np.any( (BB < 0.0) | (CC_ < 0.0) | (CC_ > f*BB) | (CC_ < BB/f) ):
            dt *= 0.5
            continue

        break

    T = T_new

    ## Calculate flux for diagnostic purposes
    F = - s * const.c/(3.0*kappaR) * util.grads(E, sigma)

    return T, E, F, dt
