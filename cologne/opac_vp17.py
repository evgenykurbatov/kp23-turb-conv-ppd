# -*- coding: utf-8 -*-
"""
"""

import os

import numpy as np
from numpy import log, log10

from numba import njit



##
## Load and set-up
##

fname = os.path.abspath(os.path.join(os.path.dirname(__file__), 'opac_vp17.dat.gz'))
T, kappaP, kappaR, kappaF = np.loadtxt(fname, skiprows=1, unpack=True)

lgkappaP = log10(kappaP)
lgkappaR = log10(kappaR)
lgkappaF = log10(kappaF)

dlgkappaP_dT = np.gradient(lgkappaP, T)
dlgkappaR_dT = np.gradient(lgkappaR, T)
dlgkappaF_dT = np.gradient(lgkappaF, T)

@njit
def fn_kappaP(T_):
    return 10**np.interp(T_, T, lgkappaP)

@njit
def fn_kappaR(T_):
    return 10**np.interp(T_, T, lgkappaR)

@njit
def fn_kappaF(T_):
    return 10**np.interp(T_, T, lgkappaF)

@njit
def fn_dkappaP_dT(T_):
    res = 10**np.interp(T_, T, lgkappaP) * log(10.0)*np.interp(T_, T, dlgkappaP_dT)
    res[(T_ < T[0]) | (T_ > T[-1])] = 0.0
    return res

@njit
def fn_dkappaR_dT(T_):
    res = 10**np.interp(T_, T, lgkappaR) * log(10.0)*np.interp(T_, T, dlgkappaR_dT)
    res[(T_ < T[0]) | (T_ > T[-1])] = 0.0
    return res

@njit
def fn_dkappaF_dT(T_):
    res = 10**np.interp(T_, T, lgkappaF) * log(10.0)*np.interp(T_, T, dlgkappaF_dT)
    res[(T_ < T[0]) | (T_ > T[-1])] = 0.0
    return res



##
## The source is executed as a main program
##

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T_ = np.logspace(log10(T[0])-1.0, log10(T[-1])+1.0, 500)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax_ = ax[0]
    ax_.loglog(T_, fn_kappaP(T_), "-o", label=r"$\kappa_\mathrm{P}$")
    ax_.loglog(T_, fn_kappaR(T_), "-o", label=r"$\kappa_\mathrm{R}$")
    ax_.loglog(T_, fn_kappaF(T_), "-o", label=r"$\kappa_\mathrm{F}$")
    ax_.legend()
    ax_.set_xlabel(r"$T$ [K]")
    ax_.set_ylabel(r"[cm$^2/$g]")

    ax_ = ax[1]
    ax_.loglog(T_, fn_dkappaP_dT(T_), "-o", label=r"$d\kappa_\mathrm{P}/dT$")
    ax_.loglog(T_, fn_dkappaR_dT(T_), "-o", label=r"$d\kappa_\mathrm{R}/dT$")
    ax_.loglog(T_, fn_dkappaF_dT(T_), "-o", label=r"$d\kappa_\mathrm{F}/dT$")
    ax_.legend()
    ax_.set_xlabel(r"$T$ [K]")
    ax_.set_ylabel(r"[cm$^2/$g$/$K]")

    plt.tight_layout()
    plt.show()
