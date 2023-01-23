# -*- coding: utf-8 -*-
"""
... using [NumbaLSODA](https://github.com/Nicholaswogan/numblsoda)
"""

import numpy as np
from numpy import sqrt, log

import numba
from numba import njit, types

import numbalsoda

from . import const
from . import util



def init(n_nod, use_diffusion=True):
    """
    Initialize data structure and right-hand-side function.
    """

    TArr = types.NestedArray(dtype=types.float64, shape=(n_nod,))
    TParams = types.Record.make_c_struct([
        ## Code parameters
        ('n_nod',         types.int_),
        ('use_diffusion', types.int_),
        ## Boosts
        ('omega',   types.float64),
        ('omega_1', types.float64),
        ('omega_2', types.float64),
        ('Srf',     types.float64),
        ('Vrf',     types.float64),
        ## Buoyancy
        ('Bzz',     TArr),
        ## Fields
        ('rho',     TArr),
        ('cs2',     TArr),
        ('dzeta',   types.NestedArray(dtype=types.float64, shape=(n_nod-1,))),
        ('dzeta_',  TArr),
    ])


    @numba.cfunc(types.void(types.float64,
                            types.CPointer(types.float64),
                            types.CPointer(types.float64),
                            types.CPointer(TParams)))
    def crhs(t, X, dotX, p_):
        p    = numba.carray(p_,   (1,))[0]
        w    = numba.carray(X,    (6, p.n_nod))
        dotw = numba.carray(dotX, (6, p.n_nod))
        rhs(t, w, dotw, p)


    p = np.recarray((1,), dtype=TParams)
    p.n_nod = n_nod
    p.use_diffusion = use_diffusion

    return p, crhs



@njit
def rhs(t, w, dotw, p):
    dotw_1 = np.empty_like(dotw)
    rhs_cross(t, w, dotw_1, p)

    dotw_2 = np.zeros_like(dotw)
    if p.use_diffusion > 0:
        rhs_diffusion(t, w, dotw_2, p)

    ## Doesn't work another way
    for i in range(6):
        dotw[i] = dotw_1[i] + dotw_2[i]



@njit
def rhs_cross(t, w, dotw, p):
    """
    """

    wrr, wff, wzz, wrf, wrz, wfz = w[0], w[1], w[2], w[3], w[4], w[5]

    ## Code parameters
    C_B, C_Pi1, C_Pi2, C_Pi3 = 0.6, 3.5, 0.61, 0.44
    #C_eps1, C_eps2, C_eps3, C_eps4 = 1.44, 1.83, 0.15, 0.1

    ## Boosts
    omega   = p.omega
    omega_1 = p.omega_1
    omega_2 = p.omega_2
    Srf = p.Srf
    Vrf = p.Vrf
    ## Buoyancy
    Bzz = p.Bzz
    ## Fields
    rho = p.rho
    cs2 = p.cs2
    #gamma = p.gamma

    ## Turbulent energy
    K = 0.5*(wrr + wff + wzz)
    K[K < 0] = 0.0
    ## Inverse correlation time
    M_T = np.sqrt(2*K/(rho*cs2))
    inv_t_T = M_T / sqrt(1 + M_T**2) * np.abs(omega)

    Pi_rr = C_Pi1*inv_t_T*(2*wrr - wff - wzz)/3 - (2*C_Pi2/3*Srf - 2*C_Pi3*Vrf)*wrf
    Pi_ff = C_Pi1*inv_t_T*(2*wff - wzz - wrr)/3 - (2*C_Pi2/3*Srf + 2*C_Pi3*Vrf)*wrf
    Pi_zz = C_Pi1*inv_t_T*(2*wzz - wrr - wff)/3 + 4*C_Pi2/3*Srf*wrf + (1-C_B)*Bzz
    Pi_rf = C_Pi1*inv_t_T*wrf - C_Pi2*Srf*(wrr + wff - 2*wzz)/3 - C_Pi3*Vrf*(wrr - wff) - 4*Srf/5*K
    Pi_rz = C_Pi1*inv_t_T*wrz - (C_Pi2*Srf - C_Pi3*Vrf)*wfz
    Pi_fz = C_Pi1*inv_t_T*wfz - (C_Pi2*Srf + C_Pi3*Vrf)*wrz

    eps = 2*K*inv_t_T
    #eps = C_eps1/C_eps2 * np.abs(omega_1 * wrf)

    dotw[0] = - Pi_rr + 4*omega*wrf - 2/3*eps
    dotw[1] = - Pi_ff - 2*omega_2*wrf - 2/3*eps
    dotw[2] = - Pi_zz + Bzz - 2/3*eps
    dotw[3] = - Pi_rf - omega_2*wrr + 2*omega*wff
    dotw[4] = - Pi_rz + 2*omega*wfz
    dotw[5] = - Pi_fz - omega_2*wrz - omega*wfz



@njit
def rhs_diffusion(t, w, dotw, p):
    """
    """

    wrr, wff, wzz, wrf, wrz, wfz = w[0], w[1], w[2], w[3], w[4], w[5]

    n_nod = p.n_nod
    ## Boost
    omega = p.omega
    ## Fields
    rho    = p.rho
    cs2    = p.cs2
    dzeta  = p.dzeta
    dzeta_ = p.dzeta_

    C_nu = 0.09

    ## Turbulent energy
    K = 0.5*(wrr + wff + wzz)
    K[K < 0] = 0.0
    ## Inverse correlation time
    M_T = np.sqrt(2*K/(rho*cs2))
    inv_t_T = M_T / sqrt(1 + M_T**2) * abs(omega)
    ## Diffusion coefficient
    nu_T = C_nu * 0.5/abs(omega) * M_T * sqrt(1 + M_T**2)

    ## Grid coefficients
    phi = 1.0 / dzeta_
    psi = 0.5*(nu_T[:-1] + nu_T[1:]) / dzeta

    ## Fluxes
    F = np.empty((6, n_nod+1))
    for k in range(6):
        ## Inner fluxes
        F[k,1:-1] = - psi * (w[k,1:] - w[k,:-1])
        ## Outer boundary conditions
        F[k,-1]   = - psi[-1] * (0.0 - w[k,-1])

    ## Midplane boundary conditions
    F[0:4,0] = - F[0:4,1]
    F[4:,0]  = F[4:,1]

    ## zz
    F[2] *= 3.0
    ## rz
    F[4] *= 2.0
    ## fz
    F[5] *= 2.0

    for k in range(6):
        dotw[k] = - phi * (F[k,1:] - F[k,:-1])



@njit
def advance_to(pp, crhs_ptr, z, rho, cs2, Bzz, w_0, Omega, Omega_1, Omega_2, dt):
    """
    """

    ## No convection, no turbulence
    if np.all(Bzz == 0.0):
        return np.zeros_like(w_0), True

    p = pp[0]

    ## Boosts
    p.omega   = Omega   / abs(Omega)
    p.omega_1 = Omega_1 / abs(Omega)
    p.omega_2 = Omega_2 / abs(Omega)
    p.Srf = 0.5 * p.omega_1
    p.Vrf = - 0.5 * p.omega_2
    ## Buoyancy
    p.Bzz[:] = Bzz / abs(Omega)
    ## Fields
    p.rho[:] = rho
    p.cs2[:] = cs2

    if p.use_diffusion > 0:
        ## Vertical coordinate in the local thermal scale units
        zeta = abs(Omega)/sqrt(cs2) * z
        dzeta = zeta[1:] - zeta[:-1]
        p.dzeta[:]  = dzeta
        p.dzeta_[:] = np.concatenate(( np.asarray([dzeta[0]]),
                                       0.5*(dzeta[:-1] + dzeta[1:]),
                                       np.asarray([dzeta[-1]]) ))

    ## Integrate with LSODA method
    ds = abs(Omega) * dt
    X, status \
        = numbalsoda.lsoda(crhs_ptr, w_0.ravel(), np.array([0.0, ds]), data=pp)

    w = X[-1].reshape((6, p.n_nod))

    return w, status



@njit
def heat(rho, cs2, w, Omega):

    wrr, wff, wzz = w[0], w[1], w[2]

    ## Turbulent energy
    K = 0.5*(wrr + wff + wzz)
    K[K < 0] = 0.0
    ## Inverse correlation time
    M_T = np.sqrt(2*K/(rho*cs2))
    inv_t_T = M_T / sqrt(1 + M_T**2) * abs(Omega)
    ## Dissipation rate
    eps = 2*K*inv_t_T

    S_turb = eps/rho

    return S_turb
