# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from numpy import pi

from numba import njit

from . import const
from . import util



@njit
def adjust(sigma, rho_0, c_T2, rho_ext, Omega_K2, wzz=None, selfgrav=0.0,
           rtol=1e-4, max_iter=10000):
    """
    NB: Midplane is always on `sigma[0]` and the outer boundary is always on `sigma[-1]`.

    Parameters
    ----------
    sigma : array_like
        Surface density at the node levels.
    rho_0 : array_like
        Density, the seed values for iterations.
    c_T2 : array_like
        Square of isothermal sound speed.
    Omega_K2 : float
        Square of local Keplerian frequency.
    wzz : array_like or None
        External stress in the cells. Default is no stress.
    selfgrav : float, optional
        Whether self-gravity should be taken into account, 1.0 or 0.0 (default).
    rtol : array_like or scalar, optional
        Relative tolerance for convergence condition for `rho`. Default is 1e-4.
    max_iter : int, optional
        Max number of iterations. Default is 10000.

    Returns
    -------
    z : ndarray
        Nodes.
    rho : ndarray
        Density in the cells.
    """

    ## Surface density may increase both from the midplane and outside
    s = 1  if sigma[-1] > sigma[0]  else -1

    rho_0_ = 0.5*(rho_0[:-1] + rho_0[1:])

    z_0 = np.empty_like(sigma)
    z_0[0]  = 0.0
    ## TODO: use smarter integration
    z_0[1:] = np.cumsum(s*(sigma[1:] - sigma[:-1]) / rho_0_)

    dg_dz = - Omega_K2 * np.ones_like(sigma)

    ##
    ## Get the Laplace operator

    dsigma  = sigma[1:] - sigma[:-1]

    dsigma_ = np.empty_like(dsigma)
    dsigma_[0]  = dsigma[0]
    dsigma_[1:] = 0.5*(dsigma[:-1] + dsigma[1:])

    phi = 1.0/dsigma_
    psi = 1.0/dsigma

    ## Lower diagonal
    A = phi[1:] * psi[:-1] * c_T2[:-2]
    ## Upper diagonal
    C = phi[:-1] * psi[:-1] * c_T2[1:-1]
    C[0] *= 2.0
    ## Diagonal
    B0 = np.empty_like(dsigma)
    B0[0]  = - 2.0 * phi[0] * psi[0]            * c_T2[0]
    B0[1:] = - phi[1:] * (psi[:-1] + psi[1:]) * c_T2[1:-1]

    ##
    ## Get the r.h.s.

    ## Self-gravity
    D0 = - 4.0*pi*const.G*selfgrav * np.ones_like(dsigma)
    ## Outer boundary condition
    D0[-1] += - phi[-1] * psi[-1] * c_T2[-1] * rho_ext
    ## Additional stress
    if wzz is not None:
        D0[0]  -= 2.0 * phi[0] * psi[0] * (wzz[1] - wzz[0])
        D0[1:] -= phi[1:] * psi[:-1]             * wzz[:-2] \
                - phi[1:] * (psi[:-1] + psi[1:]) * wzz[1:-1] \
                + phi[1:] * psi[1:]              * wzz[2:]


    ## Start iterations
    rho = rho_0.copy()
    rho_new = np.empty_like(rho_0)
    rho_new[-1] = rho_ext
    n_iter = 0
    while n_iter < max_iter:

        B = B0 + dg_dz[:-1] / rho[:-1]**2
        D = D0 + dg_dz[:-1] * 2.0/rho[:-1]
        tmp = util.tdma(A, B, C, D)
        rho_new[:-1] = tmp.copy()

        ## Convergence condition
        if np.all( np.abs(rho_new[:-1] - rho[:-1]) <= rtol*rho[:-1] ):
            break

        rho = rho_new.copy()

        n_iter += 1
        if n_iter >= max_iter:
            raise ValueError("n_iter >= max_iter")

    rho_new_ = 0.5*(rho_new[:-1] + rho_new[1:])

    z_new = np.empty_like(sigma)
    z_new[0]  = 0.0
    ## TODO: use smarter integration
    z_new[1:] = np.cumsum(s*(sigma[1:] - sigma[:-1]) / rho_new_)

    return z_new, rho_new



##
## The source is executed as a main program
##

if __name__ == "__main__":
    pass
