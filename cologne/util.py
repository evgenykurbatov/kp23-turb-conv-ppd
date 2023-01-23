# -*- coding: utf-8 -*-

import numpy as np

from numba import njit



@njit
def grad(y, x):
    dy_dx = np.empty_like(x)

    dx = x[1:] - x[:-1]
    dy_dx[0] = (y[1] - y[0]) / dx[0]
    dy_dx[1:-1] \
        = ( dx[:-1]**2 * y[2:] + (dx[1:]**2 - dx[:-1]**2) * y[1:-1] - dx[1:]**2 * y[:-2] ) \
        / ( dx[:-1] * dx[1:] * (dx[:-1] + dx[1:]) )
    dy_dx[-1] = (y[-1] - y[-2]) / dx[-1]

    return dy_dx



@njit
def grads(y, x):
    """
    dy/dx=0  at  x=x[0]
    """

    dy_dx = np.empty_like(x)

    dx = x[1:] - x[:-1]
    dy_dx[0] = 0.0
    dy_dx[1:-1] \
        = ( dx[:-1]**2 * y[2:] + (dx[1:]**2 - dx[:-1]**2) * y[1:-1] - dx[1:]**2 * y[:-2] ) \
        / ( dx[:-1] * dx[1:] * (dx[:-1] + dx[1:]) )
    dy_dx[-1] = (y[-1] - y[-2]) / dx[-1]

    return dy_dx



@njit
def laplace(x, a=None):
    """
    Calculates matrix of Laplace operator on `x` where `a` is the diffusion
    coefficient:
    L[u] = (d/dx)[a du/dx]
    Inner boundary condition:
    du/dx=0  at  x=x[0]
    Outer boundary condition:
    du/dx=0  at  x=x[-1]

    Returns
    -------
    A, B, C : array
        Diagonals of the three-diagonal matrix.
    """

    dx  = x[1:] - x[:-1]

    if a is None:
        a_ = np.ones_like(dx)
    else:
        a_ = 0.5*(a[:-1] + a[1:])

    dx_ = np.empty_like(x)
    dx_[0]    = dx[0]
    dx_[1:-1] = 0.5*(dx[:-1] + dx[1:])
    dx_[-1]   = dx[-1]

    phi = 1.0/dx_
    psi = a_/dx

    ## Lower diagonal
    A = phi[1:] * psi
    ## Upper diagonal
    C = phi[:-1] * psi
    C[0] *= 2.0
    ## Diagonal
    B = np.empty_like(x)
    B[0]    = - 2.0 * phi[0] * psi[0]
    B[1:-1] = - phi[1:-1] * (psi[:-1] + psi[1:])
    B[-1]   = - 2.0 * phi[-1] * psi[-1]

    return A, B, C



@njit
def tdma(a, b, c, d):
    """
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """
    nf = len(d)
    ac = a.copy()
    bc = b.copy()
    cc = c.copy()
    dc = d.copy()
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc
