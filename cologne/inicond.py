# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi, exp, log10, sqrt

import scipy as sp
import scipy.optimize

from . import const



def sigma_grid(n_nod, sigma_max, sigma_out, inverse=False):
    """
    Makes logarithmic grid of the surface density values from `0` to `sigma_max`.

    Parameters
    ----------
    n_nod : int
        Number of nodes.
    sigma_max : float
        Max value.
    sigma_out : float
        Approx. the outer cell of the grid.
        E.g. `sigma_out = tau_out/kappaP` for `tau_out` is an optical depth of the outer cell.
    inverse : bool, optional
        Whether to invert the grid.
        If `True`, then `sigma[0] == sigma_max`. Default is `False`.

    Returns
    -------
    sigma : ndarray (n_nod,)
        Grid of nodes.
    """

    ## Surface density grid [g cm-2]
    sigma = np.empty(n_nod)
    p = (np.arange(n_nod) / (n_nod-1))**1.0
    tmp = sigma_out * (sigma_max/sigma_out + 1)**p[::-1]
    sigma = tmp - sigma_out
    sigma[0]  = sigma_max
    sigma[-1] = 0.0

    if inverse:
        return sigma
    else:
        return sigma_max - sigma



def isothermal(sigma, T_00, rho_ext, mu_mol, Omega, verbose=False):
    """
    """

    if verbose:  print("inicond:")

    sigma_max = max(sigma)
    ## Surface density may increase both from the midplane and outside
    if sigma[-1] > sigma[0]:
        sigma_in = sigma
    else:
        sigma_in = sigma_max - sigma

    ## Isothermal sound speed [cm s-1]
    c_T_00 = sqrt(const.RR_gas/mu_mol * T_00)
    ## Characteristic vertical scale [cm]
    H = c_T_00/Omega

    ##
    ## Isothermal distribution

    ## Vertical distance of the upper bound [cm]
    f = lambda zeta : exp(zeta**2) * sp.special.erf(zeta) \
        - sqrt(2/pi) * sigma_max/(H*rho_ext)
    fprime  = lambda zeta : 2*zeta * exp(zeta**2) * sp.special.erf(zeta) + 2/sqrt(pi)
    z_ext = sqrt(2)*H * sp.optimize.newton(f, 5, fprime=fprime)
    ## Midplane density (sould be same values) [g cm-3]
    #rho_c = rho_ext * exp((z_ext/H)**2/2)
    rho_c = sqrt(2/pi) * sigma_max/H / sp.special.erf(z_ext/(sqrt(2)*H))
    ## Vertical distance grid [cm]
    z = sqrt(2)*H * sp.special.erfinv( sqrt(2/pi) * sigma_in/(rho_c*H) )
    ## Density distribution at the nodes [g cm-3]
    rho = rho_c * exp(-0.5*(z/H)**2)
    ## Initial temperature distribution [K]
    T = T_00 * np.ones_like(rho)

    if verbose:
        print("  T_00 = %g [K]" % T_00)
        print("  H = %.2e [cm] = %g [AU]" % (H, H/const.AU))
        print("  z[-1] = %.2e [cm] = %.2e [AU] = %.2e H" \
              % (z[-1], z[-1]/const.AU, z[-1]/H))
        print("  rho[0]  = %.2e [g cm-3]" % rho[0])
        print("  rho[-1] = %.2e [g cm-3]" % rho[-1])

    if verbose:  print("inicond: done")
    return z, rho, T



##
## The source is executed as a main program
##

if __name__ == "__main__":
    pass
