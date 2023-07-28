# -*- coding: utf-8 -*-
"""
"""

from math import sqrt
#from numpy import sqrt

from cologne import const


##
## Global parameters

## Stellar mass [g]
M_star = const.M_sol
## Stellar luminosity [erg s-1]
R_star = const.R_sol
## Stellar temperature [K]
T_star = 5780.0
## Stellar accretion rate [g s-1]
dotM = 1e-7 * const.M_sol/const.yr

## Temperature of interstellar radiation [K]
T_isr = 1e4

## Radial distance to the column [cm]
R = 3.34 * const.AU
## Keplerian frequeny at `R`
Omega = sqrt(const.G*M_star/R**3)

##
## Thermodynamical gas parameters

## Mean molecular weight
mu_mol = 2.3
## Adiabatic exponent
gamma = 7.0/5.0
## Specific heat [erg g-1 K-1]
C_p = gamma/(gamma-1)*const.RR_gas/mu_mol
C_v = 1/(gamma-1)*const.RR_gas/mu_mol

## Column surface density [g cm-2]
#sigma_max = 5.0
sigma_max = 271.145
## Density on the upper bound [g cm-3]
rho_ext = mu_mol*const.m_H * 1e3
#rho_ext = 1.1e-18



##
## The source is executed as a main program
##

if __name__ == "__main__":
    print()
    print("# Global parameters")
    print("M_star = %.2e [g] = %g [M_sol]" % (M_star, M_star/const.M_sol))
    print("T_star = %g [K]" % T_star)
    print("dotM = %.2e [g s-1] = %.2e [M_sol yr-1]" % (dotM, dotM/(const.M_sol/const.yr)))
    print()
    print("R = %.2e [cm] = %g [AU]" % (R, R/const.AU))
    print()
    print("# Thermodynamical gas parameters")
    print("mu_mol = %g" % mu_mol)
    print("gamma = %g" % gamma)
    print("C_p = %.2e [erg g-1 K-1]" % C_p)
    print("C_v = %.2e [erg g-1 K-1]" % C_v)
    print("sigma_max = %.2e [g cm-2]" % sigma_max)
    print("rho_ext = %.2e [g cm-3]" % rho_ext)
