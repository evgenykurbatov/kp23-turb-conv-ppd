# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import gzip

import pandas as pd

import numpy as np
np.set_printoptions(precision=3, suppress=False)

import scipy as sp
import scipy.integrate

import matplotlib as mpl
import matplotlib.colors
from  matplotlib import pyplot as plt

import cologne
from cologne import const
from param import *

from numpy import pi, sqrt, exp, sin, cos, tan, log, log10



##
## Read
##

fname = sys.argv[1]
columns = [ 'sigma', 'z', 'rho', 'T', 'E', 'F_rad', 'S_uv', 'S_ext',
            'F_conv', 'effconv', 'S_conv' ]
with gzip.open(fname, 'rt') as f:
    t = float(f.readline())
    f = pd.read_csv(f, header=None, skiprows=1, names=columns,
                    skip_blank_lines=True, sep=" ", skipinitialspace=True)
print("f.shape =", f.shape)
print(f.head())



##
## Choose a column
##

print("z_max = %g [AU]" % (f['z'].max()/const.AU))
print("z_max = %g [AU]" % (f['z'].values[-1]/const.AU))
print("sigma_tot = %g [g/cm2]" % f['sigma'].max())
print("sigma_tot = %g [g/cm2]" % f['sigma'].values[-1])


##
## Convenient additional variables

f['Omega'] = Omega * np.ones_like(f['rho'])

## Gravitational acceleration (NB: it's signed)
f['g'] = - f['Omega']**2 * f['z']

f['p'] = const.RR_gas/mu_mol * f['rho']*f['T']
f['cs2'] = gamma * const.RR_gas/mu_mol * f['T']

##
#f['dFconv_dz'] = np.gradient(f['F_conv'], f['z'])

f['nabla'] = np.gradient(np.log(f['T']), np.log(f['p']))
f['nabla_ad'] = (gamma - 1)/gamma

## No convection
f['S'] = f['S_uv'] + f['S_ext']


"""
##
## Optical depth of the column

kappa_R = f['kappa'].values
sigma = f['sigma'].values

tau_R = np.array([ sp.integrate.simpson(kappa_R[i:], sigma[i:]) \
                   for i in range(len(sigma)) ])
f['tau'] = tau_R
"""


##
## Vertical scales

## 'H_sigma' scale contains 90% of a column mass
j_sigma = int((f['sigma'] < 0.90*f['sigma'].max()).argmin())
f['H_sigma'] = f.iloc[j_sigma]['z']
print("H_sigma = %g AU" % (f['H_sigma'].unique()/const.AU))

## 'H_th' scale is an isothermal scale
cs = sqrt(f['cs2'].values)
f['H_th'] = cs[0] / np.abs(f['Omega'].values[0])
print("H_th = %g AU" % (f['H_th'].unique()/const.AU))


"""
##
## Turbulent Mach number

K = 0.5*(f['wrr'] + f['wff'] + f['wzz']).values
K[K < 0] = 0
M_T = np.sqrt(2*K/(f['rho'].values * cs**2))

Wrf = sp.integrate.simpson(f['wrf'], f['z'])
nu_SS = Wrf / (f['sigma'].values[-1] * np.abs(f['Omega_1'].values[-1]))
alpha_SS = nu_SS / (f['H_th'].values[0] * cs[0])
print("sigma_tot = %g [g/cm2]" % f['sigma'].values[-1])
print("H_th = %g [AU]" % (f['H_th'].values[0]/const.AU))
print("alpha_SS =", alpha_SS)
"""



##
## Plot
##



## rc settings (see http://matplotlib.sourceforge.net/users/customizing.html#customizing-matplotlib)
mpl.rc('font', family='sans')
mpl.rc('font', size='8.0')
#mpl.rc('text', usetex=True)
mpl.rc('lines', linewidth=1.0)
mpl.rc('axes', linewidth=0.5)
mpl.rc('legend', frameon=False)
mpl.rc('legend', handlelength=2.5)

figwidth = 16.0 / 2.54           ## convert cm to in
figheight = 24.0 / 2.54          ## convert cm to in
mpl.rc('figure', figsize=[figwidth, figheight])

fig, ax = plt.subplots(nrows=4, ncols=2)

#plt.suptitle(("'%s': " % fname) \
#             + (r"$r_\ast = %g$ [AU], $\alpha_\mathrm{SS} = %.2e$" % (r_ast/const.AU, alpha_SS)))
plt.suptitle(r"$t = %g$ [yr]" % (t/const.yr))

z = f['z'].values

ax_ = ax[0,0]
"""
ax_.semilogy(z/const.AU, f['F_rad'], label=r"$F_\mathrm{rad}$")
ax_.semilogy(z/const.AU, f['F_conv'], label=r"$F_\mathrm{conv}$")
"""
ax_.loglog(z/const.AU, f['F_rad'], label=r"$F_\mathrm{rad}$")
ax_.loglog(z/const.AU, f['F_conv'], label=r"$F_\mathrm{conv}$")
ax_.set_ylabel(r"erg$/$cm$^2/$s")
ax_.legend()

ax_ = ax[0,1]
ax_.loglog(z/const.AU, f['effconv'])
ax_.axhline(1, ls=':', c='k')
ax_.set_ylabel(r"Convection efficiency")

"""
ax_ = ax[0,1]
ax_.semilogx(z/const.AU, f['nabla'], label=r"$\nabla$")
ax_.semilogx(z/const.AU, f['nabla_ad'], label=r"$\nabla_\mathrm{ad}$")
ax_.legend()
"""

ax_ = ax[1,0]
#ax_.loglog(z/const.AU, f['p']/f['rho']**gamma)
ax_.loglog(z/const.AU, np.abs(np.gradient(f['p']/f['rho']**gamma, z)))
ax_.set_ylabel(r"$|(d/dz) p/\rho^\gamma|$ [\dots]")
"""

ax_ = ax[1,0]
ax_.loglog(z/const.AU, f['rho']*f['cs2'])
ax_.set_ylabel(r"$\rho c_\mathrm{s}^2$ [erg cm$^{-3}$]")

ax_ = ax[1,0]
ax_.loglog(z/const.AU, f['tau'])
ax_.axhline(1, ls=':', c='k')
ax_.set_ylabel(r"$\tau_\mathrm{R}$")

ax_ = ax[1,1]
ax_.loglog(z/const.AU, f['kappa'])
ax_.set_ylabel(r"$\kappa_\mathrm{R}$ [cm$^2/$g]")
ax_ = ax[1,1]
ax_.loglog(f['z']/const.AU, (f['sigma'].values[-1] - f['sigma'])/(mu*const.m_H), label=r"$\int dz\,n$")
ax_.set_ylabel(r"cm$^{-2}$")
ax_.legend()
"""

ax_ = ax[1,1]
ax_.loglog(z/const.AU, f['T'])
ax_.set_ylabel(r"$T$ [K]")

"""
ax_ = ax[2,0]
ax_.semilogx(z/const.AU, f['wrr'], label=r"$w_{rr}$")
ax_.semilogx(z/const.AU, f['wff'], label=r"$w_{\phi\phi}$")
ax_.semilogx(z/const.AU, f['wzz'], label=r"$w_{zz}$")
ax_.semilogx(z/const.AU, f['wrf'], label=r"$w_{r\phi}$")
print("min(wrf) =", np.min(f['wrf'].values))
#ax_.set_ylim(ymin=1e-1*np.min(f['wrf'].values))
ax_.set_ylabel(r"erg$/$cm$^3$")
ax_.legend()

ax_ = ax[2,1]
ax_.semilogx(z/const.AU, M_T)
#ax_.axhline(1, ls=':', c='k')
ax_.set_ylabel(r"$\mathcal{M}_\mathrm{T}$")
"""

ax_ = ax[3,0]
ax_.plot(f['z']/const.AU, f['rho']*f['S_uv'], label=r"$\rho S_\mathrm{uv}$")
ax_.plot(f['z']/const.AU, f['rho']*f['S_ext'], label=r"$\rho S_\mathrm{ext}$")
ax_.plot(f['z']/const.AU, f['rho']*f['S_conv'], label=r"$\rho S_\mathrm{conv}$")
ax_.plot(f['z']/const.AU, f['rho']*f['S'], 'k', label=r"$\rho S$")
"""
ax_.semilogx(f['z']/const.AU, f['rho']*f['S_uv'], label=r"$\rho S_\mathrm{uv}$")
ax_.semilogx(f['z']/const.AU, f['rho']*f['S_ext'], label=r"$\rho S_\mathrm{ext}$")
ax_.semilogx(f['z']/const.AU, f['rho']*f['S_conv'], label=r"$\rho S_\mathrm{conv}$")
ax_.semilogx(f['z']/const.AU, f['rho']*f['S'], 'k', label=r"$\rho S$")
"""
ax_.set_ylabel(r"erg$/$cm$^3/$s")
ax_.legend()

ax_ = ax[3,1]
ax_.semilogx(z/const.AU, f['rho']*f['g'], label=r"$\rho g$")
#ax_.semilogx(z/const.AU, -np.gradient(f['wzz'], z), label=r"$-dw_{zz}/dz$")
ax_.set_ylabel(r"erg$/$cm$^4$")
ax_.legend()
"""

ax_ = ax[3,1]
ax_.semilogx(z/const.AU, np.abs(np.gradient(f['wzz'], z)), label=r"$|dw_{zz}/dz|$")
ax_.semilogx(z/const.AU, np.abs(f['rho']*f['g']), '--k', label=r"$\rho |g|$")
ax_.set_ylabel(r"erg$/$cm$^4$")
ax_.legend()
"""

for ax_ in ax.ravel():
    ax_.axvline(f['H_th'].values[0]/const.AU, ls=':', c='k')
    ax_.axvline(f['H_sigma'].values[0]/const.AU, ls=':', c='k')
    ax_.set_xlim(xmax=np.max(f['z'])/const.AU)
    #ax_.set_xlabel(r"$z$ [AU]")
    ##
    ax__ = ax_.twiny()
    ax__.set_xscale('log')
    ax__.set_xlim(ax_.get_xlim())
    ax__.set_xticks([ f['H_th'].values[0]/const.AU, f['H_sigma'].values[0]/const.AU ])
    ax__.set_xticks([], minor=True)
    ax__.set_xticklabels([r"$H_\mathrm{th}$", r"$H_\sigma$"])

plt.tight_layout()
plt.savefig(Path(fname).stem + ".pdf")
plt.show()
