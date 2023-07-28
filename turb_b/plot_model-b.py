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
from cologne import opac_vp17 as opac
from param import *

from numpy import pi, sqrt, exp, sin, cos, tan, log, log10



##
## Read
##

snapshot = "018"

## Dry run
fname = "../dry_b/%s.gz" % snapshot
print("Reading '%s'..." % fname)
columns = [ 'sigma', 'z', 'rho', 'T', 'E', 'F_rad', 'S_uv', 'S_ext',
            'F_conv', 'effconv', 'S_conv' ]
with gzip.open(fname, 'rt') as ff:
    _ = float(ff.readline())
    f0 = pd.read_csv(ff, header=None, skiprows=1, names=columns,
                    skip_blank_lines=True, sep=" ", skipinitialspace=True)
print("  f0.shape =", f0.shape)
print(f0.head())

## Full run
fname = "./%s.gz" % snapshot
print("Reading '%s'..." % fname)
columns = [ 'sigma', 'z', 'rho', 'T', 'E', 'F_rad', 'S_uv', 'S_ext',
            'F_conv', 'effconv', 'S_conv',
            'wrr', 'wff', 'wzz', 'wrf', 'wrz', 'wfz', 'S_turb' ]
with gzip.open(fname, 'rt') as ff:
    t = float(ff.readline())
    f = pd.read_csv(ff, header=None, skiprows=1, names=columns,
                    skip_blank_lines=True, sep=" ", skipinitialspace=True)
print("  f.shape =", f.shape)
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
f['Omega_1'] = - 1.5 * f['Omega']

## Gravitational acceleration (NB: it's signed)
f['g'] = - f['Omega']**2 * f['z']

f['p'] = const.RR_gas/mu_mol * f['rho']*f['T']
f['cs2'] = gamma * const.RR_gas/mu_mol * f['T']

f0['p'] = const.RR_gas/mu_mol * f0['rho']*f0['T']

##
#f['dFconv_dz'] = np.gradient(f['F_conv'], f['z'])

f0['nabla'] = np.gradient(np.log(f0['T']), np.log(f0['p']))
f0['nabla_ad'] = (gamma - 1)/gamma

f['nabla'] = np.gradient(np.log(f['T']), np.log(f['p']))
f['nabla_ad'] = (gamma - 1)/gamma

f['S'] = f['S_uv'] + f['S_ext'] + f['S_conv'] + f['S_turb']

sigma = f['sigma'].values
kappa_R = opac.fn_kappaR(f['T'].values)
f['kappa'] = kappa_R
#tau_R = np.array([ sp.integrate.simpson(kappa_R[i:], sigma[-1] - sigma[i:]) \
#                   for i in range(len(sigma)) ])
#f['tau'] = tau_R
dtau_R = kappa_R * np.gradient(sigma)
tau = np.cumsum(dtau_R[::-1])
f['tau'] = tau[-1] - tau


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
## Rayleigh number

m = mu_mol*const.m_H
v_th = sqrt(3*const.k_B/m * f['T'])
sigma_nn = 3e-16
varkappa_mol = (1/3) * m*C_v * v_th/sigma_nn

kappaR = opac.fn_kappaR(f['T'].values)
varkappa_rad = 4*const.c*const.a_rad*f['T']**3 / (3*f['rho'] * kappaR)

f['Ra'] = f['effconv'] / (varkappa_mol/varkappa_rad)


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

## 'H_top' scale is the height of the top of the convective zone
j_top = f.query('F_conv.gt(0)').index.max()+1
H_top = f.iloc[j_top]['z']


##
## Turbulent Mach number

K = 0.5*(f['wrr'] + f['wff'] + f['wzz']).values
K[K < 0] = 0
M_T = np.sqrt(2*K/(f['rho'].values * cs**2))

Wrf = 2 * sp.integrate.simpson(f['wrf'], f['z'])
nu_SS = Wrf / (f['sigma'].values[-1] * np.abs(f['Omega_1'].values[-1]))
alpha_SS = nu_SS / (f['H_th'].values[0] * cs[0])
print("sigma_tot = %g [g/cm2]" % f['sigma'].values[-1])
print("H_th = %g [AU]" % (f['H_th'].values[0]/const.AU))
print("alpha_SS =", alpha_SS)
dotM_eff = 2*pi/Omega * Wrf
print("dotM_eff = %.2e [M_sol/yr] = %.2e dotM" \
      % (dotM_eff/(const.M_sol/const.yr), dotM_eff/dotM))

P = 2 * sp.integrate.simpson(f['p'], f['z'])
alpha_eff = Wrf / P
print("alpha_eff =", alpha_eff)

Wrr = 2 * sp.integrate.simpson(f['wrr'], f['z'])
Wff = 2 * sp.integrate.simpson(f['wff'], f['z'])
Wzz = 2 * sp.integrate.simpson(f['wzz'], f['z'])
print("Wrf/Wii =", Wrf / (Wrr + Wff + Wzz))

## Heat sources
Q_uv   = 2 * np.trapz(f['S_uv'].values,   f['sigma'].values)
Q_ext  = 2 * np.trapz(f['S_ext'].values,  f['sigma'].values)
Q_conv = 2 * np.trapz(f['S_conv'].values, f['sigma'].values)
Q_turb = 2 * np.trapz(f['S_turb'].values, f['sigma'].values)
print("Q_uv   = %.2e [erg cm-2 s-1]" % Q_uv)
print("Q_ext  = %.2e [erg cm-2 s-1]" % Q_ext)
print("Q_conv = %.2e [erg cm-2 s-1]" % Q_conv)
print("Q_turb = %.2e [erg cm-2 s-1]" % Q_turb)
#sys.exit(0)



##
## Plot
##



## rc settings
## https://matplotlib.org/stable/tutorials/introductory/customizing.html
mpl.rc('font', family='sans')
mpl.rc('font', size='6.0')
#mpl.rc('text', usetex=True)
mpl.rc('lines', linewidth=1.0)
mpl.rc('axes', linewidth=0.5)
mpl.rc('legend', frameon=False)
mpl.rc('legend', handlelength=2.0)

ratio = 4/3
ncols = 3
nrows = 3
figwidth = 17.0   ## cm

figheight = nrows * figwidth/ncols / ratio

mpl.rc('figure', figsize=[figwidth/2.54, figheight/2.54])

fig, ax = plt.subplots(ncols=ncols, nrows=nrows)


z = f['z'].values


ax_ = ax[0,0]
ax_.set_title(r"$\rho$", x=0.1, y=0.7)
p = ax_.semilogy(f0['z']/const.AU, f0['rho'], '--')
ax_.semilogy(z/const.AU, f['rho'], c=p[0].get_color())
ax_.set_xscale('log')
ax_.set_ylim(ymin=1e-16)
ax_.set_ylabel(r"g cm$^{-3}$")

ax_ = ax[0,1]
ax_.set_title(r"$T$", x=0.1, y=0.7)
p = ax_.plot(f0['z']/const.AU, f0['T'], '--')
ax_.plot(z/const.AU, f['T'], c=p[0].get_color())
ax_.set_xscale('log')
ax_.set_ylabel(r"K")

ax_ = ax[0,2]
ax_.set_title(r"$E$", x=0.1, y=0.7)
p = ax_.semilogy(f0['z']/const.AU, f0['E'], '--')
ax_.semilogy(z/const.AU, f['E'], c=p[0].get_color())
ax_.set_xscale('log')
ax_.set_ylabel(r"erg cm$^{-3}$")
"""
ax_ = ax[0,2]
#ax_.set_title(r"$\kappa_\mathrm{R}$ [cm$^2/$g]")
#ax_.semilogy(z/const.AU, f['kappa'])
#ax_.set_title(r"$\tau_\mathrm{R}$", x=0.15, y=0.7)
#ax_.semilogy(z/const.AU, f['tau'])
#ax_.set_title(r"$\kappa_\mathrm{R}$ [cm$^2/$g]")
ax_.set_xscale('log')
"""


ax_ = ax[1,0]
ax_.set_title(r"$\mathsf{Ra}$", x=0.1, y=0.7)
ax_.loglog(z/const.AU, f['Ra'])
#ax_.axhline(1, ls=':', c='k')
ax_.set_ylim(ymin=3.33e7, ymax=3.33e10)
"""
ax_ = ax[1,0]
ax_.set_title(r"$\nabla - \nabla_\mathrm{ad}$", x=0.2, y=0.6)
p = ax_.plot(f0['z']/const.AU, f0['nabla'] - f0['nabla_ad'], '--')
ax_.plot(z/const.AU, f['nabla'] - f['nabla_ad'], c=p[0].get_color())
ax_.axhline(0.0, ls=':', c='k')
ax_.set_xscale('log')
"""

ax_ = ax[1,1]
p = ax_.semilogy(f0['z']/const.AU, f0['F_rad'], '--')
ax_.semilogy(z/const.AU, f['F_rad'], c=p[0].get_color(), label=r"$F_\mathrm{rad}$")
#p = ax_.semilogy(z/const.AU, f0['F_conv'], '--')
#ax_.semilogy(z/const.AU, f['F_conv'], c=p[0].get_color(), label=r"$F_\mathrm{conv}$")
ax_.semilogy(z/const.AU, f['F_conv'], label=r"$F_\mathrm{conv}$")
ax_.set_xscale('log')
ax_.set_ylabel(r"erg cm$^{-2}$ s$^{-1}$")
ax_.legend()

"""
ax_ = ax[1,2]
ax_.plot(z/const.AU, 1e8 * f['rho']*(f['S_uv'] + f['S_ext']), label=r"$\rho (S_\mathrm{uv} + S_\mathrm{ext})$")
ax_.plot(z/const.AU, 1e8 * f['rho']*f['S_conv'], label=r"$\rho S_\mathrm{conv}$")
ax_.plot(f['z']/const.AU, 1e8 * f['rho']*f['S_turb'], label=r"$\rho S_\mathrm{turb}$")
ax_.plot(f['z']/const.AU, 1e8 * f['rho']*f['S'], 'k', label=r"$\rho S$")
ax_.set_xscale('log')
ax_.set_ylim(ymin=-0.015, ymax=0.04)
ax_.set_ylabel(r"$\times 10^{-8}$ erg$/$cm$^3/$s")
ax_.legend()
"""
ax_ = ax[1,2]
ax_.semilogy(z/const.AU, f['rho']*(f['S_uv'] + f['S_ext']), label=r"$\rho (S_\mathrm{ext} + S_\mathrm{UV})$")
p = ax_.semilogy(z/const.AU, f['rho']*f['S_conv'], label=r"$\rho S_\mathrm{conv}$")
ax_.semilogy(z/const.AU, - f['rho']*f['S_conv'], '--', c=p[0].get_color(),
             label=r"$- \rho S_\mathrm{conv}$")
ax_.semilogy(f['z']/const.AU, f['rho']*f['S_turb'], label=r"$\rho S_\mathrm{turb}$")
#ax_.semilogy(f['z']/const.AU, f['rho']*f['S'], 'k', label=r"$\rho S$")
ax_.set_xscale('log')
#ax_.set_ylim(ymin=3.33e-16, ymax=3.33e-7)
ax_.set_ylim(ymin=3.33e-17, ymax=3.33e-7)
ax_.set_ylabel(r"erg cm$^{-3}$ s$^{-1}$")
ax_.legend()


ax_ = ax[2,0]
ax_.plot(z/const.AU, f['wrr'], label=r"$w_{rr}$")
ax_.plot(z/const.AU, f['wff'], label=r"$w_{\phi\phi}$")
ax_.plot(z/const.AU, f['wzz'], label=r"$w_{zz}$")
ax_.plot(z/const.AU, f['wrf'], label=r"$w_{r\phi}$")
ax_.set_xscale('log')
ax_.set_yscale('log')
ax_.set_ylim(3.33e-9, 1e-1)
ax_.set_xlabel(r"$z$ [AU]")
ax_.set_ylabel(r"erg cm$^{-3}$")
ax_.legend()

ax_ = ax[2,1]
ax_.set_title(r"$\mathcal{M}_\mathrm{turb}$", x=0.15, y=0.75)
ax_.plot(z/const.AU, M_T)
ax_.set_xscale('log')
ax_.set_xlabel(r"$z$ [AU]")

ax_ = ax[2,2]
ax_.plot(z/const.AU, 1e13 * f['rho']*f['g'], label=r"$\rho g$")
ax_.plot(z/const.AU, 1e13 * (-np.gradient(f['wzz'], z)), label=r"$-dw_{zz}/dz$")
ax_.set_xscale('log')
ax_.set_xlabel(r"$z$ [AU]")
ax_.set_ylabel(r"$\times 10^{-13}$ erg cm$^{-4}$")
ax_.set_xlim(xmax=np.max(z)/const.AU)
ax_.legend()


for ax_ in ax.ravel():
    ax_.set_xlim(xmax=np.max(z)/const.AU)
    #ax_.set_xscale('log')
    ax_.axvline(f['H_th'].values[0]/const.AU, ls=':', c='k')
    ax_.axvline(H_top/const.AU, ls=':',
                c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    #ax_.axvline(f['H_sigma'].values[0]/const.AU, ls=':', c='k')
    #ax_.set_xlim(xmax=np.max(f['z'])/const.AU)
    #ax_.set_xlabel(r"$z$ [AU]")


plt.axis('tight')

#plt.subplots_adjust(wspace=0.45, hspace=0.25)
plt.tight_layout()

#plt.savefig("model-b.eps")
plt.savefig("model-b.pdf", bbox_inches='tight')
#plt.show()
