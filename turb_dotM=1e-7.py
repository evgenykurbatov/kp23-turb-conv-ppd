# -*- coding: utf-8 -*-
"""
"""

import gzip

import numpy as np
np.set_printoptions(formatter={'float_kind': (lambda x: "%.4e" % x)})
from numpy import pi, sqrt, log

import h5py

import numba
from numba import njit, types
from numba.experimental import jitclass

import cologne
from cologne import const
from cologne import util
from cologne import opac_vp17 as opac
from cologne import inicond
from cologne import rttb
from cologne import hydrostat
from cologne import extheat
from cologne import conv
from cologne import turb



@jitclass
class Model:
    ##
    ## Parameters of the model

    M_star:    types.float64
    R_star:    types.float64
    T_star:    types.float64
    dotM:      types.float64
    T_isr:     types.float64

    R:         types.float64
    Omega:     types.float64

    mu_mol:    types.float64
    gamma:     types.float64
    C_p:       types.float64
    C_v:       types.float64

    sigma_max: types.float64
    rho_ext:   types.float64

    ##
    ## Parameters of the run

    fpath: str
    fbase: str

    n_nod: types.int64

    t_fin: types.float64
    t_ss:  types.float64[:]

    ##
    ## Fields

    sigma:   types.float64[:]
    z:       types.float64[:]
    rho:     types.float64[:]
    T:       types.float64[:]
    E:       types.float64[:]
    F_rad:   types.float64[:]
    F_conv:  types.float64[:]
    effconv: types.float64[:]
    S_uv:    types.float64[:]
    S_ext:   types.float64[:]
    S_conv:  types.float64[:]
    w:       types.float64[:,:]
    S_turb:  types.float64[:]



    def __init__(self):
        pass



    def heat_sources(self):
        ## Star
        beta = extheat.grazing_angle_cg97(self.R)
        #beta = extheat.grazing_angle_dzn02(self.sigma, self.z, self.T_star, self.R_star, self.R)
        S_star = extheat.heat_star(self.sigma, self.T_star, self.R_star, self.R, beta)

        ## ISR
        S_isr  = extheat.heat_isr(self.sigma, self.T_isr)

        ## Accretion
        S_accr = extheat.heat_accr(self.sigma_max, self.M_star, self.dotM, self.R) \
                 * np.ones_like(self.rho)

        S_uv = S_star + S_isr
        S_ext = S_accr

        return S_uv, S_ext



    def heating_callback(self, T):

        cs2 = const.RR_gas/self.mu_mol * T

        ## External sources
        self.S_uv, self.S_ext \
            = self.heat_sources()

        ## Convection
        g = - self.Omega**2 * self.z
        ell = np.sqrt(cs2[0]) / abs(self.Omega)
        #ell = z[sigma < 0.9*self.sigma_max][-1]
        self.F_conv, self.effconv, self.S_conv, dSconv_dT \
            = conv.convection(self.z, self.rho, T, g, self.mu_mol, self.gamma,
                              opac.fn_kappaR(T), ell)

        ## Turbulence
        self.S_turb \
            = turb.heat(self.rho, cs2, self.w, self.Omega)

        S = self.S_uv + self.S_ext + self.S_conv + self.S_turb
        dS_dT = dSconv_dT

        return S, dS_dT



    def advance_to(self, turbpar, turbcrhs, dt, rtol=1e-4, max_iter=10000):
        """
        Advance the system to a given time step size.
        """

        sigma = self.sigma

        ## Gradients of keplerian velocity field
        Omega_1 = - 1.5 * self.Omega
        Omega_2 = 0.5 * self.Omega

        z_0   = self.z.copy()
        rho_0 = self.rho.copy()
        T_0   = self.T.copy()
        E_0   = self.E.copy()
        w_0   = self.w.copy()

        dt_pass = 0.0
        while dt_pass < dt:

            dt_ = dt - dt_pass

            z   = z_0.copy()
            rho = rho_0.copy()
            T   = T_0.copy()
            E   = E_0.copy()

            n_iter = 0
            while True:

                ## Opacties
                kappaP = opac.fn_kappaP(T)
                kappaR = opac.fn_kappaR(T)

                ##
                ## Radiative transfer and thermal balance

                T_new, E_new, F_rad_new, dt_ \
                    = rttb.advance2_lin(sigma, rho, T, T_0, E, E_0, self.C_v,
                                        self.heating_callback, kappaP, kappaR, dt_)

                ##
                ## Adjust the hydrostatic configuration

                cT2 = const.RR_gas/self.mu_mol * T
                z_new, rho_new \
                    = hydrostat.adjust(sigma, rho, cT2, self.rho_ext, self.Omega**2, w_0[2])


                ## Check the convergence conditions
                cond_tb = np.all( np.abs(T_new - T) <= rtol*T )
                cond_rt = np.all( np.abs(E_new - E) <= rtol*E )
                if cond_tb & cond_rt:
                    break

                z   = z_new.copy()
                rho = rho_new.copy()
                T   = T_new.copy()
                E   = E_new.copy()

                n_iter += 1
                if n_iter >= max_iter:
                    #print("T =", T)
                    #print("E =", E)
                    #print("rho =", rho)
                    raise ValueError("advance_to: n_iter >= max_iter")

            z_0   = z_new.copy()
            rho_0 = rho_new.copy()
            T_0   = T_new.copy()
            E_0   = E_new.copy()

            dt_pass += dt_

        ##
        ## Turbulence

        cT2 = const.RR_gas/self.mu_mol * T_new
        Bzz = - 2.0*(self.gamma-1.0)/self.gamma * self.F_conv * util.grads(log(rho_new*cT2), z_new)
        w_new, _ \
            = turb.advance_to(turbpar, turbcrhs,
                              z_new, rho_new, cT2, Bzz, w_0,
                              self.Omega, Omega_1, Omega_2, dt)

        self.z       = z_new.copy()
        self.rho     = rho_new.copy()
        self.T       = T_new.copy()
        self.E       = E_new.copy()
        self.w       = w_new.copy()
        self.F_rad   = F_rad_new.copy()

        return



def init(model):

    ## Number of nodes
    model.n_nod = 1001

    ##
    ## Make logarithmic grid of the surface density, `sigma` [g cm-2]

    ## Outer bound of the column
    tau_out = 1e-5
    T_out = 3e2  ## [K]
    sigma_out = tau_out / opac.fn_kappaP(T_out)
    ## NB: If `inverse=True` then the surface density values grow from the upper bound
    model.sigma = inicond.sigma_grid(model.n_nod, model.sigma_max, sigma_out, inverse=False)

    ##
    ## Make initial distributions

    ## Nodes' vertical coordinates [cm], volume density [g cm-3] and temperature [K]
    T_00 = 100.0  ## [K]
    model.z, model.rho, model.T \
        = inicond.isothermal(model.sigma, T_00, model.rho_ext, model.mu_mol, model.Omega,
                             verbose=True)

    ## Initial radiative energy distribution [erg cm-3]
    T_cmb = 2.73
    model.E = const.a_rad*T_cmb**4 * np.ones_like(model.sigma)
    #model.E = const.a_rad*T**4
    ## Radiative flux for _uniform_ radiative energy distribution [erg cm-2 s-1]
    model.F_rad = np.zeros_like(model.sigma)

    ## External heat sources
    model.S_uv, model.S_ext \
        = model.heat_sources()

    ## Convection
    model.F_conv  = np.zeros_like(model.sigma)
    model.effconv = np.zeros_like(model.sigma)
    model.S_conv  = np.zeros_like(model.sigma)

    ## Turbulence
    model.w       = np.zeros((6, model.n_nod))
    model.S_turb  = np.zeros_like(model.sigma)



def run(model, t_ss):
    print("run:")

    ## Init the model for turbulence, there is no other place, right
    mturb = turb.init(model.n_nod, use_diffusion=True)

    ## Index of the current time point
    n = 0
    ## Index of the current snapshot time point
    n_ss = 0

    nt_ss = []

    ## Initial time point
    #t = np.array([ 0.0 ])
    t = [ 0.0 ]
    ## Initial time step
    dt = 1.0  ## [s]

    while True:

        ## Snapshot time?
        if t[-1] >= t_ss[n_ss]:
            write_snapshot(model, t[-1], n_ss)
            nt_ss = np.append(nt_ss, [n])
            n_ss += 1

        ## Time to stop?
        if t[-1] >= model.t_fin:
            break

        ## Going to cross the snapshot time?
        if t[-1] + dt > t_ss[n_ss]:
            dt = t_ss[n_ss] - t[-1]

        try:
            ## Advance to the given time step
            model.advance_to(mturb[0], mturb[1].address, dt)
        except ValueError as e:
            write_snapshot(model, t[-1], 999)
            raise ValueError("run_pure: ValueError in `advance_to`")

        ## Switch to the next time point
        t_new = t[-1] + dt
        #t = np.vstack([t, t_new])
        t.append(t_new)

        if not (n % 100):
            print("%d:  t = %.2e [yr]    dt = %.2e [yr]" \
                  % (n, t[-1]/const.year, dt/const.year))

        #dt *= 1.001
        dt *= 1.01
        n += 1

    print("run: done")
    print("%d:  t = %.2e [yr]" % (n, t[-1]/const.year))



def write_snapshot(model, t, n_ss):
    ffullname = '%s/%s%03d.gz' % (model.fpath, model.fbase, n_ss)
    print("Write snapshot '%s' at t = %.2e [yr]" % (ffullname, t/const.yr))

    X = np.stack((model.sigma, model.z, model.rho, model.T, model.E,
                  model.F_rad, model.S_uv, model.S_ext,
                  model.F_conv, model.effconv, model.S_conv,
                  model.w[0], model.w[1], model.w[2], model.w[3], model.w[4], model.w[5], model.S_turb), axis=1)

    f = gzip.open(ffullname, 'wt')
    f.write("%e\n" % t)
    np.savetxt(f, X, fmt='%.6e', comments='# ',
               header="sigma [g cm-2], z [cm], rho [g cm-3], T [K], E [erg cm-3]"
               ", F_rad [erg cm-2 s-1], S_uv [erg g-1 s-1], S_ext [erg g-1 s-1]"
               ", F_conv [erg cm-2 s-1], effconv, S_conv [erg g-1 s-1]"
               ", wrr [erg cm-3], wff, wzz, wrf, wrz, wfz, S_turb [erg g-1 s-1]")
    f.close()

    ##
    Q_uv   = ( 0.5*(model.S_uv[:-1]   + model.S_uv[1:]  ) * np.diff(model.sigma) ).sum()
    Q_ext  = ( 0.5*(model.S_ext[:-1]  + model.S_ext[1:] ) * np.diff(model.sigma) ).sum()
    Q_conv = ( 0.5*(model.S_conv[:-1] + model.S_conv[1:]) * np.diff(model.sigma) ).sum()
    Q_turb = ( 0.5*(model.S_turb[:-1] + model.S_turb[1:]) * np.diff(model.sigma) ).sum()
    print("  Q_uv   = %.2e [erg cm-2 s-1]" % Q_uv)
    print("  Q_ext  = %.2e [erg cm-2 s-1]" % Q_ext)
    print("  Q_conv = %.2e [erg cm-2 s-1]" % Q_conv)
    print("  Q_turb = %.2e [erg cm-2 s-1]" % Q_turb)



##
## The source is executed as a main program
##

if __name__ == '__main__':

    model = Model()

    ##
    ## Read parameters into the model instance

    print("Read parameters:")

    ## Path to data folder
    model.fpath = 'turb_dotM=1e-7'

    import importlib
    param = importlib.import_module(model.fpath + '.param')

    keys = []
    for key in dir(param):
        val = getattr(param, key)
        if type(val) is int or type(val) is float:
            setattr(model, key, val)
            keys.append(key)
    print(" ", keys)

    ##
    ## Parameters of the run

    model.fbase = ''

    ## Final time [s]
    model.t_fin = 1000.0 * const.yr

    ## Snapshot time points [s]
    #t_ss = np.arange(0.0, model.t_fin, 0.25*const.yr)
    t_ss = np.linspace(0.0, 1.0, 50)**3 * model.t_fin
    print("t_ss [yr] =", t_ss/const.yr)
    model.t_ss = np.unique( np.concatenate((t_ss, [model.t_fin])) )


    ##
    ## Init and run

    print("dotM = %.2e [M_sol/yr]" % (model.dotM/(const.M_sol/const.yr)))

    init(model)

    #import sys
    #sys.exit()

    run(model, model.t_ss)
