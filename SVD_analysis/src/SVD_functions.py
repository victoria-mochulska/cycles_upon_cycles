
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, fields
import pandas as pd
import os
from skimage.restoration import unwrap_phase
from SVD_analysis.src.kymo_functions import (posterior_kymo, anterior_kymo, posterior_kymo_half_mask, stretched_kymo,
                                             spline_kymo)
from SVD_analysis.src.fitting_utils import fit_function, linear, exp_plus_const, exp_sigmoid, linear_plus_6h, par_multiply, par_divide
from SVD_analysis.src.data_settings import load_kymo

size=12
params = {'legend.fontsize': 'large',
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size,
          'ytick.labelsize': size,
          'axes.titlepad': 5,
          'lines.linewidth': 2,
          'figure.dpi': 200}
plt.rcParams.update(params)


def get_input_kymo(data_dir, exp, date, sample, df, naming=1, anterior=False, preregistered=False, crop=None,
                   stretch=False, extrapolate=False, output=False, dx=1.38):
    ss = df.loc[(exp, date, sample), 'ss']
    dt = df.loc[(exp, date, sample), 'dt']
    # dx = 1.38  ##

    if not preregistered:
        mask_kymo = ~ load_kymo('mask', data_dir, exp, date, sample, naming).astype('bool')
        coords = np.argwhere(mask_kymo)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        mask_kymo = mask_kymo[y_min:y_max + 1, x_min:x_max + 1]
        masked_mask = np.ma.MaskedArray(np.ones(mask_kymo.shape, dtype='int'), mask=~mask_kymo)

        intensity_kymo = load_kymo('intensity', data_dir, exp, date, sample, naming)[y_min:y_max + 1, x_min:x_max + 1]

        detrended_kymo = load_kymo('detrended', data_dir, exp, date, sample, naming)[y_min:y_max + 1, x_min:x_max + 1]
        detrended_kymo = np.ma.MaskedArray(detrended_kymo, mask=~mask_kymo)

        period_kymo = load_kymo('period', data_dir, exp, date, sample, naming)[y_min:y_max + 1, x_min:x_max + 1]
        period_kymo = np.ma.MaskedArray(period_kymo, mask=~mask_kymo)

        phase_kymo = load_kymo('phase', data_dir, exp, date, sample, naming)[y_min:y_max + 1, x_min:x_max + 1]
        phase_kymo = np.ma.MaskedArray(phase_kymo, mask=~mask_kymo)

        phase_kymo = unwrap_phase(phase_kymo)  ## unwrapped_masked_phase_kymo - phi(x, t)

        # create a registered kymo
        if anterior:
            rect_phi_full = anterior_kymo(phase_kymo)
        else:
            rect_phi_full = posterior_kymo(phase_kymo)
        # _________________________________________

        if stretch:
            rect_phi = stretched_kymo(rect_phi_full)
        elif extrapolate:
            rect_phi = spline_kymo(rect_phi_full, dx)
            excluded_area = np.ones(rect_phi_full.shape) * np.nan
        else: # crop to shortest PSM length
            if crop is None:
                # crop_ind = max_area_ind(phi)
                # crop = phi.shape[1] - crop_ind
                crop = df.loc[(exp, date, sample), 'crop']
                # print('Crop timepoints:', crop)
            excluded_area = np.ones(rect_phi_full.shape) * np.nan
            if crop > 0:
                excluded_area[:, -crop:] = 1.
            # psm_edges = np.ma.notmasked_edges(phi, axis=0)
            # psm_length = (psm_edges[1][0] - psm_edges[0][0]).astype(int)

            width = mask_kymo.shape[1]
            psm_length = mask_kymo.sum(axis=0)[:width-crop]
            min_psm_length = np.min(psm_length)
            max_psm_length = np.max(psm_length)


            if anterior:
                rect_phi = rect_phi_full[:min_psm_length, :width-crop].copy()
                excluded_area[min_psm_length:, :] = 1.
            else:
                rect_phi = rect_phi_full[-min_psm_length+1:, :width-crop].copy()
                excluded_area[:max_psm_length - min_psm_length, :] = 1.


    if preregistered:
        mask_kymo = ~ load_kymo('mask', data_dir, exp, date, sample, naming).astype('bool')
        coords = np.argwhere(mask_kymo)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        mask_kymo = mask_kymo[y_min:y_max + 1, 0:x_max + 1]
        mask_kymo[:, :x_min] = mask_kymo[:, x_min][:, None]
        x_min = 0


        psm_length = mask_kymo.sum(axis=0)
        max_psm_length = np.max(psm_length)

        masked_mask = np.ma.MaskedArray(np.ones(mask_kymo.shape, dtype='int'), mask=~mask_kymo)
        post_mask = posterior_kymo_half_mask(masked_mask, mask_kymo).astype('bool')[-max_psm_length:, :]
        width = post_mask.shape[1]
        height=post_mask.shape[0]
        intensity_kymo = load_kymo('intensity', data_dir, exp, date, sample, naming)[:, x_min:x_max + 1]

        detrended_kymo = load_kymo('detrended', data_dir, exp, date, sample, naming)[-height:, x_min:x_max + 1]
        detrended_kymo = np.ma.MaskedArray(detrended_kymo, mask=~post_mask)

        phase_kymo = load_kymo('phase', data_dir, exp, date, sample, naming=naming)[-height:, x_min:x_max + 1]
        phase_kymo = np.ma.MaskedArray(phase_kymo, mask=~post_mask)

        # already posteriorly registered
        rect_phi_full = unwrap_phase(phase_kymo)

        if extrapolate:
            rect_phi = spline_kymo(rect_phi_full, dx)
            excluded_area = np.ones(rect_phi_full.shape) * np.nan
        else:
            if crop is None:
                crop = df.loc[(exp, date, sample), 'crop']
                # print('Crop timepoints:', crop)
            excluded_area = np.ones(rect_phi_full.shape) * np.nan
            if crop > 0:
                excluded_area[:, -crop:] = 1.

            min_psm_length = np.min(psm_length[:width-crop])
            rect_phi = rect_phi_full[-min_psm_length + 1:, :width-crop].copy()
            excluded_area[:max_psm_length - min_psm_length, :] = 1.
            rect_phi = rect_phi.data

    t0 = x_min

    if output:
        plot_input_kymos(intensity_kymo, detrended_kymo, phase_kymo, rect_phi_full, excluded_area, rect_phi, dx, dt)

    # row = 30
    # plt.figure(figsize=(4, 2))
    # plt.plot(detrended_kymo[row, :], lw=2)
    # plt.plot(intensity_kymo[row, :]-np.mean(intensity_kymo[row, :]), lw=2)
    # # plt.plot(100*np.cos(phase_kymo[row, :]), c='k', lw=2)
    # plt.show()
    masked_mask = masked_mask[:, :width-crop]         # crop the mask
    return rect_phi, masked_mask, rect_phi_full, t0


def velocities_from_kymo(masked_phi, dx=1., dt=1., plot=False):
    psm_edges = np.ma.notmasked_edges(masked_phi, axis=0)
    tt = np.arange(masked_phi.shape[1]) * dt

    fit_f, pcov_f = np.polyfit(tt, psm_edges[0][0], 1, cov=True)
    fit_tb, pcov_tb = np.polyfit(tt, psm_edges[1][0], 1, cov=True)
    perr_f = np.sqrt(np.diag(pcov_f))
    perr_tb = np.sqrt(np.diag(pcov_tb))

    v_f = fit_f[0] * dx
    v_f_err = perr_f[0] * dx
    v_tb = fit_tb[0] * dx
    v_tb_err = perr_tb[0] * dx

    x0_tb = fit_tb[1] * dx
    x0_f = fit_f[1] * dx

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow(~masked_phi.mask, cmap='Blues', vmax=5)
        ax.plot(psm_edges[1][1], psm_edges[1][0] - 1, '--', c='steelblue')
        ax.plot(psm_edges[0][1], psm_edges[0][0], '--', c='steelblue')
        ax.set_title('PSM edges')
        ax.plot(fit_f[0] * tt + fit_f[1], c='indianred')
        ax.plot(fit_tb[0] * tt + fit_tb[1], c='indianred')
        plt.show()
    return ValErr(v_tb, v_tb_err), ValErr(v_f, v_f_err)

# _______________________________________________________


@dataclass(frozen=True, slots=True)
class ValErr:
    val: float | np.ndarray
    err: float | np.ndarray

@dataclass(frozen=True, slots=True)
class Products:
    U0U0: float | np.ndarray
    U0U1: float | np.ndarray
    U1U1: float | np.ndarray
    V0V0: float | np.ndarray
    V0V1: float | np.ndarray
    V1V1: float | np.ndarray

@dataclass(slots=True)
class FitModesResult:
    s0: float
    s1: float

    fit_u0: np.ndarray
    fit_v0: np.ndarray
    fit_u1: np.ndarray
    fit_v1: np.ndarray

    pars_v0: np.ndarray
    perr_v0: np.ndarray
    pars_u1: np.ndarray
    perr_u1: np.ndarray
    # fit_u1_exp: np.ndarray
    # fit_u1_quad: np.ndarray

    omega_instant: np.ndarray

    # fit parameters
    A0: ValErr
    A1: ValErr
    omega0: ValErr
    beta: ValErr
    Abeta: ValErr
    C0: float
    C1: ValErr
    x0: ValErr

    # optional: velocities
    v_tb: ValErr | None = None
    v_f: ValErr | None = None

    resid_svd: float | None = None
    resid_fit: float | None = None

    # computed post-init
    prod: Products = field(init=False)
    good_fit: bool = field(init=False)
    T0: ValErr = field(init=False)
    Amp: ValErr = field(init=False)
    alpha: ValErr | None = field(init=False, default=None)
    alpha_c: ValErr | None = field(init=False, default=None)
    lambda_tb: ValErr | None = field(init=False, default=None)
    lambda_f: ValErr | None = field(init=False, default=None)
    resid_cycle: float | None = field(init=False, default=None)

    def __post_init__(self):
        T0, T0_err = par_divide((2 * np.pi, 0.), (self.omega0.val, self.omega0.err))
        self.T0 = ValErr(T0, T0_err)

        self.Amp = ValErr(*par_multiply((self.s1*self.A1.val, self.s1*self.A1.err), (np.abs(self.Abeta.val), self.Abeta.err)))

        self._update_pars()

        U0U0 = np.dot(self.fit_u0, self.fit_u0)
        U0U1 = np.dot(self.fit_u0, self.fit_u1)
        U1U1 = np.dot(self.fit_u1, self.fit_u1)
        V0V0 = np.dot(self.fit_v0, self.fit_v0)
        V0V1 = np.dot(self.fit_v0, self.fit_v1)
        V1V1 = np.dot(self.fit_v1, self.fit_v1)

        self.prod = Products(U0U0, U0U1, U1U1, V0V0, V0V1, V1V1)

        # criteria for quality of fit
        poor_fit = (
            self.Abeta.err > abs(self.Abeta.val)
            or self.C1.err > abs(self.C1.val)
            or U1U1 < 0.9
            or V1V1 < 0.9
        )

        if poor_fit:
            print("⚠ Poor mode fit")
            # self.Abeta = ValErr(np.nan, np.nan)
            # self.beta = ValErr(np.nan, np.nan)
            # self.C1 = ValErr(np.nan, np.nan)
            self.good_fit = False
        else:
            self.good_fit = True

    def _update_pars(self):
        T0, T0_err = self.T0.val, self.T0.err
        if self.v_tb is not None:
            self.alpha = ValErr(*par_multiply((self.beta.val, self.beta.err), (self.v_tb.val, self.v_tb.err)))
            # if self.alpha.val > 0.008:
            #     print('Alpha outlier')
            #     self.good_fit = False
            self.alpha_c = ValErr(*par_multiply((self.beta.val, self.beta.err),
                                                (self.v_tb.val, self.v_tb.err), (T0, T0_err)))
            self.lambda_tb = ValErr(*par_multiply((self.v_tb.val, self.v_tb.err), (T0, T0_err)))
        if self.v_f is not None:
            self.lambda_f = ValErr(*par_multiply((self.v_f.val, self.v_f.err), (T0, T0_err)))

    #.    Velocities are being shifted to lab frame here !
    def set_vel(self, v_tb, v_f):
        self.v_tb = ValErr(1.5*v_tb.val, 1.5*v_tb.err)
        self.v_f = ValErr(v_f.val + 0.5*v_tb.val, v_f.err + 0.5*v_tb.err)
        self._update_pars()

    def set_resid(self, resid_svd, resid_fit):
        self.resid_svd = resid_svd
        self.resid_fit = resid_fit
        self.resid_cycle = self.resid_fit / (2*np.pi)*100
        if resid_fit / (2 * np.pi) > 0.15:
            print("⚠ Fit residual > 15%")
            self.good_fit = False

    def to_dict(self):
        row = {}
        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            # parameters to skip
            if name in {"A0", "A1", "Abeta"}:
                continue

            if isinstance(val, ValErr):
                row[name] = val.val
                row[f"{name} err"] = val.err

            elif isinstance(val, (type(None), float, int, np.floating, np.integer)):
                row[name] = val
            else:
                continue
        return row

    def write_row(self, df, idx):
        row = self.to_dict()
        for c in row:
            if c not in df.columns:
                df[c] = pd.NA
        df.loc[idx, list(row)] = pd.Series(row)

# ______________________________________________


def fit_modes(u, vh, s, xx, xx_full, tt, dt, ss, date, output=True, weighted_fit=False, cycling=False,
              fit_functions=None):

    if fit_functions is None:
        if cycling:
            fit_functions = (linear_plus_6h, exp_plus_const)
            # fit_functions = (linear_plus_6h, exp_sigmoid)  ###
            # print('Sigmoid fit')
        else:
            fit_functions = (linear, exp_plus_const)      # for temporal and spatial mode
            # fit_functions = (linear, exp_sigmoid)  ###
            # print('Sigmoid fit')

    xx_psm = np.linspace(0., 1., u.shape[0])

    mode = 0  # 000000000000000000000000000000000000000000000000000000000000000000000000
    if np.mean(np.gradient(vh[mode, :])) < 0:
        u[:, mode] *= -1.
        vh[mode, :] *= -1.

    A0 = np.mean(u[:, mode])
    A0_err = np.std(u[:, mode])
    fit_u0 = A0 * np.ones(u.shape[0])

    fit_v0, pars_v0, perr_v0 = fit_function(fit_functions[0], tt, vh[mode, :])
    C0 = s[0] * A0 * pars_v0[-1]
    omega0, omega0_err = par_multiply((pars_v0[0], perr_v0[0]), (A0 * s[0], A0_err * s[0]))
    T0, T0_err = par_divide((2 * np.pi, 0.), (omega0, omega0_err))

    if cycling:
        T_instant_fit = 2*np.pi/A0/s[0] / ( pars_v0[0] + 2*np.pi*pars_v0[1]/360. * np.cos(2*np.pi/360.*tt + pars_v0[2]) )
        period_amplitude = np.max(T_instant_fit) - np.min(T_instant_fit)

    omega_instant = (np.gradient(vh[mode, :], dt) * A0 * s[0])
    T_instant = 2 * np.pi / omega_instant
    tt_cycle = tt / T0 + ss

    #     np.savetxt(data_dir+exp+'/output_omega_instant_'+date+'_P'+str(sample)+'.txt', omega_instant)   ######
    if output:
        fig, ax = plt.subplots(1, 2, figsize=[6, 2])
        plot_mode(ax, mode, u, vh, s, tt, xx)
        ax[0].plot(xx, fit_u0, '--', c='k', alpha=0.8, lw=2)
        ax[0].fill_between(xx, fit_u0 - A0_err, fit_u0 + A0_err, alpha=0.25, color='grey')

        # ax[0].set_ylim([0.9*A0, 1.1*A0])
        ax[1].plot(tt, fit_v0, '--', c='k', alpha=0.8, lw=2, zorder=3)
        ax1 = ax[1].twinx()
        ax1.plot(tt, T_instant, lw=2, zorder=0, alpha=0.3)
        ax1.plot(tt, T0 * np.ones(len(tt)), ':', c='k', alpha=0.7, lw=2, zorder=1)
        if cycling:
            ax1.plot(tt, T_instant_fit, '--', c='k', alpha=0.7, lw=2, zorder=2)
        ax1.set_ylim([0.7 * T0, 1.3 * T0])
        ax1.set_ylabel(r'$T_0$ (mins)', labelpad=8)
        print('T0:', T0, '+-', T0_err)
        if cycling:
            print('Period amplitude:', period_amplitude, 'mins')

        # if date == '20210604':
        #     ax1_1 = ax[1].twinx()
        #     ax1_1.plot(temp_20210604.index / T0 + tt_cycle[0], temp_20210604['avg temp'], c='k', alpha=0.3)
        #     ax1_1.set_yticks([])
        # if date == '20210605':
        #     ax1_1 = ax[1].twinx()
        #     ax1_1.plot(temp_20210605.index / T0 + tt_cycle[0], temp_20210605['avg temp'], c='k', alpha=0.3)
        #     ax1_1.set_yticks([])
        fig.tight_layout()
        plt.show()

    # _________________________________________________________________________________________
    mode = 1
    if np.mean(np.gradient(u[::-1, mode])) > 0:
        # exchange the sign of modes u and v to have negative spatial gradient
        u[:, mode] *= -1.
        vh[mode, :] *= -1.

    #    temporal mode
    A1 = np.mean(vh[mode, :])
    A1_err = np.std(vh[mode, :])
    fit_v1 = A1 * np.ones(len(vh[mode, :]))

    #    spatial mode
    if weighted_fit == True:
        fit_sigma = np.ones(len(xx)) - 0.75 * (xx_psm > 0.75)  ### put more weight in anterior
        # fit_sigma = 1./(1+xx_psm**20)  ### put more weight in anterior
        # fit_sigma = np.ones(len(xx)) - 0.75*(xx_psm > 0.25)*(xx_psm < 0.8) ### put more weight in the middle
    else:
        fit_sigma = np.ones(len(xx))
    ##################### constrained amplitude (fixed A)

    #   Exponential
    A_fixed = -0.4 / s[1] / A1
    fit_u1, pars_u1_exp, perr_u1_exp = fit_function(fit_functions[1], xx, u[::-1, mode], p0=(A_fixed, 0.01, 0.1),
                                                        bounds=((A_fixed-0.0001, 0., -100.), (A_fixed+0.0001, 100., 100.)), sigma=fit_sigma)

    Abeta = pars_u1_exp[0]
    Abeta_err = perr_u1_exp[0]
    beta = pars_u1_exp[1]
    beta_err = perr_u1_exp[1]
    C1 = pars_u1_exp[2]
    C1_err = perr_u1_exp[2]
    x0 = None
    x0_err = None

    # Sigmoid
    # A_fixed = - 2.*np.pi / s[1] /A1
    # fit_u1, pars_u1_exp, perr_u1_exp = fit_function(fit_functions[1], xx, u[::-1, mode], p0=(A_fixed, 0.02, -0.1, 0),
    #                                                     bounds=((A_fixed-1e-3, 0.005, -2., 0), (A_fixed+1e-3, 0.3, 2., 500)), sigma=fit_sigma)
    # Abeta = pars_u1_exp[0]
    # Abeta_err = perr_u1_exp[0]
    # beta = pars_u1_exp[1]
    # beta_err = perr_u1_exp[1]
    # C1 = pars_u1_exp[2]
    # C1_err = perr_u1_exp[2]
    # x0 = pars_u1_exp[3]
    # x0_err = perr_u1_exp[3]

    ####
    pars_u1_exp_1 = pars_u1_exp.copy()
    pars_u1_exp_2 = pars_u1_exp.copy()
    pars_u1_exp_1[1] -= beta_err
    pars_u1_exp_2[1] += beta_err

    if output:
        fig, ax = plt.subplots(1, 2, figsize=[5, 2])
        plot_mode(ax, mode, u, vh, s, tt, xx)
        ax1 = ax[1].twinx()
        v_gradient = np.gradient(vh[mode, :], dt)
        ax1.plot(tt, v_gradient, lw=2, zorder=0, alpha=0.15)
        ax1.set_yticks([])
        ax[0].plot(xx, fit_u1, '--', c='k', alpha=0.6, lw=2)
        # ______________ Extrapolations _________________________________________________________
        # ax[0].plot(xx_full, exp_plus_const(xx_full, *pars_u1_exp), '--', c='k', alpha=0.6, lw=2)
        # #                 ax[0].plot(fit_u1_quad, '--', c = 'indianred', alpha=0.6, lw=2)
        # ax[0].plot(xx_full, quadratic(xx_full, *pars_u1_quad), '--', c='indianred', alpha=0.6, lw=2)
        # ax[0].fill_between(xx_full, exp_plus_const(xx_full, *pars_u1_exp_1), exp_plus_const(xx_full, *pars_u1_exp_2),
        #                    alpha=0.25, color='grey')
        ax[0].fill_between(xx, fit_functions[1](xx, *pars_u1_exp_1), fit_functions[1](xx, *pars_u1_exp_2),
                           alpha=0.25, color='grey')
        # ax[0].set_ylim((-0.5, 0.25))
        # ax[0].set_ylim((-0.2, 0.2))  # ____limits without extrapolation
        print('Spatial pars:', np.round(pars_u1_exp, 3), ' with errors', np.round(perr_u1_exp, 3))
        ax[1].plot(tt, fit_v1, '--', c='k', alpha=0.8, lw=2)
        ax[1].fill_between(tt, fit_v1 - A1_err, fit_v1 + A1_err, alpha=0.25, color='grey')
        fig.tight_layout()
        plt.show()

    res = FitModesResult(
        s0=s[0], s1=s[1],
        fit_u0=fit_u0, fit_u1=fit_u1,
        fit_v0=fit_v0, fit_v1=fit_v1,
        A0=ValErr(A0, A0_err),
        A1=ValErr(A1, A1_err),
        omega0=ValErr(omega0, omega0_err),
        beta=ValErr(beta, beta_err),
        Abeta=ValErr(Abeta, Abeta_err),
        C1=ValErr(C1, C1_err),
        C0=C0, x0 = ValErr(x0, x0_err),
        omega_instant=omega_instant,
        pars_u1=pars_u1_exp,
        perr_u1=perr_u1_exp
    )

    return res


# ___________________________________________________________________________________________________________________

def fit_modes_sigmoid(u, vh, s, xx, xx_full, tt, dt, ss, date, output=True, weighted_fit=False, cycling=False,
              fit_functions=None):

    # print('Sigmoid fit')

    if fit_functions is None:
        if cycling:
            fit_functions = (linear_plus_6h, exp_sigmoid)
        else:
            fit_functions = (linear, exp_sigmoid)      # for temporal and spatial mode


    xx_psm = np.linspace(0., 1., u.shape[0])

    mode = 0  # _____________________________________________________________________________
    if np.mean(np.gradient(vh[mode, :])) < 0:
        u[:, mode] *= -1.
        vh[mode, :] *= -1.

    A0 = np.mean(u[:, mode])
    A0_err = np.std(u[:, mode])
    fit_u0 = A0 * np.ones(u.shape[0])

    fit_v0, pars_v0, perr_v0 = fit_function(fit_functions[0], tt, vh[mode, :])
    C0 = s[0] * A0 * pars_v0[-1]
    omega0, omega0_err = par_multiply((pars_v0[0], perr_v0[0]), (A0 * s[0], A0_err * s[0]))
    T0, T0_err = par_divide((2 * np.pi, 0.), (omega0, omega0_err))

    if cycling:
        T_instant_fit = 2*np.pi/A0/s[0] / ( pars_v0[0] + 2*np.pi*pars_v0[1]/360. * np.cos(2*np.pi/360.*tt + pars_v0[2]) )
        period_amplitude = np.max(T_instant_fit) - np.min(T_instant_fit)
        # print('Period amplitude:', period_amplitude)

    omega_instant = (np.gradient(vh[mode, :], dt) * A0 * s[0])
    T_instant = 2 * np.pi / omega_instant
    tt_cycle = tt / T0 + ss

    #     np.savetxt(data_dir+exp+'/output_omega_instant_'+date+'_P'+str(sample)+'.txt', omega_instant)   ######
    if output:
        fig, ax = plt.subplots(1, 2, figsize=[5.55, 2])
        plot_mode(ax, mode, u, vh, s, tt, xx)
        ax[0].plot(xx, fit_u0, '--', c='k', alpha=0.8, lw=2)
        ax[0].fill_between(xx, fit_u0 - A0_err, fit_u0 + A0_err, alpha=0.25, color='grey')

        # ax[0].set_ylim([0.9*A0, 1.1*A0])
        ax[1].plot(tt, fit_v0, '--', c='k', alpha=0.8, lw=2, zorder=3)
        ax1 = ax[1].twinx()
        ax1.plot(tt, T_instant, lw=2, zorder=0, alpha=0.3)
        ax1.plot(tt, T0 * np.ones(len(tt)), ':', c='k', alpha=0.7, lw=2, zorder=1)
        if cycling:
            ax1.plot(tt, T_instant_fit, '--', c='k', alpha=0.7, lw=2, zorder=2)
        ax1.set_ylim([0.7 * T0, 1.3 * T0])
        ax1.set_ylabel(r'$T_0$ (mins)', labelpad=8)
        print('T0:', T0, '+-', T0_err)
        if cycling:
            print('Period amplitude:', period_amplitude, 'mins')
        fig.tight_layout()
        plt.show()

    # _________________________________________________________________________________________
    mode = 1
    if np.mean(np.gradient(u[::-1, mode])) < 0:
        # exchange the sign of modes u and v to have positive spatial gradient
        u[:, mode] *= -1.
        vh[mode, :] *= -1.

    #    temporal mode
    A1 = np.mean(vh[mode, :])
    A1_err = np.std(vh[mode, :])
    fit_v1 = A1 * np.ones(len(vh[mode, :]))

    #    spatial mode
    if weighted_fit == True:
        fit_sigma = np.ones(len(xx)) - 0.75 * (xx_psm > 0.75)  ### put more weight in anterior
        # fit_sigma = 1./(1+xx_psm**20)  ### put more weight in anterior
        # fit_sigma = np.ones(len(xx)) - 0.75*(xx_psm > 0.25)*(xx_psm < 0.8) ### put more weight in the middle
    else:
        fit_sigma = np.ones(len(xx))

    # Sigmoid
    A_fixed = - 2*np.pi / s[1] /A1      # constrained amplitude
    fit_u1, pars_u1, perr_u1 = fit_function(fit_functions[1], xx, u[::-1, mode], p0=(A_fixed, 0.02, -0.1, 110),
                                                        bounds=(((A_fixed-1e-2), 0.005, -2., 0), ((A_fixed+1e-2), 0.1, 2., 300)), sigma=fit_sigma)
    Abeta = pars_u1[0]
    Abeta_err = perr_u1[0]
    beta = pars_u1[1]
    beta_err = perr_u1[1]
    C1 = pars_u1[2].copy()
    C1_err = perr_u1[2].copy()
    x0 = pars_u1[3]
    x0_err = perr_u1[3]

    C1 *= s[1] * A1
    C1_err *= s[1] * A1

    ####
    print(pars_u1)

    pars_u1_exp_1 = pars_u1.copy()
    pars_u1_exp_2 = pars_u1.copy()
    pars_u1_exp_1[1] -= beta_err
    pars_u1_exp_2[1] += beta_err

    pars_u1_1 = pars_u1.copy()
    pars_u1_2 = pars_u1.copy()
    pars_u1_1[3] -= x0_err
    pars_u1_2[3] += x0_err

    if output:
        fig, ax = plt.subplots(1, 2, figsize=[5, 2])
        plot_mode(ax, mode, u, vh, s, tt, xx)
        ax1 = ax[1].twinx()
        v_gradient = np.gradient(vh[mode, :], dt)
        # ax1.plot(tt, v_gradient, lw=2, zorder=0, alpha=0.15)
        ax1.set_yticks([])
        ax[0].plot(xx, fit_u1, '--', c='k', alpha=0.6, lw=2)
        ax[0].fill_between(xx, fit_functions[1](xx, *pars_u1_exp_1), fit_functions[1](xx, *pars_u1_exp_2),
                           alpha=0.25, color='grey')
        ax[0].fill_between(xx, fit_functions[1](xx, *pars_u1_1), fit_functions[1](xx, *pars_u1_2),
                           alpha=0.25, color='grey')
        print('Spatial pars:', np.round(pars_u1, 3), ' with errors', np.round(perr_u1, 3))
        ax[1].plot(tt, fit_v1, '--', c='k', alpha=0.8, lw=2)
        ax[1].fill_between(tt, fit_v1 - A1_err, fit_v1 + A1_err, alpha=0.25, color='grey')
        fig.tight_layout()
        plt.show()

    res = FitModesResult(
        s0=s[0], s1=s[1],
        fit_u0=fit_u0, fit_u1=fit_u1,
        fit_v0=fit_v0, fit_v1=fit_v1,
        A0=ValErr(A0, A0_err),
        A1=ValErr(A1, A1_err),
        omega0=ValErr(omega0, omega0_err),
        beta=ValErr(beta, beta_err),
        Abeta=ValErr(Abeta, Abeta_err),
        C1=ValErr(C1, C1_err),
        C0=C0, x0 = ValErr(x0, x0_err),
        omega_instant=omega_instant,
        pars_u1=pars_u1,
        perr_u1=perr_u1,
        pars_v0=pars_v0,
        perr_v0=perr_v0,
    )

    return res

# ___________________________________________________________________________________________________________

def deltaphi_analysis(rect_phi_full, res, exp, date, sample, df, cycling=False):
    # subtracting mode 0 fit

    fit_function = linear_plus_6h if cycling else linear

    K = rect_phi_full

    if np.ma.isMaskedArray(K):
        mask = ~K.mask
        K = K.filled(np.nan).astype(float)
    else:
        K = K.astype(float, copy=False)
        mask = ~np.isnan(K)

    h, w = K.shape
    dt = df.loc[(exp, date, sample), 'dt']
    tt = np.arange(w)*dt
    it = np.arange(w)


    pars_v0 = res.pars_v0
    phi0 = fit_function(tt, *pars_v0) *res.A0.val *res.s0

    omega0 = res.omega0.val
    tt_cycle = tt *omega0 /2/np.pi
    # phi0 = omega0*tt

    deltaphi = K - phi0
    deltaphi -= np.nanmean(deltaphi)
    deltaphi-= np.nanmedian(deltaphi[-5:, :])

    x = np.arange(h)[:, None]

    top = np.where(mask, x, np.inf).min(axis=0)
    bottom = np.where(mask, x, -np.inf).max(axis=0)

    x_psm = (bottom - x) / (bottom - top)
    x_psm = np.where(mask, x_psm, np.nan)

    # plt.figure(figsize=[2, 2])
    # plt.imshow(deltaphi, cmap='magma')
    # plt.colorbar()
    # # plt.show()
    # # ax1 = plt.gca()

    fig, ax = plt.subplots(1, 2, figsize=[5, 2])

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, 5))

    for k, psm_loc in enumerate((0.05, 0.25, 0.5, 0.75, 0.95)):
        dist = np.where(mask, np.abs(x_psm - psm_loc), np.inf)
        i = dist.argmin(axis=0)
        i0 = i[0]*np.ones(w).astype(int)
        # ax1.plot(i, c = colors[k])
        # ax1.plot(i0, c = colors[k])
        vals = np.abs(deltaphi[i, it])/np.pi
        ax[0].plot(tt, vals, label=int(psm_loc*100), c=colors[k])
        if psm_loc == 0.95:
            linear_fit = np.polyfit(tt_cycle, vals, 1)
            ax[0].plot(tt, linear_fit[0]*tt_cycle +linear_fit[1], '--', c='k')
            kappa_c = linear_fit[0]

            linear_fit = np.polyfit(tt, vals, 1)
            kappa = linear_fit[0]

            # if deltaphi_change > 0:
            #     deltaphi_change = np.nan
            df.loc[(exp, date, sample), 'kappa'] = -kappa
            df.loc[(exp, date, sample), 'kappa_c'] = -kappa_c
            df.loc[(exp, date, sample), 'deltaphi_init'] = linear_fit[1]

        vals = np.abs(deltaphi[i0, it])/np.pi
        ax[1].plot(tt, vals, label=int(psm_loc*100), c=colors[k])
    ax[0].set_title('PSM percentages')
    ax[0].set_ylim((0, 2.5))
    ax[1].set_title('Fixed location')
    ax[1].set_ylim((0, 2.5))

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()


# ____________________________________________________________________________________________________________

def kymo_svd(K, masked_phi, exp, date, sample, df, output=True, plot_matrices=False, plot_modes=2, n_modes=2, t0=0,
             write=False, weighted_fit=False, cycling=False,
             save_dir = None, sigmoid=True):
    ss = df.loc[(exp, date, sample), 'ss']
    dt = df.loc[(exp, date, sample), 'dt']
    dx = df.loc[(exp, date, sample), 'dx']

    phi_shift = np.mean(K)
    K -= phi_shift

    xx = np.arange(K.shape[0]) * dx
    xx_full = xx
    xx_psm = np.linspace(0., 1., len(xx))

    tt = (np.arange(K.shape[1]) + t0) * dt

    # tt_full = np.arange(phi_og.shape[1]) * dt
    # xx_full = np.arange(max_psm_length) * dx

    u, s, vh = np.linalg.svd(K, full_matrices=False)

    if output:
        print('s0:', s[0], '| s1:', s[1], '| sum s:', np.sum(s))
        if plot_matrices == True:
            plot_svd_matrices(u, vh, s)

        # _________ Fitting the modes ________________________________________
    if sigmoid:
        res = fit_modes_sigmoid(u, vh, s, xx, xx_full, tt, dt, ss, date, output, weighted_fit=weighted_fit, cycling=cycling)
    else:
        res = fit_modes(u, vh, s, xx, xx_full, tt, dt, ss, date, output, weighted_fit=weighted_fit, cycling=cycling)

    # fitting velocities to mask kymo
    v_tb, v_f = velocities_from_kymo(masked_phi, dx, dt)
    res.set_vel(v_tb, v_f)

    psm = ~masked_phi.mask
    psm0 = np.sum(psm[:, 0])*dx
    print('psm0:', psm0)
    df.loc[(exp, date, sample), 'psm0'] = psm0
    df.loc[(exp, date, sample), 'psm_f'] = K.shape[0]*dx

    T0 = res.T0.val
    tt_abs = tt + ss * T0
    tt_cycle = tt_abs / T0
    tt_cycle_13 = tt_cycle - 13.

    # spatial_mode_0 = u[::-1, 0]
    # temporal_mode_0 = vh[0, :]
    # spatial_mode_1 = u[::-1, 1]
    # temporal_mode_1 = vh[1, :]


    mode_1_time_avg = s[1] * res.A1.val * u[::-1, 1]
    mode_0_space_avg = s[0] * res.A0.val * vh[0, :]

    # saving to files _______________________________________________________________________________________
    if save_dir is not None:
        os.makedirs(save_dir + '/'+ exp, exist_ok=True)
        save_to = save_dir + '/' + exp
        np.savetxt(save_to + '/time_' + date + '_P' + str(sample) + '.txt', tt)
        np.savetxt(save_to + '/omega_instant_' + date + '_P' + str(sample) + '.txt', res.omega_instant)
        np.savetxt(save_to + '/mode1_time_avg_' + date + '_P' + str(sample) + '.txt', mode_1_time_avg)
        np.savetxt(save_to + '/mode0_space_avg_' + date + '_P' + str(sample) + '.txt', mode_0_space_avg)

        # np.savetxt(save_to + '/spatial_mode_0_' + date + '_P' + str(sample) + '.txt', spatial_mode_0)
        # np.savetxt(save_to + '/spatial_mode_1_' + date + '_P' + str(sample) + '.txt', spatial_mode_1)
        # np.savetxt(save_to + '/temporal_mode_0_' + date + '_P' + str(sample) + '.txt', temporal_mode_0)
        # np.savetxt(save_to + '/temporal_mode_1_' + date + '_P' + str(sample) + '.txt', temporal_mode_1)
        np.savetxt(save_to + '/singular_values_' + date + '_P' + str(sample) + '.txt', s)


    if output:
        for mode in range(2, plot_modes):
            fig, ax = plt.subplots(1, 2, figsize=[3.5 * 2 + 0.7, 2.5])
            # print('Mode', mode)
            plot_mode(ax, mode, u, vh, s, tt)
            fig.tight_layout()
            plt.show()

    # reconsruct kymo
    reconstr_phi, residual_reconstr, fit_phi, residual_fit = kymo_reconstruct_and_fit(K, u, vh, s, n_modes, res, xx, tt, output=output)
    res.set_resid(residual_reconstr, residual_fit)

    if write:
        res.write_row(df, (exp, date, sample))
    return res

# ___________________________________________________________________________________________________________________
# ___________________________________________________________________________________________________________________

def plot_mode(ax, mode, u, vh, s, tt, xx=None):
    # print('Mode', mode, ', sigma = ', s[mode])
    if xx is None:
        ax[0].plot(u[::-1, mode], alpha=0.6, lw=3)
        ax[0].set_xlabel('Dist. from posterior (pixels)')
    else:
        ax[0].plot(xx, u[::-1, mode], alpha=0.6, lw=3)
        ax[0].set_xlabel('Dist. from posterior (um)', fontsize=12)
    ax[0].set_title('Spatial component')
    ax[0].set_ylabel('MODE ' + str(mode), labelpad=12)
    if np.max(np.abs(u[::-1, mode])) > 0.25:
        ax[0].set_ylim((-0.5, 0.5))
    else:
        ax[0].set_ylim((-0.25, 0.25))

    ax[1].plot(tt, vh[mode, :], alpha=0.6, zorder=2, lw=3)
    ax[1].set_title('Temporal component')
    max_h = np.max(tt)/60 + 1
    # adjust x ticks here
    tick = 2
    ax[1].set_xticks((np.arange(0, max_h, tick))*60, labels=np.arange(0, max_h, tick).astype('int'))
    ax[1].set_xlabel('Time (hours)', fontsize=12)
    ax[1].set_ylim((-0.25, 0.25))


# _____________________________________________________________________________________________

def plot_svd_matrices(u, vh, s):
    fig, ax = plt.subplots(2, 1, figsize=[3, 8], gridspec_kw={'height_ratios': [u.shape[0], vh.shape[1]]})
    ax[0].imshow(u, cmap='RdBu', extent=[0, u.shape[1], 0, u.shape[0]])
    ax[0].set_aspect('equal')
    ax[0].set_title('U (spatial)')
    ax[0].set_xlabel('Mode number')
    ax[0].set_ylabel('Space (pixels)')

    ax[1].imshow(vh.T, cmap='RdBu', extent=[0, vh.shape[1], vh.shape[0], 0])
    ax[1].set_aspect('equal')
    ax[1].set_title('V (temporal)')
    ax[1].set_xlabel('Mode number')
    ax[1].set_ylabel('Time (pixels)')
    ax[1].set_yticks(np.arange(0, vh.shape[1], 20))
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    plt.plot(s[:10], '*', markersize=11)
    ax.set_yscale('log')
    ax.set_title('Singular values')
    ax.set_xlabel('Mode number')
    ax.set_xticks(np.arange(10))
    plt.show()

# ____________________________________________________________________________________________________________________
from matplotlib.ticker import MultipleLocator

def plot_input_kymos(intensity_kymo, detrended_kymo, phase_kymo, rect_phi_full, excluded_area, rect_phi, dx, dt, vmin=None, vmax=None):
    psm_edges = np.ma.notmasked_edges(phase_kymo, axis=0)

    fig, ax = plt.subplots(1, 5, figsize=[6 * 4 * 0.8, 5 ], gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]})
    extent = [0, phase_kymo.shape[1]*dt, (psm_edges[1][0][0]-phase_kymo.shape[0])*dx, psm_edges[1][0][0]*dx]
    ax[0].imshow(intensity_kymo, cmap='magma', vmin=vmin, vmax=vmax,  extent=extent)
    ax[0].yaxis.set_major_locator(MultipleLocator(50))
    ax[0].xaxis.set_major_locator(MultipleLocator(150))

    # ax[0].set_title('Intensity')

    extent = [0, phase_kymo.shape[1]*dt, (psm_edges[1][0][0]-phase_kymo.shape[0])*dx, psm_edges[1][0][0]*dx]
    ax[1].imshow(detrended_kymo, cmap='magma', interpolation='none',  extent=extent)
    # ax[1].set_title('Detrended')

    # ax[3].imshow(1./period_kymo, cmap='magma',  extent=[0, phi_kymo.shape[1], psm_edges[1][0][0]-phi_kymo.shape[0], psm_edges[1][0][0]])
    # ax[3].set_title('Frequency')

    ax[2].imshow(np.cos(phase_kymo), cmap='magma', extent=extent)
    # ax[2].set_title(r'cos $\phi$')

    extent = [0, rect_phi_full.shape[1]*dt, 0, rect_phi_full.shape[0]*dx]
    ax[3].imshow(np.cos(rect_phi_full), cmap='magma', extent=extent)
    ax[3].imshow(excluded_area, cmap='Greys', alpha=0.6, vmin=1., vmax=1.5,
                 extent=extent)
    # ax[3].set_title(r'cos $\phi$')

    extent = [0, rect_phi.shape[1]*dt, 0, rect_phi.shape[0]*dx]
    ax[4].imshow(np.cos(rect_phi), cmap='magma', extent=extent)
    # ax[4].set_title('cos $\phi$')

    for i in range(5):
        if i ==0:
            ax[i].set_xlabel('Time (mins)')
            ax[i].set_ylabel('Space (um)')
        else:
            ax[i].set_xticks(())
            ax[i].set_yticks(())
            # ax[i].axis('off')
        ax[i].set_aspect('auto')
    # fig.tight_layout()
    plt.show()

# _____________________________________________________________________________________________________________________

def plot_mode_2d(ax, mode, u, vh, s, u_fit=None, vh_fit=None):
    # fig, ax = plt.subplots(1,6,figsize = [6*4*0.6,3])
    mode_2d = s[mode] * np.outer(u[:, mode], vh[mode, :])
    if u_fit is not None and vh_fit is not None:

        mode_fit_2d = s[mode] * np.outer(u_fit[:, mode], vh_fit[mode, :])

        vmin = np.min(mode_fit_2d)
        vmax = np.max(mode_fit_2d)

        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Space')
        ax[0].contourf(mode_2d, cmap='Blues', origin='upper', vmin=vmin, vmax=vmax)
        ax[0].set_title('Mode ' + str(mode))
        ax[1].contourf(mode_fit_2d, cmap='Blues', origin='upper', vmin=vmin, vmax=vmax)
        ax[1].set_title('Mode ' + str(mode) + ' fit')
        ax[2].imshow(np.cos(mode_2d), cmap='magma', origin='upper', vmin=-1., vmax=1.) ## mode_fit_2d
        # ax[2].set_title('cos (Mode ' + str(mode) + ' fit)')
        ax[2].set_title('cos (Mode ' + str(mode) + ')')

    else:
        # ax.imshow(mode_2d, cmap='Blues', origin='upper')
        # print(np.min(np.cos(mode_2d)), np.max(np.cos(mode_2d)))
        ax.imshow(np.cos(mode_2d), cmap='magma', origin='upper', vmin=-0.05, vmax=0.05)
        # ax.imshow(mode_2d, cmap='Blues', origin='upper')
        ax.set_title('cos (Mode ' + str(mode) + ')')
        # ax.set_title('Mode ' + str(mode))

    # vmin = np.min(s[0]*np.outer(u_fit[:,0], vh_fit[0,:]))
    # vmax = np.max(s[0]*np.outer(u_fit[:,0], vh_fit[0,:]))
    # ax[0].set_xlabel('Time')
    # ax[0].set_ylabel('Space')
    # ax[0].contourf(s[0]*np.outer(u[:,0], vh[0,:]), cmap ='Blues', origin='upper', vmin=vmin, vmax=vmax)
    # ax[0].set_title('0th mode')
    # ax[1].contourf(s[0]*np.outer(u_fit[:,0], vh_fit[0,:]), cmap ='Blues', origin='upper', vmin=vmin, vmax=vmax)
    # ax[1].set_title('0th mode fit')
#         ax[5].imshow(np.cos(s[0]*np.outer(u_fit[:,0], vh_fit[0,:])), cmap='magma')
#         ax[5].set_title('cos(phi0)')

#         vmin = np.min(s[1]*np.outer(u_fit[:,1], vh_fit[1,:]))
#         vmax = np.max(s[1]*np.outer(u_fit[:,1], vh_fit[1,:]))
#         ax[2].contourf(s[1]*np.outer(u[:,1], vh[1,:]), cmap ='Blues', origin='upper', vmin=vmin, vmax=vmax)
#         ax[2].set_title('1st mode')
#         ax[3].contourf(s[1]*np.outer(u_fit[:,1], vh_fit[1,:]), cmap ='Blues', origin='upper', vmin=vmin, vmax=vmax)
#         ax[3].set_title('1st mode fit')
# #         ax[6].imshow(np.cos(s[1]*np.outer(u_fit[:,1], vh_fit[1,:])), cmap = 'magma', vmin=-1, vmax=1.)
# #         ax[6].set_aspect('equal')
# #         ax[6].set_title('cos(phi1)')
#
#         ax[4].contourf(s[2]*np.outer(u[:,2], vh[2,:]), cmap ='Blues', origin='upper')
#         ax[4].set_title('2nd mode')
#         ax[5].contourf(s[3]*np.outer(u[:,3], vh[3,:]), cmap ='Blues', origin='upper')
#         ax[5].set_title('3rd mode')


def kymo_reconstruct_and_fit(K, u, vh, s, n_modes, res, xx, tt, output=True, fig=None, ax=None):
    u_fit = np.zeros(u.shape)
    vh_fit = np.zeros(vh.shape)
    u_fit[:, 0] = res.fit_u0
    u_fit[:, 1] = res.fit_u1[::-1]
    vh_fit[0, :] = res.fit_v0
    vh_fit[1, :] = res.fit_v1

    smat = np.zeros((u.shape[1], vh.shape[0]), dtype=float)
    smat[:n_modes, :n_modes] = np.diag(s[:n_modes])
    reconstr_phi = u @ smat @ vh
    residual_reconstr = np.median(np.abs(reconstr_phi - K))

    fit_phi = u_fit @ smat @ vh_fit
    residual_fit = np.median(np.abs(fit_phi - K))

    if output:
        extent = [0, np.max(tt), 0, np.max(xx)]
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=[4 * 3 * 0.6, 3], gridspec_kw={'width_ratios': [1, 1, 1]})
        ax[0].imshow(np.cos(K), cmap='magma', extent=extent)
        # ax[0].set_title('Original')
        # ax[0].set_xlabel('Time')
        # ax[0].set_ylabel('Space')
        ax[1].imshow(np.cos(reconstr_phi), cmap='magma', extent=extent)
        # ax[1].set_title('Reconstr, res =' + str(np.round(residual_reconstr, 2)))
        ax[2].imshow(np.cos(fit_phi), cmap='magma', extent=extent)
        # ax[2].set_title('Fit, res =' + str(np.round(residual_fit, 2)))

        # resid_plot = (K - fit_phi)/np.pi
        # max_abs = np.max(np.abs(resid_plot))
        # max_abs =np.pi/5
        # im = ax[2].imshow(resid_plot, cmap='RdBu', vmin=-max_abs, vmax=max_abs)
        # # ax[2].set_title(r'Residual ($\pi$)')
        for i in range(3):
            ax[i].set_xticks(np.arange(0, np.max(tt), 6*60), labels=np.arange(0, np.max(tt)/60, 6).astype('int'))
            ax[i].set_aspect(5)
            ax[i].set_xlabel('Time (hours)')
            ax[i].set_ylabel('Space (um)')
            # ax[i].set_yticks([])

        # cbar = fig.colorbar(im, fraction=0.05, pad=0.1)
        # cbar.set_ticks((-np.pi/10, 0, np.pi/10))  # positions
        # cbar.set_ticklabels(('$-\pi/10$', '$0$', '$\pi/10$'))  # optional custom labels
        plt.tight_layout()
        plt.show()

        # fig, ax = plt.subplots(1, 3, figsize=[4 * 3 * 0.6, 3])
        # s0 = s.copy()
        # s0[1:]=0
        # smat = np.zeros((u.shape[1], vh.shape[0]), dtype=float)
        # smat[:n_modes, :n_modes] = np.diag(s0[:n_modes])
        # reconstr_0 = u @ smat @ vh
        # fit_0 = u_fit @ smat @ vh_fit
        # ax[1].contourf(fit_0, cmap='Blues', origin='lower')
        # ax[2].imshow(np.cos(fit_0), cmap='magma')
        # for i in range(3):
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])
        # plt.show()
        #
        #
        # fig, ax = plt.subplots(1, 3, figsize=[4 * 3 * 0.6, 3])
        # s1 = s.copy()
        # s1[2:]=0
        # s1[0] =0
        # smat = np.zeros((u.shape[1], vh.shape[0]), dtype=float)
        # smat[:n_modes, :n_modes] = np.diag(s1[:n_modes])
        # reconstr_0 = u @ smat @ vh
        # fit_0 = u_fit @ smat @ vh_fit
        # ax[0].contourf(reconstr_0[::-1], cmap='Blues', origin='lower')
        # ax[1].contourf(fit_0[::-1], cmap='Blues', origin='lower')
        # ax[2].imshow(np.cos(fit_0), cmap='magma')
        # for i in range(3):
        #     ax[i].set_xticks([])
        #     ax[i].set_yticks([])
        # plt.show()


    return reconstr_phi, residual_reconstr, fit_phi, residual_fit

