import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.restoration import unwrap_phase

from SVD_analysis.src.data_settings import load_kymo
from SVD_analysis.src.fitting_utils import fit_function, linear, exp_plus_const, quadratic

from scipy.interpolate import CubicSpline


def posterior_kymo(masked_kymo, dtype='float'):
    psm_edges = np.ma.notmasked_edges(masked_kymo, axis=0)
    psm_length = (psm_edges[1][0]-psm_edges[0][0]).astype(int)
    max_psm_length = np.max(psm_length)
    post_kymo = np.ones((max_psm_length, masked_kymo.shape[1]), dtype=dtype)*np.nan
    for col in range(masked_kymo.shape[1]):
        post_kymo[max_psm_length-psm_length[col]:, col] = masked_kymo[psm_edges[0][0][col]:psm_edges[1][0][col], col]
    return post_kymo


def undo_posterior_kymo(post_kymo, masked_kymo, x0_tb=None, v=None, dx=None, dt=None):
    diag_kymo = np.nan * np.ones(masked_kymo.shape)
    psm_length = post_kymo.shape[0]
    if x0_tb is None and v is None:
        psm_edges = np.ma.notmasked_edges(masked_kymo, axis=0)
        x_tb = psm_edges[1][0]

    else:
        x0 = x0_tb / dx
        tt = np.arange(diag_kymo.shape[1]) * dt
        x_tb = np.round(x0 + (v / dx * tt)).astype('int')
    x00 = x_tb[0]
    for col in range(masked_kymo.shape[1]):
        ant_gap = x_tb[col] - psm_length
        diag_kymo[ant_gap:x_tb[col], col] = post_kymo[:, col]
    return diag_kymo, x00


def anterior_kymo(masked_kymo, dtype='float'):
    psm_edges = np.ma.notmasked_edges(masked_kymo, axis=0)
    psm_length = (psm_edges[1][0]-psm_edges[0][0]).astype(int)
    max_psm_length = np.max(psm_length)
    ant_kymo = np.ones((max_psm_length, masked_kymo.shape[1]), dtype=dtype)*np.nan
    for col in range(masked_kymo.shape[1]):
        # print(-max_psm_length+psm_length[col])
        ant_kymo[:psm_length[col], col] = masked_kymo[psm_edges[0][0][col]:psm_edges[1][0][col], col]
    return ant_kymo


def posterior_kymo_half_mask(masked_kymo, full_kymo, dtype='int'):
    psm_edges = np.ma.notmasked_edges(masked_kymo, axis=0)
    # psm_length = (psm_edges[1][0]-psm_edges[0][0]).astype(int)
    psm_length = np.min(psm_edges[1][0])
    # max_psm_length = np.min((max_psms_length, np))
    # print(max_psm_length)
    post_kymo = np.zeros((psm_length, masked_kymo.shape[1]), dtype=dtype)
    for col in range(masked_kymo.shape[1]):
        start_ind = np.max((0, psm_edges[1][0][col] - psm_length))
        # print(psm_edges[1][0][col] - max_psm_length, psm_edges[1][0][col])
        post_kymo[:, col] = full_kymo[start_ind:psm_edges[1][0][col], col]
    return post_kymo


def stretched_kymo(post_kymo):
    psm_length = post_kymo.shape[0] - np.argmin(np.isnan(post_kymo), axis=0)
    max_psm_length = np.max(psm_length)
    stretch_kymo = np.zeros((max_psm_length, post_kymo.shape[1]))
    for col in range(post_kymo.shape[1]):
        stretch_kymo[:, col] = zoom(post_kymo[-psm_length[col]:, col], max_psm_length/psm_length[col])
    return stretch_kymo


# ______________________________________________________________________________________

def spline_kymo(phase_kymo, dx):
    extended_phase_kymo = np.zeros(phase_kymo.shape)

    # For every column of the phase kymo, get column, calculate spline and add new values to the column
    for col in range(phase_kymo.shape[1]):

        spat_mode0 = phase_kymo[:, col]
        true_spat_mode0 = [i for i in spat_mode0 if i == i]  # removes all NaN values
        range_to_cut = round(0.1 * len(true_spat_mode0))  # cut beginning and end of column, as well as xx
        crop_spat_mode0 = true_spat_mode0[range_to_cut:-range_to_cut]
        xx = np.arange(len(true_spat_mode0)) * dx
        crop_xx = xx[range_to_cut:-range_to_cut]

        spline = CubicSpline(crop_xx, crop_spat_mode0,
                             bc_type='natural')  # 'natural' boundary cond. produces the best results

        for i in range(phase_kymo.shape[0]):  # arbitrary number, always larger than the actual amount of loops
            if i < range_to_cut:  # extend spline to the left and right to replaced cropped pixels
                extend_spline(spline, dx, True)
            else:  # extend spline only to the left, replacing NaN values which were removed
                extend_spline(spline, dx, False)

            if len(spline.x) == phase_kymo.shape[0]:
                break

        extended_phase_kymo[0:, col] = spline(spline.x)  # add 'extended' column to new kymo

        # if (col+5) % 10 == 0:
        #     plt.figure(figsize=(3,2))
        #     # plt.plot(np.arange(len(true_spat_mode0)) * dx, true_spat_mode0[::-1], c='tab:blue')
        #     plt.plot(true_spat_mode0[::-1], c='tab:blue')
        #     plt.plot(spline(spline.x)[::-1], c='k', linestyle='--')
        #     plt.show()
    return extended_phase_kymo





# def smooth_inside_mask(phi_ma, iters=40, step=0.25):
#     """
#     Anisotropic Laplacian smoothing constrained to the masked region.
#     Keeps known pixels fixed; smooths only the extrapolated wedge.
#     """
#     out  = phi_ma.filled(0.0).copy()
#     mask = phi_ma.mask
#
#     for _ in range(iters):
#         nbr = (np.roll(out,1,0) + np.roll(out,-1,0) +
#                np.roll(out,1,1) + np.roll(out,-1,1)) * 0.25
#         out[mask] = (1.0 - step) * out[mask] + step * nbr[mask]
#     return np.ma.array(out, mask=mask)


# Utility function for extending spline, adds one knot to the left and/or right
def extend_spline(spline, dx, right):
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)
    # add knots at the same interval as spline.x (which is dx)
    leftxnext = leftx - dx

    leftynext = lefty + leftslope * (leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    if right:
        rightx = spline.x[-1]
        righty = spline(rightx)
        rightslope = spline(rightx, nu=1)
        rightxnext = rightx + dx

        rightynext = righty + rightslope * (rightxnext - rightx)
        rightcoeffs = np.array([0, 0, rightslope, rightynext])
        spline.extend(rightcoeffs[..., None], np.r_[rightxnext])







# _______________________________________________________________________________________

def deltaphi(post_kymo):
    linear_fit_phi0 = np.polyfit(np.arange(post_kymo.shape[1]), post_kymo[-1,:], 1)
    phi0_kymo = np.tile((linear_fit_phi0[0]*np.arange(post_kymo.shape[1]) + linear_fit_phi0[1]), (post_kymo.shape[0],1))
    deltaphi_kymo = post_kymo - phi0_kymo
    return deltaphi_kymo


def max_area_ind(masked_kymo):
    psm_edges = np.ma.notmasked_edges(masked_kymo, axis=0)
    psm_length = (psm_edges[1][0]-psm_edges[0][0]).astype(int)
    max_psm_length = np.max(psm_length)
    weighted_area_crop = 2.*(max_psm_length - psm_length)*np.arange(masked_kymo.shape[1]) + [np.sum(psm_length[col:]) for col in range(masked_kymo.shape[1])]
    ind = np.argmin(weighted_area_crop)
    return ind


def deltaphi_analysis(data_dir, exp, date, sample, results_df, naming=1):
    ss = results_df.loc[(exp, date, sample), 'ss']
    dt = results_df.loc[(exp, date, sample), 'dt']
    T0 = results_df.loc[(exp, date, sample), 'T0']

    mask_kymo = ~ load_kymo('mask', data_dir, exp, date, sample, naming).astype('bool')
    coords = np.argwhere(mask_kymo)
    x_min, y_min = coords.min(axis=0);
    x_max, y_max = coords.max(axis=0)
    mask_kymo = mask_kymo[x_min:x_max + 1, y_min:y_max + 1]
    phase_kymo = load_kymo('phase', data_dir, exp, date, sample, naming)[x_min:x_max + 1, y_min:y_max + 1]
    masked_phase_kymo = np.ma.MaskedArray(phase_kymo, mask=~mask_kymo)
    phi = unwrap_phase(masked_phase_kymo)

    rect_phi_full = posterior_kymo(phi)
    rect_phi_stretch = stretched_kymo(rect_phi_full)

    tt = np.arange(phi.shape[1]) * dt
    tt_cycle_13 = tt / T0 + ss - 13.

    psm50 = int(0.5 * rect_phi_stretch.shape[0])
    psm25 = int(0.75 * rect_phi_stretch.shape[0])
    psm75 = int(0.25 * rect_phi_stretch.shape[0])
    psm95 = int(0.05 * rect_phi_stretch.shape[0])

    linear_phi0, linear_fit_phi0, perr_phi0 = fit_function(linear, tt, rect_phi_stretch[-1, :])
    omega0_tb = linear_fit_phi0[0]
    omega0_tb_err = perr_phi0[0]

    linear_phiA, linear_fit_phiA, perr_phiA = fit_function(linear, tt, rect_phi_stretch[psm95, :])
    omegaA_front = linear_fit_phiA[0]
    omegaA_front_err = perr_phiA[0]

    linear_deltaphi, linear_fit_deltaphi, perr_deltaphi = fit_function(linear, tt_cycle_13[10:],
                                                                       rect_phi_stretch[-1, 10:] - rect_phi_stretch[
                                                                                                   psm95, 10:],
                                                                       xx_fit=tt_cycle_13)

    linear_psm50, linear_fit_psm50, perr_psm50 = fit_function(linear, tt_cycle_13,
                                                              linear_phi0 - rect_phi_stretch[psm50, :])

    fig, ax = plt.subplots(1, 1)
    ax.plot(tt, (-rect_phi_stretch[-1, :] + linear_phi0) / np.pi, label='post', c='tab:blue', lw=2)
    # ax[4].plot(tt, linear_phi0, '--', c='k')
    ax.plot(tt, (-rect_phi_stretch[psm25, :] + linear_phi0) / np.pi, label='25%', c='lightsteelblue', lw=2)
    ax.plot(tt, (-rect_phi_stretch[psm50, :] + linear_phi0) / np.pi, label='50%', c='lightgreen', lw=2)
    ax.plot(tt, linear_psm50 / np.pi, '--', c='k', lw=2)
    ax.plot(tt, (-rect_phi_stretch[psm75, :] + linear_phi0) / np.pi, label='75%', c='gold', lw=2)
    ax.plot(tt, (-rect_phi_stretch[psm95, :] + linear_phi0) / np.pi, label='ant', c='indianred', lw=2)
    ax.plot(tt, linear_deltaphi / np.pi, '--', c='k')
    ax.set_title('deltaphi(t)')
    ax.set_ylim((-0.25, 2))
    plt.show()

    results_df.loc[(exp, date, sample), 'omega_tb'] = omega0_tb
    results_df.loc[(exp, date, sample), 'omega_tb err'] = omega0_tb_err
    results_df.loc[(exp, date, sample), 'omega_front'] = omegaA_front
    results_df.loc[(exp, date, sample), 'omega_front err'] = omegaA_front_err
    results_df.loc[(exp, date, sample), 'deltaphi_change_cycle'] = linear_fit_deltaphi[0]
    results_df.loc[(exp, date, sample), 'deltaphi_change_cycle err'] = perr_deltaphi[0]
    results_df.loc[(exp, date, sample), 'deltaphi_13'] = linear_fit_deltaphi[1]
    results_df.loc[(exp, date, sample), 'deltaphi_13 err'] = perr_deltaphi[1]
    results_df.loc[(exp, date, sample), 'deltaphi_psm50_change'] = linear_fit_psm50[0]
    results_df.loc[(exp, date, sample), 'deltaphi_psm50_change err'] = perr_psm50[0]


def velocities_from_kymo(masked_phi, dx=1., dt=1., plot=False):
    psm_edges = np.ma.notmasked_edges(masked_phi, axis=0)
    tt = np.arange(masked_phi.shape[1]) * dt

    #     fit_f, pcov_f = curve_fit(linear, tt, psm_edges[0][0], p0=(0., 0.))
    fit_f, pcov_f = np.polyfit(tt, psm_edges[0][0], 1, cov=True)
    fit_tb, pcov_tb = np.polyfit(tt, psm_edges[1][0], 1, cov=True)
    #     fit_tb, pcov_tb = curve_fit(linear, tt, psm_edges[1][0], p0=(0., 0.))
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
    return v_tb, v_tb_err, v_f, v_f_err, x0_tb, x0_f

    # ______________________________________________________________________________________________________________________


def synt_kymo(phi, dx, dt, x0_tb, v_tb, v_f, C, omega0, At, pars_u1_exp, pars_u1_quad):  ### C = C0 + phi_shift # At = s[1]*Atemp_avg
    ## synthetic kymo from fitted parameters
    tt_full = np.arange(phi.shape[1])*dt
    fit_diag = np.ones(phi.shape)*np.nan
    fit_diag_quad = np.ones(phi.shape)*np.nan
    synt_dphi = np.zeros(phi.shape[1])
    synt_dphi_quad = np.zeros(phi.shape[1])
    x0 = x0_tb/dx
    x_tb = np.round(x0 + (v_tb/dx*tt_full)).astype('int')
    x_f = np.round((v_f/dx*tt_full)).astype('int')
    for col in range(phi.shape[1]):
#             print('True edges:', (psm_edges[1][0][col], psm_edges[0][0][col]), ', Synt edges:', (x_tb[col], x_f[col]))
        psm_length = x_tb[col] - x_f[col]
        xx_synt = np.arange(psm_length)*dx
        phi_synt = C + omega0*tt_full[col] + At*exp_plus_const(xx_synt, *pars_u1_exp)

        phi_synt_quad = C + omega0*tt_full[col] + At*quadratic(xx_synt, *pars_u1_quad)
        synt_dphi[col] = phi_synt[0] - phi_synt[-1]
        synt_dphi_quad[col] = phi_synt_quad[0] - phi_synt_quad[-1]
        if x_tb[col] > fit_diag.shape[0]-1:
            phi_synt = phi_synt[x_tb[col]-fit_diag.shape[0]+1:]
            phi_synt_quad = phi_synt_quad[x_tb[col]-fit_diag.shape[0]+1:]
            x_tb[col] = fit_diag.shape[0]-1
        fit_diag[x_tb[col]:x_f[col]:-1, col] = phi_synt
        fit_diag_quad[x_tb[col]:x_f[col]:-1, col] = phi_synt_quad
        residual_fit_diag = np.nanmean(np.abs(phi - fit_diag))
        residual_fit_diag_quad = np.nanmean(np.abs(phi - fit_diag_quad))
    synt_mode1 = fit_diag - C - omega0*tt_full[:phi.shape[1]]
    return fit_diag, synt_dphi, residual_fit_diag, fit_diag_quad, synt_dphi_quad, residual_fit_diag_quad, synt_mode1