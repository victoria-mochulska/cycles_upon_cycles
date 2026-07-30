
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import matplotlib.cm as cm
cmap = cm.get_cmap('magma')

# __________________________________________________________


def funnel_plot(deltaphi, times, n_angles=80, rad_spacing=10):
    radii = deltaphi[::rad_spacing]
    n_radii = len(radii)
    theta = np.linspace(0, 2*np.pi, n_angles, endpoint=True)
    angles = np.repeat(theta[..., np.newaxis], n_radii, axis=1)
    X = radii * np.cos(angles)
    Y = radii * np.sin(angles)
    Z = radii*np.ones(angles.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = times[np.where(deltaphi==Z[i,j])]
    return X, Y, Z


def hill(x ,k ,n, A=1.):
    if n == float('Inf'):
        return A*step_function(np.abs(x), k)
    return A * k ** n / (k ** n + x ** n)


def step_function(x, k):
    return np.array(x < k)


def moving_average(array, kernel_size):
    array_padded = np.pad(array, (kernel_size//2, kernel_size-1-kernel_size//2), mode='edge')
    array_smooth = np.convolve(array_padded, np.ones((kernel_size,))/kernel_size, mode='valid')
    return array_smooth

# ____________________________________________________________


@np.vectorize
def tb(t, pars):
    if pars['mode'] == 'const':
        return pars['x0'] - pars['v0'] * t

    if pars['mode'] == 'piecewise':
        if t < pars['t0']:
            return pars['x0'] - pars['v1'] * t
        else:
            return pars['x0'] - pars['v1'] * pars['t0'] - pars['v2'] * (t - pars['t0'])

    if pars['mode'] == 'cycling':
        Omega = pars['Omega']
        v0 = pars['v0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return pars['x0'] - v0 * (t - variation / Omega * (np.sin(Omega * (t - time_lag)) + np.sin(Omega * time_lag)))

    if pars['mode'] == 'double cycling':
        Omega1 = pars['Omega1']
        Omega2 = pars['Omega2']
        variation1 = pars['variation1']
        variation2 = pars['variation2']
        v0 = pars['v0']
        time_lag1 = pars['time_lag1']
        time_lag2 = pars['time_lag2']
        x_tb = pars['x0'] - v0 * (t - variation1/Omega1 * (np.sin(Omega1*(t - time_lag1)) + np.sin(Omega1*time_lag1)) -
                                  variation2/Omega2 * (np.sin(Omega2*(t - time_lag2)) + np.sin(Omega2*time_lag2)))
        return x_tb


    else:
        return pars['x0']

def v_tb(t, pars):
    if pars['mode'] == 'const':
        return pars['v0']

    if pars['mode'] == 'piecewise':
        if t < pars['t0']:
            return pars['v1']
        else:
            return pars['v2']

    if pars['mode'] == 'cycling':
        Omega = pars['Omega']
        v0 = pars['v0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return v0 * (1 - variation * np.cos(Omega * (t - time_lag)))

    else:
        return 0.


@np.vectorize
def alpha(t, pars):
    if pars['mode'] == 'const':
        return pars['alpha0']

    if pars['mode'] == 'cycling':
        Omega = pars['Omega']
        alpha0 = pars['alpha0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return alpha0 * (1. - variation * np.cos(Omega * (t - time_lag)))

    if pars['mode'] == 'cycling_steps':
        Omega = pars['Omega']
        alpha0 = pars['alpha0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return alpha0 * (1. - variation * np.tanh(10. * np.cos(Omega * (t - time_lag))))

    if pars['mode'] == 'piecewise':
        if t < pars['t0']:
            return pars['alpha1']
        else:
            return pars['alpha2']

    if pars['mode'] == 'spatial change':
        if t < pars['t0']:
            return pars['alpha0']
        else:
            tb_pars = pars['tb_pars']
            x_change = int(tb(pars['t1'], tb_pars))
            x = np.arange(pars['len'])-x_change
            a = pars['alpha1'] + (pars['alpha2'] - pars['alpha1'])/(1+ np.exp(x/pars['L']))
            return a

    if pars['mode'] == 'front':
        if t < pars['t0']:
            return pars['alpha0']
        else:
            tb_pars = pars['tb_pars']
            x_change = int(tb(t, tb_pars)) + pars['x0']
            x = np.arange(pars['len'])-x_change
            a = pars['alpha1'] + (pars['alpha2'] - pars['alpha1'])/(1+ np.exp(x/50))
            return a

    else:
        return 0.


@np.vectorize
def omega(t, pars):
    if pars['mode'] == 'const':
        return pars['omega0']

    if pars['mode'] == 'cycling':
        Omega = pars['Omega']
        omega0 = pars['omega0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return omega0 * (1. - variation * np.cos(Omega * (t - time_lag)))

    if pars['mode'] == 'cycling_steps':
        Omega = pars['Omega']
        omega0 = pars['omega0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return omega0 * (1. - variation * np.tanh(10. * np.cos(Omega * (t - time_lag))))


    if pars['mode'] == 'piecewise':
        if t < pars['t0']:
            return pars['omega1']
        else:
            return pars['omega2']

    if pars['mode'] == 'spatial change':
        if t < pars['t0']:
            return pars['omega1']
        else:
            tb_pars = pars['tb_pars']
            x_change = int(tb(pars['t0'], tb_pars))
            a = np.ones(pars['len'])
            a[x_change:] = pars['omega1']
            a[:x_change] = pars['omega2']
            return a

    else:
        return 0.


def epsilon(t, pars):
    if pars['mode'] == 'const':
        return pars['epsilon0']

    if pars['mode'] == 'piecewise':
        if t < pars['t0']:
            return pars['epsilon1']
        else:
            return pars['epsilon2']

    if pars['mode'] == 'cycling':
        Omega = pars['Omega']
        epsilon0 = pars['epsilon0']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return epsilon0 * (1. - variation * np.cos(Omega * (t - time_lag)))

    if pars['mode'] == 'double cycling':
        Omega1 = pars['Omega1']
        Omega2 = pars['Omega2']
        variation1 = pars['variation1']
        variation2 = pars['variation2']
        epsilon0 = pars['epsilon0']
        time_lag1 = pars['time_lag1']
        time_lag2 = pars['time_lag2']
        return epsilon0*(1 - variation1 * np.cos(Omega1 * (t - time_lag1)) - variation2 * np.cos(Omega2 * (t - time_lag2)))
    else:
        return 0.




@np.vectorize
def delta_star(t, pars):
    if pars['mode'] == 'const':
        return pars['delta0']

    if pars['mode'] == 'linear':
        return pars['delta0'] - pars['slope']*t

    if pars['mode'] == 'cycling':
        Omega = pars['Omega']
        slope = pars['slope']
        variation = pars['variation']
        time_lag = pars['time_lag']
        return pars['delta0'] - slope * (t - variation / Omega * (np.sin(Omega * (t - time_lag)) + np.sin(Omega * time_lag)))


    if pars['mode'] == 'piecewise':
        if t < pars['t0']:
            return pars['delta0']
        else:
            return pars['delta0'] - pars['slope'] * (t - pars['t0'])

    else:
        return 1.
# slope = 0.03





# ________ Numerical integration _____________________________________________

# function for integration (RHS)
def f_alpha(t, phi, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars, hill_coeff):
    dpdt = np.zeros(len(phi))
    phi0 = phi[0]  # the first element of the array is always kept for the reference oscillator
    delta_phi = phi - phi0
    tb_loc = int(tb(t, tb_pars))
    omega_t = omega(t, omega_pars)
    alpha_t = alpha(t, alpha_pars)
    epsilon_t = epsilon(t, epsilon_pars)
    delta_star_t = delta_star(t, delta_star_pars)
    if tb_loc>0:  # if tb is within the field of view
        # PSM:
        dpdt[tb_loc:] = ((omega_t - epsilon_t + alpha_t*delta_phi) * hill(np.abs(delta_phi), delta_star_t, hill_coeff))[tb_loc:]
        # fantom cells before tb oscillate as reference:
        dpdt[:tb_loc] = omega_t
    else:  # tb is past the field of view
        # all cells in PSM:
        dpdt = (omega_t - epsilon_t + alpha_t*delta_phi) * hill(np.abs(delta_phi), delta_star_t, hill_coeff)
        # reference:
        dpdt[0] = omega_t
    return dpdt



def f_saturation(t, phi, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars, hill_coeff):
    dpdt = np.zeros(len(phi))
    phi0 = phi[0]  # the first element of the array is always kept for the reference oscillator
    delta_phi = phi - phi0
    tb_loc = int(tb(t, tb_pars))
    omega_t = omega(t, omega_pars)
    alpha_t = alpha(t, alpha_pars)
    epsilon_t = epsilon(t, epsilon_pars)
    delta_star_t = delta_star(t, delta_star_pars)
    satur = 1 - np.abs(delta_phi)/delta_star_pars['delta0']
    deriv = (omega_t  + ( alpha_t*delta_phi - epsilon_t ) * satur) * hill(np.abs(delta_phi), delta_star_t, hill_coeff)
    if tb_loc>0:  # if tb is within the field of view
        # PSM:
        dpdt[tb_loc:] = deriv[tb_loc:]
        # fantom cells before tb oscillate as reference:
        dpdt[:tb_loc] = omega_t
    else:  # tb is past the field of view
        # all cells in PSM:
        dpdt = deriv
        dpdt[0] = omega_t   # reference osc
    return dpdt


def f_saturation_no_front(t, phi, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars, hill_coeff):
    dpdt = np.zeros(len(phi))
    phi0 = phi[0]  # the first element of the array is always kept for the reference oscillator
    delta_phi = phi - phi0
    tb_loc = int(tb(t, tb_pars))
    omega_t = omega(t, omega_pars)
    alpha_t = alpha(t, alpha_pars)
    epsilon_t = epsilon(t, epsilon_pars)
    delta_star_t = delta_star(t, delta_star_pars)
    satur = 1 - np.abs(delta_phi)/(2*np.pi)
    deriv = omega_t  + ( alpha_t*delta_phi - epsilon_t ) * satur
    if tb_loc>0:  # if tb is within the field of view
        # PSM:
        dpdt[tb_loc:] = deriv[tb_loc:]
        # fantom cells before tb oscillate as reference:
        dpdt[:tb_loc] = omega_t
    else:  # tb is past the field of view
        # all cells in PSM:
        dpdt = deriv
        dpdt[0] = omega_t   # reference osc
    return dpdt


# main function
def alpha_model(xmax, init_cond, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars,
                hill_coeff=6, timesteps=300, tmax=120, sigmoid=False):
    phi_map = np.zeros((xmax, timesteps))
    dphi_map = np.zeros((xmax, timesteps))
    times = np.linspace(0, tmax, timesteps)
    if len(init_cond) == xmax:
        initial = np.zeros(xmax+1)
        initial[1:] = init_cond
    else:
        print('error in initial condition, using zeros')
        initial = np.zeros(xmax+1)
    if sigmoid:
        f = f_saturation
    else:
        f = f_alpha
    ans = integrate.solve_ivp(f, [0.,tmax], initial, args=(omega_pars, alpha_pars, tb_pars, epsilon_pars,
                                                           delta_star_pars, hill_coeff),
                              dense_output=True, method='LSODA', rtol=1e-5, atol=1e-7)
    sol = ans.sol(times)
    phi0 = sol[0, :]
    phi_map = sol[1:, :]
    tb_locs = np.maximum(tb(times, tb_pars), np.zeros(len(times)))
    for it in range(timesteps):
        phi_map[:int(tb_locs[it]), it] = np.nan
    for x in range(xmax):
        dphi_map[x,:] = phi_map[x,:] - phi0
    return phi_map, dphi_map, phi0, tb_locs, times

def alpha_model_w_front(xmax, init_cond, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars,
                        hill_coeff=None, timesteps=300, tmax=120, sigmoid=False):

    phi_map = np.zeros((xmax, timesteps))
    dphi_map = np.zeros((xmax, timesteps))
    times = np.linspace(0, tmax, timesteps)
    if len(init_cond) == xmax:
        initial = np.zeros(xmax+1)
        initial[1:] = init_cond
    else:
        print('error in initial condition, using zeros')
        initial = np.zeros(xmax+1)
    if sigmoid:
        f = f_saturation_no_front
    else:
        f = f_alpha
    ans = integrate.solve_ivp(f, [0.,tmax], initial, args=(omega_pars, alpha_pars, tb_pars, epsilon_pars,
                                                           delta_star_pars, hill_coeff),
                              dense_output=True, method='LSODA', rtol=1e-5, atol=1e-7)
    sol = ans.sol(times)
    phi0 = sol[0, :]
    phi_map = sol[1:, :]
    tb_locs = np.maximum(tb(times, tb_pars), np.zeros(len(times)))
    for it in range(timesteps):
        phi_map[:int(tb_locs[it]), it] = np.nan
    for x in range(xmax):
        dphi_map[x,:] = phi_map[x,:] - phi0

    dphi_map[0, 0] = 0.
    delta_star_t = delta_star(times, delta_star_pars)

    x = np.arange(xmax)
    tt, xx = np.meshgrid(times, x, indexing='xy')
    x_2pi_t = xx[:, 0][np.nanargmin(np.abs(np.abs(dphi_map) - delta_star_t), axis=0)]
    phi_map[xx > x_2pi_t] = np.nan

    # Segments - imprint of front phase
    tl = np.nan * phi_map
    m = ~np.isnan(phi_map)
    nrows, ncols = phi_map.shape
    last_j = np.where(m, np.arange(ncols), -1).max(axis=1)
    for i, j in enumerate(last_j):
        if 0 <= j < ncols - 1:
            tl[i, j + 1:] = phi_map[i, j]

    return phi_map, tl, phi0, tb_locs, times
# _________________________________________________________________________________________________________



# ___________ Analytic approximation __________________________


def tau(t, A, Omega):
    return t - A/Omega*np.sin(Omega*t)


def tau_fit_1(x, a0, a1, b1, A, Omega):
    tau_phase = Omega*tau(x, A, Omega)
    return a0 + a1*np.cos(tau_phase) + b1*np.sin(tau_phase)


def deltaphi_1(tau_tau0, epsilon0, alpha0):
    return epsilon0/alpha0*(np.exp(alpha0*tau_tau0)-1)

# def deltaphi_2(tau, tau0, a1, b1, epsilon0, alpha0, Omega):
#     amplitude = epsilon0/np.sqrt(alpha0**2 + Omega**2)
#     ratio = Omega/alpha0
#     term1 = a1*amplitude*exp_modulated(tau, tau0, alpha0, Omega, np.arctan(ratio))
#     term2 = b1*amplitude*exp_modulated(tau, tau0, alpha0, Omega, np.arctan(-1/ratio))
#     return term1 + term2


def front(segmented_idx, times, taus_t, t0, tau0, x_abs, xsteps, delta_star_t, epsilon_pars, alpha_pars, a1, b1, plot_vf_components=False):
    front_times = np.zeros(xsteps)
    front_locs = np.zeros(xsteps)
    v_f_approx_0 = np.zeros(xsteps)
    v_f_approx = np.zeros(xsteps)
    v_f_num = np.zeros(xsteps)
    v_f_den = np.zeros(xsteps)
    h_tau0 = np.zeros(xsteps)

    Omega = alpha_pars['Omega']
    epsilon0 = epsilon_pars['epsilon0']
    alpha0 = alpha_pars['alpha0']
    c = np.sqrt(a1**2 + b1**2)
    den = 1 + alpha0/epsilon0*1.5*np.pi
    sqrt = np.sqrt(alpha0**2 + Omega**2)
    print('den:', den, 'c:', c, 'sqrt:', sqrt)
    for x in range(xsteps):
        if np.any(segmented_idx[x]):
            start_segmented = np.argmax(segmented_idx[x])
            #
            front_times[x] = times[start_segmented]
            front_locs[x] = x_abs[x]

            # growth_t0 = 1. - epsilon_pars['variation'] * np.cos(Omega * (times[t0[x]] - epsilon_pars['time_lag']))
            growth_t0 = (1. + a1*np.cos(Omega*tau0[x]) + b1*np.sin(Omega*tau0[x])) * (1 - alpha_pars['variation']*np.cos(Omega * times[start_segmented]))

            # den = 1 + alpha0/epsilon0*delta_star_t[start_segmented]
            cos_1 = np.cos(Omega*taus_t[start_segmented] + np.arctan(-b1/a1))

            ratio = Omega / alpha0
            phase = np.arctan((a1 * ratio - b1) / (a1 + b1 * ratio))
            if a1 + b1 * ratio < 0:
                phase += np.pi
            cos_2 = alpha0 * np.cos(Omega*taus_t[start_segmented] + phase)
            exp_sin = Omega * np.exp(alpha0*(taus_t[start_segmented]-tau0[x])) * np.sin(Omega*tau0[x]+phase)
            v_f_approx_0[x] = growth_t0
            v_f_approx[x] = growth_t0*(1 + c/den*cos_1)/(1 + c/den/sqrt*(exp_sin + cos_2))
            # v_f_approx[x] = growth_t0/(1 + c/den/sqrt)
            # v_f_approx[x] = growth_t0*(1 + c/den*cos_1)
            # v_f_approx[x] = (1 + c/den*cos_1)/(1 + c/den/sqrt*(exp_sin + cos_2))
            # v_f_approx[x] = growth_t0/(1 + c/den/sqrt*(exp_sin + cos_2))
            v_f_num[x] = (1 + c/den*cos_1)
            v_f_den[x] = (1 + c/den/sqrt*(exp_sin + cos_2))
            h_tau0[x] = (1. + a1*np.cos(Omega*tau0[x]) + b1*np.sin(Omega*tau0[x]))
        else:
            front_times[x] = np.nan
            v_f_approx[x] = np.nan
            front_locs[x] = np.nan
    if plot_vf_components:
        plt.figure(figsize=(8,4))
        plt.plot(front_times, v_f_approx_0, label= 'Growth(t0)', c='darkblue')
        plt.plot(front_times, h_tau0, label= 'h(tau0)', c='tab:blue')
        plt.plot(front_times, v_f_num, label= 'num', c = 'tab:green')
        plt.plot(front_times, 1/v_f_den, label= '1/den', c='tab:orange')
        plt.plot(front_times, v_f_approx, label= 'V_f approx', c = 'k')
        plt.ylim((0., 2))
        plt.title('Contributions to v_f')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    v_f_deriv = np.gradient(front_locs, front_times)
    return front_times, v_f_approx, v_f_approx_0, v_f_deriv


def alpha_model_analytic_X(xmax, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars,
                         timesteps=300, tmax=120, xsteps=500, order=1, plot_deltaphi=True, plot_h=False):
    # This model works only if v_tb and epsilon behave the same
    times = np.linspace(0, tmax, timesteps)
    x_abs = np.linspace(0, xmax, xsteps)
    tb_locs = tb(times, tb_pars)
    # print(tb_locs)
    # print('tb', tb_locs)
    X = np.tile(x_abs, (timesteps, 1)).T - np.tile(tb_locs, (xsteps, 1))
    X[X < 0] = np.nan

    taus_t = tau(times, omega_pars['variation'], omega_pars['Omega'])
    Tau = np.tile(taus_t, (xsteps, 1))
    phi0 = omega_pars['omega0']*Tau

    delta_star_t = delta_star(times, delta_star_pars)
    delta_star_t = np.tile(delta_star_t, (xsteps, 1))

    Omega = omega_pars['Omega']
    A1 = omega_pars['variation']

    epsilon_bar = epsilon_pars['epsilon0']/tb_pars['v0']

    alpha_t = alpha(times, alpha_pars)
    v_t = v_tb(times, tb_pars)
    beta_t = alpha_t/v_t
    beta0 = alpha_pars['alpha0']/tb_pars['v0']

    if order == 0:
        dphi = epsilon_bar / beta0 * (np.exp(beta0 * X) - 1.)
    elif order==1:
        variation = alpha_pars['variation']/100/Omega
        A_cycle = epsilon_bar / beta0 * (1. - variation*np.cos(Omega*times-np.pi/4))
        dphi = A_cycle * (np.exp(beta0 * X) - 1.)
        # dphi = epsilon_bar/beta_t * (np.exp(beta_t*X) - 1.)
    phi = phi0 - dphi
    segmented_idx = np.abs(dphi) > delta_star_t
    for x in range(xsteps):
        if np.any(segmented_idx[x]):
            start_segmented = np.argmax(segmented_idx[x])
            phi[x, start_segmented:] = phi[x, start_segmented]
    dphi[segmented_idx] = np.nan

    plt.imshow(np.cos(phi), origin='lower', cmap='magma')
    plt.title('Analytic')
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('t', fontsize=15)
    plt.ylabel('x', fontsize=15)

    if plot_h:
        plt.figure(figsize=(4,3))
        plt.plot(times[~np.isnan(X[0])], beta_t[~np.isnan(X[0])]/beta0-1, lw=2, alpha=0.6)

        plt.title('h(t)')
        plt.ylim(np.array([-1, 1]))
        plt.show()

        plt.figure(figsize=(4,3))
        plt.plot(X[0], beta_t/beta0-1, lw=2, alpha=0.6)
        plt.plot(X[50], beta_t/beta0-1, lw=2, alpha=0.6)
        plt.plot(X[100], beta_t/beta0-1, lw=2, alpha=0.6)

        plt.title('h(t(X, x=const))')
        plt.ylim(np.array([-1, 1]))
        plt.show()

        plt.figure(figsize=(4,3))
        plt.imshow(X, origin='lower')
        plt.title('X')
        plt.show()


    plt.show()
    return phi, dphi, phi0[0, :]



def deltaphi_2(tau, tau0, a1, b1, epsilon0, alpha0, Omega):
    amplitude = epsilon0*np.sqrt(a1**2 + b1**2)/np.sqrt(alpha0**2 + Omega**2)
    ratio = Omega/alpha0
    phase = np.arctan((a1*ratio - b1)/(a1 + b1*ratio))
    if a1 + b1*ratio < 0:
        phase += np.pi
    print('Delta phi phase shift:', phase)
    term = amplitude * (np.exp(alpha0*(tau-tau0))*np.cos(Omega*tau0 + phase) - np.cos(Omega*tau + phase))
    return term


# _____________________________________________________________________________________________________________________
# ____________________________________________________________________________________________________________________
def alpha_model_analytic(xmax, omega_pars, alpha_pars, tb_pars, epsilon_pars, delta_star_pars,
                         timesteps=300, tmax=120, xsteps=500, order=1, plot_deltaphi=True, plot_h=True, plot_vf=False):
    phi_map = np.zeros((xsteps, timesteps))
    dphi_map = np.zeros((xsteps, timesteps))

    times = np.linspace(0, tmax, timesteps)
    x_abs = np.linspace(0, xmax, xsteps)
    taus_t = tau(times, omega_pars['variation'], omega_pars['Omega'])

    tb_locs = tb(times, tb_pars)
    # print(tb_locs)
    # print('tb', tb_locs)
    X = np.tile(x_abs, (timesteps, 1)).T - np.tile(tb_locs, (xsteps,1))
    X[X<0] = np.nan
    t0 = np.argmax(np.isfinite(X), axis = 1)
    tau0 = taus_t[t0]
    Tau0 = np.tile(tau0, (timesteps, 1)).T
    Tau = np.tile(taus_t, (xsteps, 1))
    elapsed_Tau = Tau - Tau0
    non_psm_idx = elapsed_Tau<0
    elapsed_Tau[non_psm_idx] = np.nan

    delta_star_t = delta_star(times, delta_star_pars)
    delta_star_t = np.tile(delta_star_t, (xsteps, 1))

    Omega = omega_pars['Omega']
    A1 = omega_pars['variation']
    f = 1. - A1*np.cos(Omega*times)
    g = 1. - epsilon_pars['variation']*np.cos(Omega*(times-epsilon_pars['time_lag']))
    pars, pcov = curve_fit(lambda x, a0, a1, b1: tau_fit_1(x, a0, a1, b1, A1, Omega), times, g/f)
    a1 = pars[1]
    b1 = pars[2]
    print(pars, np.sqrt(a1**2+b1**2))

    if plot_h:
        plt.figure(figsize=(8,3))
        plt.plot(times, g/f, label=r'$h(\tau(t))$')
        plt.plot(times, tau_fit_1(times, *pars, A1, Omega), '--', label = 'Fit')
        # plt.plot(times, f)
        # plt.plot(times, g)
        plt.xlabel('t')
        plt.ylim(0., 2)
        plt.legend()
        plt.show()

    phi0 = omega_pars['omega0']*Tau

    deltaphi1 = deltaphi_1(elapsed_Tau, epsilon_pars['epsilon0'], alpha_pars['alpha0'])
    deltaphi2 = deltaphi_2(Tau, Tau0, a1, b1, epsilon_pars['epsilon0'], alpha_pars['alpha0'], Omega)
    deltaphi = deltaphi1 + order*deltaphi2

    plt.figure(figsize=(4, 3))

    # deltaphi_zoom = ndimage.zoom(deltaphi, 2)
    # plt.imshow(deltaphi_zoom, cmap='magma')
    # plt.show()
    # plt.contour(deltaphi, levels=(0, delta_star_pars['delta0']/4., delta_star_pars['delta0']/2., 3*delta_star_pars['delta0']/4., delta_star_pars['delta0']))
    deltaphi_contours = plt.contour(deltaphi, levels=(0, delta_star_pars['delta0']/2., delta_star_pars['delta0']))
    front_contour = deltaphi_contours.allsegs[2][0]
    plt.plot(front_contour[:, 0], front_contour[:, 1], '--', c='k')

    # print('Contour:', front_contour)
    plt.show()
    # plt.plot(front_contour[100:150, 0], front_contour[100:150, 1], '.-', c='k', lw=1)
    # plt.show()
    # front_contour_x = moving_average(front_contour[:, 1], 5)
    v_f_contour = np.abs(np.gradient(front_contour[:, 1]*xmax/xsteps, front_contour[:, 0]*tmax/timesteps))/tb_pars['v0']
    # v_f_contour /= np.nanmean(v_f_contour)
    # v_f_contour = moving_average(v_f_contour, 11)
    # plt.plot(front_contour[:, 0], v_f_contour)
    # plt.scatter(front_contour[:, 0], front_contour[:, 1], s=1)
    # plt.show()

    phi = phi0 - deltaphi
    segmented_idx = np.abs(deltaphi) > delta_star_t
    for x in range(xsteps):
        if np.any(segmented_idx[x]):
            start_segmented = np.argmax(segmented_idx[x])
            phi[x, start_segmented:] = phi[x, start_segmented]

    front_times, v_f_approx, v_f_approx_0, v_f_deriv = front(segmented_idx, times, taus_t, t0, tau0, x_abs, xsteps,
                                                             delta_star_t, epsilon_pars, alpha_pars, a1, b1, plot_vf_components=plot_vf)

    # phi[segmented_idx] = np.nan
    deltaphi1[segmented_idx] = np.nan
    deltaphi1[non_psm_idx] = np.nan
    deltaphi2[segmented_idx] = np.nan
    deltaphi2[non_psm_idx] = np.nan
    #
    if plot_deltaphi:
        fig, ax = plt.subplots(1, 2, figsize=(11,5))
        ax[0].imshow(deltaphi1, origin='lower', cmap='magma', vmin=0, vmax=delta_star_pars['delta0'])
        # plt.colorbar()
        # plt.show()
        ax[1].imshow(deltaphi2, origin='lower', cmap='magma', vmin=-delta_star_pars['delta0']/2, vmax=delta_star_pars['delta0']/2)
        # plt.colorbar()
        plt.show()


    # print(segmented_idx)
    front_idx = x_abs[np.argmax(segmented_idx, axis=0)]
    # plt.figure(figsize=(4, 3))
    # plt.plot(front_idx, '.-')
    # plt.plot(tb_locs, '.-')
    # plt.title('Kymo edges')
    # plt.ylim((380, 400))
    # plt.xlim((200, 650))

    plt.show()

    t_start_front = np.argmax(front_idx)
    front_loc = moving_average(front_idx[t_start_front:], 21)
    tb_loc = moving_average(tb_locs, 3)
    v_f = np.abs(np.gradient(front_loc, times[t_start_front:]))/tb_pars['v0']  #/np.mean(np.abs(np.gradient(front_loc))[150:-150])


    plt.figure(figsize=(8, 4))
    plt.plot(times, f, label='omega', c='tab:green', lw=3)
    plt.plot(times, np.abs(np.gradient(tb_loc))/np.mean(np.abs(np.gradient(tb_loc))), c='tab:blue', label='growth')
    plt.plot(times, g, label='epsilon', c='tab:purple')
    plt.plot(times[t_start_front:], v_f, c='tab:orange', label='front numerical')
    plt.legend()
    plt.xlabel('Time')
    plt.ylim(0.5, 1.5)
    plt.show()

    if plot_vf:
        plt.figure(figsize=(8, 4))
        plt.plot(times[t_start_front:], v_f, c='tab:orange', label='front numerical', zorder=1)
        # plt.plot(front_contour[:, 0]*tmax/timesteps, v_f_contour, c='brown', label='Front contour', lw=3)
        # plt.plot(times[t_start_front:], v_f/f[t_start_front:], c='indianred', label='Wavelength', zorder=0)
        plt.plot(front_times, v_f_approx, c='k', label='Front approx', lw=1.5)
        plt.plot(front_times, v_f_approx_0, '--', c='k', label='Growth(t0)', lw=1.5)
        plt.xlabel('Time')
        # plt.plot(front_times, v_f_deriv-np.nanmean(v_f_deriv), c='lightcoral', label='Front deriv')
        plt.legend()
        plt.ylim(0., 2.)
        plt.show()

    # segmented_idx[non_psm_idx] = True
    # plt.figure(figsize=(4,3))
    # plt.plot(times, np.sum(~segmented_idx.astype('int'), axis=0))
    # plt.title('PSM length')
    # plt.show()
    # plt.imshow(segmented_idx, origin='lower')
    # plt.show()

    # plt.figure(figsize=(4, 4))
    plt.imshow(np.cos(phi), origin='lower', cmap='magma')
    plt.title('Analytic')
    plt.xticks(())
    plt.yticks(())
    plt.xlabel('t', fontsize=15)
    plt.ylabel('x', fontsize=15)

    plt.show()
    return phi, deltaphi, phi0[0, :]







