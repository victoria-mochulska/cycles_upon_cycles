
import numpy as np
from scipy.optimize import curve_fit


def exp_plus_const(x, a, b, c):
    return a*np.exp(b*x) + c

def exp_sigmoid(x, a, b, c, d):
    return a/(1+np.exp(-b*(x-d))) + c

def exponential(x, a, b):
    return a*np.exp(b*x)


def linear(x, a, b):
    return a*x + b


def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


def quadratic_only(x, a, c):
    return a*x**2 + c


def linear_plus_6h(x, a, b, c, d):
    T = 360
    return a*x + b*np.sin(2*np.pi*x/T + c) + d


def sine_6h(x, a, b, c):
    T = 360
    return a + b*np.cos(2*np.pi/T*x + c)

# ____________________________________
def allometric(x, a, b):
    return a*(x**b)

def arrhenius(x, a, E): # centered
    # x is temp/Tref
    # E is in units of k_b Tref
    # y(x) = a * np.exp(-E*(1/kT - 1/kTref) = a * np.exp(-E/kTref*(Tref/T - 1)
    return a * np.exp(-E*(1/x - 1))
# ____________________________________

def fit_function(f, xx, yy, p0=None, bounds=(-np.inf, np.inf), xx_fit=None, sigma=None):
    try:
        pars, cov = curve_fit(f, xx, yy, p0=p0, bounds=bounds, sigma=sigma)
        perr = np.sqrt(np.diag(cov))
        if xx_fit is None:
            xx_fit = xx
        fit = f(xx_fit, *pars)
    except RuntimeError:
        npars = f.__code__.co_argcount
        pars = np.ones(npars)*np.nan
        perr = pars.copy()
        fit = xx*np.nan
    return fit, pars, perr


def par_multiply(par1_tuple, par2_tuple, par3_tuple=(1., 0.)):
    par1 = par1_tuple[0]
    err1 = par1_tuple[1]
    par2 = par2_tuple[0]
    err2 = par2_tuple[1]
    par3 = par3_tuple[0]
    err3 = par3_tuple[1]

    par = par1 * par2 * par3
    err = np.sqrt((err1 * par2 * par3) ** 2 + (par1 * err2 * par3) ** 2 + (par1 * par2 * err3) ** 2)
    return par, err


def par_divide(par1_tuple, par2_tuple):
    par1 = par1_tuple[0]
    err1 = par1_tuple[1]
    par2 = par2_tuple[0]
    err2 = par2_tuple[1]

    par = par1 / par2
    err = np.sqrt((err1 / par2) ** 2 + (par / par2 * err2) ** 2)
    return par, err

