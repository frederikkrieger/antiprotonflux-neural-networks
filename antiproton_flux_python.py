import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.integrate
from scipy.constants import c, m_p, e, parsec, year
from scipy.special import jn_zeros, jv, j0
from timeit import default_timer as timer
from matplotlib.lines import Line2D

# Defining all constants from the Cirelli paper

R = 20                                  # kpc
L = [1, 4, 15]                          # kpc
V_conv = [13.5, 12, 5]                  # km/s
K_diff_0 = [0.0016, 0.0112, 0.0765]     # kpc^2/Myr
delta = [0.85, 0.70, 0.46]
r_sun = 8.33                            # kpc
rho_sun = 0.3                           # GeV/cm^3
n_H = 1                                 # 1/cm^3
n_He = 0.07 * n_H                       # 1/cm^3
h = 0.1                                 # kpc
r_c = 0.5                               # kpc

m_p_GeV = 1e-9 * m_p*c**2 / e           # GeV

df = pd.read_table('data/AtProduction_antiprotons.dat', delim_whitespace=True)

# Maximum value for n in R(k)
m_max = 500
limit = 100

# Zeros of Bessel function
zeta_n = jn_zeros(0, m_max)


def integrand1_iso(r, n):
    r_s = 4.38      # kpc
    rho_s = 1.387   # GeV/cm^3
    
    # computation of factors
    arg1 = r * j0(zeta_n[n-1]*r/R)
    arg2 = (rho_s / (1 + (r/r_s)**2)) / rho_sun

    return arg1 * arg2**2


def integrand1_bur(r, n):
    r_s = 12.67     # kpc
    rho_s = 0.712   # GeV/cm^3

    # computation of factors
    arg1 = r * j0(zeta_n[n-1]*r/R)
    arg2 = (rho_s / ((1+r/r_s)*(1+(r/r_s)**2))) / rho_sun

    return arg1 * arg2**2 


def integrand1_nfw(r, n):
    r_s = 24.42
    rho_s = 0.184
    rho_c = rho_s*r_s/r_c * (1 + r_c/r_s)**-2
    gamma = 1
    eta = 3/(3-2*gamma)  

    # computation of factors
    arg1 = r * j0(zeta_n[n-1]*r/R)
    arg2 = 0.0

    if r > r_c:
        arg2 = rho_s*r_s/r * (1 + r/r_s)**-2
    else:
        arg2 = rho_c * np.sqrt(1 + (2*np.pi**2/3 * (eta-1) * (1-(np.pi*r/r_c)**2/6 + (np.pi*r/r_c)**4/120)**2))

    return arg1 * arg2**2 


def integrand1_moo(r, n):
    r_s = 30.28
    rho_s = 0.105
    rho_c = rho_s*(r_s/r_c)**1.16 * (1+r_c/r_s)**-1.84
    gamma = 1.16
    eta = 3/(3-2*gamma)

    # computation of factors
    arg1 = r * j0(zeta_n[n-1]*r/R)
    arg2 = 0.0

    if r > r_c:
        arg2 = rho_s*(r_s/r)**1.16 * (1+r/r_s)**-1.84
    else:
        arg2 = rho_c * np.sqrt(1 + (2*np.pi**2/3 * (eta-1) * (1-(np.pi*r/r_c)**2/6 + (np.pi*r/r_c)**4/120)**2))

    return arg1 * arg2**2 


def integrand1_ein(r, n):
    alpha = 0.17
    r_s = 28.44
    rho_s = 0.033
    rho_c = rho_s*np.exp(-2/alpha * ((r_c/(10*r_s))**alpha - 1))

    # computation of factors
    arg1 = r * j0(zeta_n[n-1]*r/R)
    arg2 = 0.0

    if r > r_c/10:
        arg2 = rho_s*np.exp(-2/alpha * ((r/r_s)**alpha - 1))
    else:
        arg2 = rho_c

    return arg1 * arg2**2 



def integrand1_eib(r, n):
    alpha = 0.11
    r_s = 35.24
    rho_s = 0.021
    rho_c = rho_s*np.exp(-2/alpha * ((r_c/(10*r_s))**alpha - 1))

    # computation of factors
    arg1 = r * j0(zeta_n[n-1]*r/R)
    arg2 = 0.0

    if r > r_c/10:
        arg2 = rho_s*np.exp(-2/alpha * ((r/r_s)**alpha - 1))
    else:
        arg2 = rho_c

    return arg1 * arg2**2 


def integrand2(z, Z, a, b):
    return np.exp(a*(Z-z)) * np.sinh(b*(Z-z))


def K_diff(K, K_diff_0, delta):
    p = (K**2 + 2 * m_p_GeV * K)**(1/2)
    beta = (1 - m_p_GeV**2 / (K + m_p_GeV)**2)**(1/2)
    res = K_diff_0 * beta * p**delta
    return res


def S_n(n, K, V_conv, K_diff_0, delta):
    V_conv_kpcMyr = V_conv*(1e6*year/parsec)
    res = (V_conv_kpcMyr**2 / K_diff(K, K_diff_0, delta)**2 + 4 *
           (zeta_n[n-1])**2 / R**2)**(1/2)
    return res


def Gamma_ann(K):
    v_p = c * (1 - m_p_GeV**2 / (K + m_p_GeV)**2)**(1/2)
    sigma_ann = 0.0
    if K < 15.5:
        # typo in Cirelli paper: last coefficient is 0.948, not 0.984
        # see L. C. Tan and L. K. Ng, J. Phys. G 9 (1983) 227.
        sigma_ann = 661 * (1 + 0.0115*K**-0.774 - 0.948*K**0.0151)
    else:
        sigma_ann = 36 * K**-0.5
    sigma_ann += 24.7*(1 + 0.584*K**-0.115 + 0.856*K**-0.566)
    res = (n_H + 4**(2/3)*n_He) * sigma_ann * v_p * 1e-25  # 1/s
    return res


def A_n(n, K, V_conv, L, K_diff_0, delta):
    V_conv_kpcMyr = (1e6*year/parsec)*V_conv
    S = S_n(n, K, V_conv, K_diff_0, delta)
    K_diff_ = K_diff(K, K_diff_0, delta)
    res = 2*h*Gamma_ann(K)*1e6*year + V_conv_kpcMyr + \
        K_diff_ * S / (np.tanh(S*L/2))
    return res


def y_n(n, K, profile, V_conv, Z, K_diff_0, delta):
    V_conv_kpcMyr = (1e6*year/parsec)*V_conv

    arg1 = 4 / (jv(1, zeta_n[n-1])**2 * R**2)

    # arg2 is the r-integral
    arg2 = 0.0
    if profile == 'Iso':
        arg2 = scipy.integrate.quad(integrand1_iso, 0, R, args=(n))[0]
    elif profile == 'Bur':
        arg2 = scipy.integrate.quad(integrand1_bur, 0, R, args=(n))[0]
    elif profile == 'NFW':
        arg2 = scipy.integrate.quad(integrand1_nfw, 0, R, args=(n))[0]
    elif profile == 'Moo':
        arg2 = scipy.integrate.quad(integrand1_moo, 0, R, args=(n))[0]
    elif profile == 'Ein':
        arg2 = scipy.integrate.quad(integrand1_ein, 0, R, args=(n))[0]
    elif profile == 'EiB':
        arg2 = scipy.integrate.quad(integrand1_eib, 0, R, args=(n))[0]

    # arg3 is the z-integral, implemented as exp(a*(Z-z)) * sinh(b*(Z-z)) with
    # a = V_conv/(2*K_diff), b = S_n/2
    a = V_conv_kpcMyr / (2*K_diff(K, K_diff_0, delta))
    b = S_n(n, K, V_conv, K_diff_0, delta) / 2
    arg3 = scipy.integrate.quad(
        integrand2, 0, Z, args=(Z, a, b))[0]

    return arg1 * arg2 * arg3


def R_K(K, profile, V_conv, L, K_diff_0, delta):
    V_conv_kpcMyr = (1e6*year/parsec)*V_conv
    res = 0.0
    for n in range(1, m_max):
        arg1 = jv(0, zeta_n[n-1] * r_sun/R)
        arg2 = np.exp(-V_conv_kpcMyr*L / (2*K_diff(K, K_diff_0, delta)))
        y = y_n(n, K, profile, V_conv, L, K_diff_0, delta)
        A = A_n(n, K, V_conv, L, K_diff_0, delta)
        arg3 = y / (A * np.sinh(S_n(n, K, V_conv, K_diff_0, delta) * L/2))
        temp_res = arg1 * arg2 * arg3
        if (res != 0) and (abs(temp_res/res) < 0.0005):
            break
        res += temp_res
    return res


def dNdK(channel, mass, K):
    logx = df['Log[10,x]'].drop_duplicates().to_numpy()
    K_list = (10**logx) * mass
    mDM = df['mDM'].drop_duplicates().to_numpy()
    flux = df[channel].to_numpy()

    flux_interp = []

    for i in range(179):
        flux_sub = flux[i::179]
        flux_interp.append(np.interp(mass, mDM, flux_sub))

    flux_interp = flux_interp * 1 / (K*np.log(10))

    dNdK = np.interp(K, K_list, flux_interp)

    return(dNdK)


def v(K):
    return c * (1 - m_p_GeV**2 / (K + m_p_GeV)**2)**(1/2)


def dphidK(K, mass, channel, cross_section, profile, V_conv, L, K_diff_0, delta):
    # return value has unit 1/(cm^2 s)
    arg1 = v(K)/(4*np.pi) * (rho_sun/mass)**2 * \
        R_K(K, profile, V_conv, L, K_diff_0, delta) * 1e6 * year

    arg2 = 1/2 * cross_section * dNdK(channel, mass, K)

    return arg1*arg2


def dphidlogK(K, mass, channel, cross_section, profile, V_conv, L, K_diff_0, delta):
    arg1 = v(K)/(4*np.pi) * (rho_sun/mass)**2 * \
        R_K(K, profile, V_conv, L, K_diff_0, delta) * 1e6 * year
    arg2 = 1/2 * cross_section * dNdK(channel, mass, K)
    return 1e6*K*np.log(10)*arg1*arg2
