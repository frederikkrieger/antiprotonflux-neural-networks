import cython
from libc.math cimport exp, sinh, sqrt
from scipy.special import jn_zeros
from numpy import pi

# constants
m_max = 1000
zeta_n = jn_zeros(0, m_max)
R = 20.0        # kpc
rho_sun = 0.3   # GeV/cm^3
r_c = 0.5       # kpc


@cython.cdivision(True)
cdef double integrand1_iso(int n, double[2] args):
    
    # constants
    r_s = 4.38      # kpc
    rho_s = 1.387   # GeV/cm^3
    
    # r-coordinate
    r = args[0]

    # coefficients
    n_ = int(args[1])

    # computation of factors
    arg1 = r * j0(zeta_n[n_-1]*r/R)
    arg2 = (rho_s / (1 + (r/r_s)**2)) / rho_sun

    return arg1 * arg2**2


@cython.cdivision(True)
cdef double integrand1_bur(int n, double[2] args):
    
    # constants
    r_s = 12.67     # kpc
    rho_s = 0.712   # GeV/cm^3
    
    # r-coordinate
    r = args[0]

    # coefficients
    n_ = int(args[1])

    # computation of factors
    arg1 = r * j0(zeta_n[n_-1]*r/R)
    arg2 = (rho_s / ((1+r/r_s)*(1+(r/r_s)**2))) / rho_sun

    return arg1 * arg2**2 


@cython.cdivision(True)
cdef double integrand1_nfw(int n, double[2] args):
    
    # constants
    r_s = 24.42
    rho_s = 0.184
    rho_c = rho_s*r_s/r_c * (1 + r_c/r_s)**-2
    gamma = 1
    eta = 3/(3-2*gamma)  
    
    # r-coordinate
    r = args[0]

    # coefficients
    n_ = int(args[1])

    # computation of factors
    arg1 = r * j0(zeta_n[n_-1]*r/R)
    arg2 = 0.0

    if r > r_c:
        arg2 = rho_s*r_s/r * (1 + r/r_s)**-2
    else:
        arg2 = rho_c * sqrt(1 + (2*pi**2/3 * (eta-1) * (1-(pi*r/r_c)**2/6 + (pi*r/r_c)**4/120)**2))

    return arg1 * arg2**2 


@cython.cdivision(True)
cdef double integrand1_moo(int n, double[2] args):
    
    # constants
    r_s = 30.28
    rho_s = 0.105
    rho_c = rho_s*(r_s/r_c)**1.16 * (1+r_c/r_s)**-1.84
    gamma = 1.16
    eta = 3/(3-2*gamma)
    
    # r-coordinate
    r = args[0]

    # coefficients
    n_ = int(args[1])

    # computation of factors
    arg1 = r * j0(zeta_n[n_-1]*r/R)
    arg2 = 0.0

    if r > r_c:
        arg2 = rho_s*(r_s/r)**1.16 * (1+r/r_s)**-1.84
    else:
        arg2 = rho_c * sqrt(1 + (2*pi**2/3 * (eta-1) * (1-(pi*r/r_c)**2/6 + (pi*r/r_c)**4/120)**2))

    return arg1 * arg2**2 


@cython.cdivision(True)
cdef double integrand1_ein(int n, double[2] args):
    
    # constants
    alpha = 0.17
    r_s = 28.44
    rho_s = 0.033
    rho_c = rho_s*exp(-2/alpha * ((r_c/(10*r_s))**alpha - 1))

    # r-coordinate
    r = args[0]

    # coefficients
    n_ = int(args[1])

    # computation of factors
    arg1 = r * j0(zeta_n[n_-1]*r/R)
    arg2 = 0.0

    if r > r_c/10:
        arg2 = rho_s*exp(-2/alpha * ((r/r_s)**alpha - 1))
    else:
        arg2 = rho_c

    return arg1 * arg2**2 


@cython.cdivision(True)
cdef double integrand1_eib(int n, double[2] args):
    
    # constants
    alpha = 0.11
    r_s = 35.24
    rho_s = 0.021
    rho_c = rho_s*exp(-2/alpha * ((r_c/(10*r_s))**alpha - 1))

    # r-coordinate
    r = args[0]

    # coefficients
    n_ = int(args[1])

    # computation of factors
    arg1 = r * j0(zeta_n[n_-1]*r/R)
    arg2 = 0.0

    if r > r_c/10:
        arg2 = rho_s*exp(-2/alpha * ((r/r_s)**alpha - 1))
    else:
        arg2 = rho_c

    return arg1 * arg2**2 


@cython.cdivision(True)
cdef double integrand2(int n, double[4] args):
    
    # z-coordinate
    z = args[0]

    # coefficients
    Z = args[1]
    a = args[2]
    b = args[3]

    return exp(a*(Z-z)) * sinh(b*(Z-z))
