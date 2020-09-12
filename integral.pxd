# We have to import the Bessel function j_0 from math.h (C Library)
cdef extern from "math.h":
    double j0(double x)

cdef double integrand1_iso(int, double[2])
cdef double integrand1_bur(int, double[2])
cdef double integrand1_nfw(int, double[2])
cdef double integrand1_moo(int, double[2])
cdef double integrand1_ein(int, double[2])
cdef double integrand1_eib(int, double[2])

cdef double integrand2(int, double[4])