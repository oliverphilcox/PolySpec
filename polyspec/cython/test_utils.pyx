#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs, M_PI, M_E, sqrt, exp, pow as dpow, sin
from cython.parallel import prange

cdef extern from "gsl/gsl_errno.h" nogil:
    void gsl_set_error_handler_off()
    
cdef extern from "gsl/gsl_sf_bessel.h" nogil:
    int gsl_sf_bessel_jl_steed_array(int, double, double*)

cdef extern from "gsl/gsl_sf_gamma.h" nogil:
    double gsl_sf_lngamma(double)
    
cdef extern from "gsl/gsl_sf_hyperg.h" nogil:
    double gsl_sf_hyperg_2F1(double, double, double, double)

cdef extern from "complex.h" nogil:
    double complex cpow(double complex, double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void p_integral_many(double[:] k_arr, double[:] Pzeta_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the p_l^X(r) integral with the trapezium rule. This is a special case for N transfer functions."""
    
    cdef int il, ik, ir, nk = len(k_arr), nr = _integs.shape[2], nl = lmax+1-lmin, itrans, ntrans = len(Tl_arr)
    cdef double[:] kprod = np.zeros((nk),dtype=np.float64)
    cdef double lpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kprod[ik] = 2./M_PI*k_arr[ik]*k_arr[ik]/2.*Pzeta_arr[ik]
    
    # Perform sum for each transfer function
    for itrans in xrange(ntrans):
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
                
            # Iterate over r
            for ir in xrange(nr):
                
                # Compute trapezium rule
                f_low = kprod[0]*Tl_arr[itrans,lmin+il,0]*jlkr[il,ir,0]
                for ik in xrange(1,nk):
                    f_high = kprod[ik]*Tl_arr[itrans,lmin+il,ik]*jlkr[il,ir,ik]
                    integs[lmin+il,itrans,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void q_integral_many(double[:] k_arr, double[:,:,::1] Tl_arr, double[:,:,::1] jlkr, 
                     int lmin, int lmax, int nthreads, np.ndarray[np.float64_t,ndim=3] _integs):
    """Compute the q_l^X(r) integral with the trapezium rule. This is a special case for N transfer functions."""
    
    cdef int il, ik, ir, nk = len(k_arr), nr = _integs.shape[2], nl = lmax+1-lmin, itrans, ntrans = len(Tl_arr)
    cdef double[:] kprod = np.zeros((nk),dtype=np.float64)
    cdef double lpref, f_low, f_high, ksum
    cdef double[:,:,::1] integs = _integs

    # Compute k-dependent piece
    for ik in prange(nk,nogil=True,schedule='static',num_threads=nthreads):
        kprod[ik] = 2./M_PI*k_arr[ik]*k_arr[ik]/2.
    
    # Perform sum for each polarization
    for itrans in xrange(ntrans):
        for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
            lpref = dpow(-1.,lmin+il)
            
            # Iterate over r
            for ir in xrange(nr):
                
                # Compute trapezium rule
                f_low = kprod[0]*Tl_arr[itrans,lmin+il,0]*jlkr[il,ir,0]
                for ik in xrange(1,nk):
                    f_high = kprod[ik]*Tl_arr[itrans,lmin+il,ik]*jlkr[il,ir,ik]
                    integs[lmin+il,itrans,ir] += lpref*(k_arr[ik]-k_arr[ik-1])*(f_low+f_high)
                    f_low = f_high

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,:,::1] zetaPQ(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:,:,::1] inv_Cl_mat, double[:,::1] legs, int lmin, int lmax, int nthreads):
    """Compute zetaPQ(r, r', mu) = Sum_{l,X,Y} 4pi/(2l+1) p_l^X(r) C^{-1,XY}_l q_l^Y(r') L_l(mu)."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(legs)
    cdef int il, ir, ijr, imu, jr, ipol, jpol
    cdef double PQsum=0.
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_mu = np.zeros((nmu,nr,nr),dtype=np.float64)
    
    # Precompute l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)/(4.*M_PI)

    # Compute (2l+1)/4pi u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]
    
    # Compute sum over l for each r, r', mu
    for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
        ir = ijr//nr
        jr = ijr%nr
        for imu in xrange(nmu):
            PQsum = 0.
            for il in xrange(nl):
                PQsum = PQsum + zetaPQ_l[il,ir,jr]*legs[imu,il]
            zetaPQ_mu[imu,ir,jr] = PQsum
    return zetaPQ_mu

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void inpaint_cython(long[:,::1] neighbors, double[:] this_map, int[:] this_mask, int n_average, int nthreads):
    cdef long i,ip=0,j,n,iteration,n_neighbors=neighbors.shape[0], map_size=this_map.shape[0],masked_size=neighbors.shape[1]
    cdef double mean, count
    cdef double[:] new_map = np.empty_like(this_map)

    for iteration in xrange(n_average):
        for ip in prange(masked_size, num_threads=nthreads, schedule='static', nogil=True):
            mean = 0
            count = 0
            for n in xrange(n_neighbors):
                j = neighbors[n,ip]
                if j==-1: continue
                mean = mean+this_map[j]
                count = count+1.0
            if count!=0:
                new_map[ip] = mean/count
        # Update map
        ip = 0
        for i in xrange(map_size):
            if this_mask[i]==1:
                this_map[i] = new_map[ip]
                ip += 1