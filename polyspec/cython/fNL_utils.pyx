#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs, M_PI, sqrt
from cython.parallel import prange

cdef extern from "<complex.h>" namespace "std" nogil:
    double complex pow(double complex, double complex)
    double real(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class fNL_utils:

    # Define local memviews
    cdef int [:] ls, ms, l_arr, m_arr, lmin_indices
    cdef int base_lmax, lmin, lmax, nl, nthreads, nr

    def __init__(self, int nthreads, int nr, np.ndarray[np.int32_t,ndim=1] l_arr, np.ndarray[np.int32_t,ndim=1] m_arr,
                 np.ndarray[np.int32_t,ndim=1] ls, np.ndarray[np.int32_t,ndim=1] ms):
        """Initialize the class with various l and L quantities."""

        self.l_arr = l_arr
        self.m_arr = m_arr
        self.ls = ls
        self.ms = ms
        self.lmin = min(ls)
        self.lmax = max(ls)
        self.nl = len(ls)
        self.nr = nr
        self.base_lmax = max(l_arr)
        self.nthreads = nthreads

        # Define indices for L array
        cdef int i, ip, jp
        cdef int ct=0
        for i in xrange(len(self.l_arr)):
            if self.l_arr[i]<=self.lmax: ct += 1
        self.lmin_indices = np.ones(ct,dtype=np.int32)*-1
        ip = -1
        jp = -1
        for i in xrange(len(self.l_arr)):
            if self.l_arr[i]>self.lmax: continue
            ip += 1
            if self.l_arr[i]<self.lmin: continue
            jp += 1
            self.lmin_indices[ip] = jp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] apply_fl_weights(self, double[:,:,::1] flX, complex[:,::1] h_lm, double weight):
        """Compute w*f_l^X(r)*h^X_lm, summing over polarizations X (assuming only T and/or E contribute)."""

        cdef int nlm = h_lm.shape[1], npol = flX.shape[1], nr = flX.shape[2]
        cdef int ir, ilm, l
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((nr,nlm),dtype=np.complex128)

        # Iterate over l, r and sum over polarizations
        if npol==1:
            for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
                for ilm in xrange(nlm):
                    l = self.ls[ilm]
                    out[ir, ilm] = flX[l,0,ir]*h_lm[0,ilm]*weight
        else:
            for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
                for ilm in xrange(nlm):
                    l = self.ls[ilm]
                    out[ir, ilm] = (flX[l,0,ir]*h_lm[0,ilm]+flX[l,1,ir]*h_lm[1,ilm])*weight  
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double fnl_sum(self, double[:] r_weights, double[:,::1] A_maps, double[:,::1] B_maps, double[:,::1] C_maps):
        """Compute Sum_i w_i A_i(r)B_i(r)C_i(r) for maps A, B, C."""
        cdef int ir, ipix, nr = A_maps.shape[0], npix = A_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + A_maps[ir,ipix]*B_maps[ir,ipix]*C_maps[ir,ipix]
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double neural_sum(self, double[:] r_weights, double[:] neural_weights, double[:,:,::1] A_maps, double[:,:,::1] B_maps, double[:,:,::1] C_maps):
        """Compute Sum_n W_n Sum_i w_i An_i(r)Bn_i(r)Cn_i(r) for maps A, B, C."""
        cdef int iterm, ir, ipix, nterm = A_maps.shape[0], nr = A_maps.shape[1], npix = A_maps.shape[2]
        cdef double tmp_sum, tmp_sum2, out = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum2 = 0.
            for iterm in xrange(nterm):
                tmp_sum = 0.
                for ipix in xrange(npix):
                    tmp_sum = tmp_sum + A_maps[iterm,ir,ipix]*B_maps[iterm,ir,ipix]*C_maps[iterm,ir,ipix]
                tmp_sum2 += neural_weights[iterm]*tmp_sum
            out += r_weights[ir]*tmp_sum2
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.float64_t,ndim=2] multiply(self, double[:,::1] map1, double[:,::1] map2):
        """Multiply two maps together in parallel"""
        cdef int n1 = map1.shape[0], n2 = map1.shape[1], i1, i2
        cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)

        for i1 in prange(n1,nogil=True,schedule='static',num_threads=self.nthreads):
            for i2 in xrange(n2):
                out[i1,i2] = map1[i1,i2]*map2[i1,i2]
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=2] radial_sum(self, complex[:,::1] lm_map, double[:] r_weights, double[:,:,::1] flXs):
        """Compute [Sum_r weight(r) f^X_l(r) A^X_lm(r)], where A is complex. """
        cdef int nlm = lm_map.shape[1], nr = lm_map.shape[0], npol = flXs.shape[1]
        cdef int ilm, ir, ipol, l
        cdef complex tmp_out
        cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((npol,nlm),dtype=np.complex128)

        with nogil:
            for ipol in xrange(npol):
                for ilm in prange(nlm, schedule='static', num_threads=self.nthreads):
                    l = self.ls[ilm]
                    tmp_out = 0.
                    for ir in xrange(nr):
                        tmp_out = tmp_out + lm_map[ir,ilm]*flXs[l,ipol,ir]*r_weights[ir]
                    out[ipol,ilm] = tmp_out
        return out