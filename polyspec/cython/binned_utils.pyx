#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport M_PI, sqrt, pow as dpow
from cython.parallel import prange

cdef extern from "complex.h" nogil:
    double creal(double complex)
    double cimag(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] assemble_b3_all(double complex[:,:,::1] p1_H_maps, double complex[:,:,::1] m2_H_maps, int[:,::1] all_bins, int[:] chi_arr, int[:] sym_factor, int N_b, int nthreads):
    """Assemble the cubic term of the bispectrum numerator given filtered real-space maps."""
    cdef int u1,u2,u3,bin1,bin2,bin3,p_u
    cdef long i, n = p1_H_maps.shape[2]
    cdef int index, chi_index, full_index
    cdef double tmp_sum
    cdef np.ndarray[np.float64_t,ndim=1] b3 = np.zeros(N_b, dtype=np.float64)

    for full_index in prange(N_b,nogil=True,schedule='static',num_threads=nthreads):
        index = full_index%len(all_bins)
        chi_index = full_index//len(all_bins)

        # Define which bin we're in
        u1 = all_bins[index,0]
        u2 = all_bins[index,1]
        u3 = all_bins[index,2]
        bin1 = all_bins[index,3]
        bin2 = all_bins[index,4]
        bin3 = all_bins[index,5]
        p_u = all_bins[index,6]
        
        # Compute combination of fields
        tmp_sum = 0.
        if p_u*chi_arr[chi_index]==-1:
            for i in xrange(n):
                tmp_sum = tmp_sum-cimag(p1_H_maps[bin1,u1,i]*p1_H_maps[bin2,u2,i]*m2_H_maps[bin3,u3,i]+p1_H_maps[bin2,u2,i]*p1_H_maps[bin3,u3,i]*m2_H_maps[bin1,u1,i]+p1_H_maps[bin3,u3,i]*p1_H_maps[bin1,u1,i]*m2_H_maps[bin2,u2,i])
        else:
            for i in xrange(n):
                tmp_sum = tmp_sum+creal(p1_H_maps[bin1,u1,i]*p1_H_maps[bin2,u2,i]*m2_H_maps[bin3,u3,i]+p1_H_maps[bin2,u2,i]*p1_H_maps[bin3,u3,i]*m2_H_maps[bin1,u1,i]+p1_H_maps[bin3,u3,i]*p1_H_maps[bin1,u1,i]*m2_H_maps[bin2,u2,i])

        b3[full_index] = tmp_sum/sym_factor[index]
    return b3


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=1] assemble_b1_all(double complex[:,:,::1] p1_H_maps, double complex[:,:,::1] m2_H_maps, double complex[:,:,::1] this_p1_H_maps, double complex[:,:,::1] this_m2_H_maps, int[:,::1] all_bins, int[:] chi_arr, int[:] sym_factor, int N_b, int nthreads):
    """Assemble the linear term of the bispectrum numerator given filtered real-space maps."""
    cdef int u1,u2,u3,bin1,bin2,bin3,p_u
    cdef long i, n = p1_H_maps.shape[2]
    cdef int index, chi_index, full_index
    cdef double tmp_sum
    cdef np.ndarray[np.float64_t,ndim=1] b1 = np.zeros(N_b, dtype=np.float64)

    for full_index in prange(N_b,nogil=True,schedule='static',num_threads=nthreads):
        index = full_index%len(all_bins)
        chi_index = full_index//len(all_bins)

        # Define which bin we're in
        u1 = all_bins[index,0]
        u2 = all_bins[index,1]
        u3 = all_bins[index,2]
        bin1 = all_bins[index,3]
        bin2 = all_bins[index,4]
        bin3 = all_bins[index,5]
        p_u = all_bins[index,6]
        
        # Compute combination of fields
        tmp_sum = 0.
        if p_u*chi_arr[chi_index]==-1:
            for i in xrange(n):
                tmp_sum = tmp_sum - cimag(p1_H_maps[bin1,u1,i]*this_p1_H_maps[bin2,u2,i]*this_m2_H_maps[bin3,u3,i]+p1_H_maps[bin2,u2,i]*this_p1_H_maps[bin3,u3,i]*this_m2_H_maps[bin1,u1,i]+p1_H_maps[bin3,u3,i]*this_p1_H_maps[bin1,u1,i]*this_m2_H_maps[bin2,u2,i])
                tmp_sum = tmp_sum - cimag(this_p1_H_maps[bin1,u1,i]*p1_H_maps[bin2,u2,i]*this_m2_H_maps[bin3,u3,i]+this_p1_H_maps[bin2,u2,i]*p1_H_maps[bin3,u3,i]*this_m2_H_maps[bin1,u1,i]+this_p1_H_maps[bin3,u3,i]*p1_H_maps[bin1,u1,i]*this_m2_H_maps[bin2,u2,i])
                tmp_sum = tmp_sum - cimag(this_p1_H_maps[bin1,u1,i]*this_p1_H_maps[bin2,u2,i]*m2_H_maps[bin3,u3,i]+this_p1_H_maps[bin2,u2,i]*this_p1_H_maps[bin3,u3,i]*m2_H_maps[bin1,u1,i]+this_p1_H_maps[bin3,u3,i]*this_p1_H_maps[bin1,u1,i]*m2_H_maps[bin2,u2,i])
        else:
            for i in xrange(n):            
                tmp_sum = tmp_sum + creal(p1_H_maps[bin1,u1,i]*this_p1_H_maps[bin2,u2,i]*this_m2_H_maps[bin3,u3,i]+p1_H_maps[bin2,u2,i]*this_p1_H_maps[bin3,u3,i]*this_m2_H_maps[bin1,u1,i]+p1_H_maps[bin3,u3,i]*this_p1_H_maps[bin1,u1,i]*this_m2_H_maps[bin2,u2,i])
                tmp_sum = tmp_sum + creal(this_p1_H_maps[bin1,u1,i]*p1_H_maps[bin2,u2,i]*this_m2_H_maps[bin3,u3,i]+this_p1_H_maps[bin2,u2,i]*p1_H_maps[bin3,u3,i]*this_m2_H_maps[bin1,u1,i]+this_p1_H_maps[bin3,u3,i]*p1_H_maps[bin1,u1,i]*this_m2_H_maps[bin2,u2,i])
                tmp_sum = tmp_sum + creal(this_p1_H_maps[bin1,u1,i]*this_p1_H_maps[bin2,u2,i]*m2_H_maps[bin3,u3,i]+this_p1_H_maps[bin2,u2,i]*this_p1_H_maps[bin3,u3,i]*m2_H_maps[bin1,u1,i]+this_p1_H_maps[bin3,u3,i]*this_p1_H_maps[bin1,u1,i]*m2_H_maps[bin2,u2,i])
        # else:
        #     for i in xrange(n):
        #         tmp_sum = tmp_sum+creal(p1_H_maps[bin1,u1,i]*p1_H_maps[bin2,u2,i]*m2_H_maps[bin3,u3,i]+p1_H_maps[bin2,u2,i]*p1_H_maps[bin3,u3,i]*m2_H_maps[bin1,u1,i]+p1_H_maps[bin3,u3,i]*p1_H_maps[bin1,u1,i]*m2_H_maps[bin2,u2,i])

        b1[full_index] = tmp_sum/sym_factor[index]
    return b1