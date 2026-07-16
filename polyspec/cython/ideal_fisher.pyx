#cython: language_level=3

from __future__ import print_function
import numpy as np
import os
cimport numpy as np
cimport cython
from libc.math cimport M_PI, sqrt, pow as dpow
from cython.parallel import prange, threadid
from scipy.linalg.cython_blas cimport dgemm   # fast BLAS (OpenBLAS/MKL) for the mu-projection

cdef extern from "complex.h" nogil:
    double creal(double complex)
    double cimag(double complex)

## General Utilities
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] r2c(double[:,::1] re_arr, double[:,::1] im_arr, int nthreads):
    """Utility function to transform real/im parts of a 2D array to a complex array"""
    cdef int i, j, si = re_arr.shape[0], sj = re_arr.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out_arr = np.zeros((si, sj), dtype=np.complex128)
    for i in prange(si,nogil=True,schedule='static',num_threads=nthreads):
        for j in xrange(sj):
            out_arr[i,j] = re_arr[i,j]+1.0j*im_arr[i,j]
    return out_arr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] r2cstar(double[:,::1] re_arr, double[:,::1] im_arr, int nthreads):
    """Utility function to transform real/im parts of a 2D array to a complex array"""
    cdef int i, j, si = re_arr.shape[0], sj = re_arr.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out_arr = np.zeros((si, sj), dtype=np.complex128)
    for i in prange(si,nogil=True,schedule='static',num_threads=nthreads):
        for j in xrange(sj):
            out_arr[i,j] = re_arr[i,j]-1.0j*im_arr[i,j]
    return out_arr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void r2cstar_inplace(double[:,::1] re_arr, double[:,::1] im_arr, complex[:,:,::1] out_arr, int mu_index, int nthreads):
    """Utility function to transform real/im parts of a 2D array to a complex array"""
    cdef int i, j, si = re_arr.shape[0], sj = re_arr.shape[1]
    for i in prange(si,nogil=True,schedule='static',num_threads=nthreads):
        for j in xrange(sj):
            out_arr[mu_index,i,j] = re_arr[i,j]-1.0j*im_arr[i,j]
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRC(double[:,::1] arrR, complex[:,::1] arrC, int nthreads):
    """Multiply a real and a complex 2D map together."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arrR[i1,i2]*arrC[i1,i2]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRCstar(double[:,::1] arrR, complex[:,::1] arrC, double fac, int nthreads):
    """Multiply a real and a complex 2D map together, adding a conjugate."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = fac*arrR[i1,i2]*arrC[i1,i2].conjugate()
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRC_sym(double[:,::1] arrR, complex[:,::1] arrC, double[:,::1] arrR2, complex[:,::1] arrC2, int nthreads):
    """Multiply a real and a complex 2D map together and add to an output array."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arrR[i1,i2]*arrC[i1,i2]+arrR2[i1,i2]*arrC2[i1,i2]
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] multiplyRCstar_sym(double[:,::1] arrR, complex[:,::1] arrC, double[:,::1] arrR2, complex[:,::1] arrC2, double fac, int nthreads):
    """Multiply a real and a complex 2D map together and add to an output array, adding a conjugate."""
    cdef int i1, i2, n1 = arrR.shape[0], n2 = arrR.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1, n2), dtype=np.complex128)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = fac*(arrR[i1,i2]*arrC[i1,i2].conjugate()+arrR2[i1,i2]*arrC2[i1,i2].conjugate())
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] multiplyRR(double[:,::1] arr1, double[:,::1] arr2, int nthreads):
    """Multiply two real 2D maps together."""
    cdef int i1, i2, n1 = arr1.shape[0], n2 = arr2.shape[1]
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1, n2), dtype=np.float64)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arr1[i1,i2]*arr2[i1,i2]
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] multiplyRR_sym(double[:,::1] arr1, double[:,::1] arr2, double[:,::1] arr1b, double[:,::1] arr2b, int nthreads):
    """Multiply a real and a complex 2D map together and add to an output array."""
    cdef int i1, i2, n1 = arr1.shape[0], n2 = arr2.shape[1]
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1, n2), dtype=np.float64)
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            out[i1,i2] = arr1[i1,i2]*arr2[i1,i2]+arr1b[i1,i2]*arr2b[i1,i2]
    return out
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=1] lens_phi_sum(complex[:,::1] umap, complex[:,::1] vmap, int nthreads):
    """Compute the sum over U and V maps required for the lensing Phi estimator""" 
    cdef int i, ipol, npol = umap.shape[0], npix = umap.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=1] out = np.zeros(npix, dtype=np.complex128)

    for i in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
        if npol==1:
            # Spin-0
            out[i] = 2*umap[0,i]*vmap[0,i].conjugate()
        else:
            # All spins
            out[i] = 2*umap[0,i]*vmap[0,i].conjugate()+(umap[1,i]*vmap[1,i].conjugate()-umap[1,i].conjugate()*vmap[2,i])+1.0j*(umap[2,i]*vmap[1,i].conjugate()+umap[2,i].conjugate()*vmap[2,i])

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=1] lens_phi_sum_sym(complex[:,::1] umap1, complex[:,::1] umap2, complex[:,::1] vmap1, complex[:,::1] vmap2, int nthreads):
    """Compute the sum over U and V maps required for the lensing Phi estimator, symmetrizing over two sets of fields.""" 
    cdef int i, ipol, npol = umap1.shape[0], npix = umap1.shape[1]
    cdef np.ndarray[np.complex128_t,ndim=1] out = np.zeros(npix, dtype=np.complex128)
    
    for i in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
        if npol==1:
            # Spin-0
            out[i] = 2*(umap1[0,i]*vmap2[0,i].conjugate()+umap2[0,i]*vmap1[0,i].conjugate())
        else:   
            # All spins
            out[i] =  2*umap1[0,i]*vmap2[0,i].conjugate()+(umap1[1,i]*vmap2[1,i].conjugate()-umap1[1,i].conjugate()*vmap2[2,i])+1.0j*(umap1[2,i]*vmap2[1,i].conjugate()+umap1[2,i].conjugate()*vmap2[2,i])+2*umap2[0,i]*vmap1[0,i].conjugate()+(umap2[1,i]*vmap1[1,i].conjugate()-umap2[1,i].conjugate()*vmap1[2,i])+1.0j*(umap2[2,i]*vmap1[1,i].conjugate()+umap2[2,i].conjugate()*vmap1[2,i])
    
    return out
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productP_all(complex[:,:,::1] P_map, complex[:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar."""
    cdef int n1 = P_map.shape[1], n2 = P_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef double tmp_out
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)
    for i1 in prange(n1, nogil=True, schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out + weights[iw]*creal(P_map[pq_inds[iw],i1,i2]*F_PQ[F_inds[iw],i1,i2])
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productP_real_all(double[:,:,::1] P_map, complex[:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar."""
    cdef int n1 = P_map.shape[1], n2 = P_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef double tmp_out
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)
    # Parallelize if there's a lot of terms
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out + weights[iw]*P_map[pq_inds[iw],i1,i2]*creal(F_PQ[F_inds[iw],i1,i2])
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productPnmu_sym_all(complex[:,:,:,::1] Pnmu_maps, complex[:,:,:,::1] F_PQ, double[:] weights, int n1, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar. We sum over the mu axis, assuming symmetries."""
    cdef int na = Pnmu_maps.shape[2], nb = Pnmu_maps.shape[3], nw = len(weights)
    cdef int i1, i2, mu1, iw, ip, iF
    cdef double tmp_sum
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((na,nb),dtype=np.float64)
    for i1 in prange(na, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(nb):
            tmp_sum = 0.
            for iw in xrange(nw):
                ip = pq_inds[iw]
                iF = F_inds[iw]
                tmp_sum = tmp_sum + weights[iw]*creal(Pnmu_maps[ip,0,i1,i2]*F_PQ[iF,0,i1,i2])
                for mu1 in xrange(1,n1+1):
                    tmp_sum = tmp_sum + 2.* weights[iw]*creal(Pnmu_maps[ip,mu1,i1,i2]*F_PQ[iF,mu1,i1,i2])
            out[i1,i2] = tmp_sum
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productPnmu_all(complex[:,:,:,::1] Pnmu_maps, complex[:,:,:,::1] conjPnmu_maps, complex[:,:,:,::1] F_PQ, double[:] weights, int n1, int[:] pq_inds, int[:] F_inds, int nthreads):
    """Utility function to take the real product of two maps and a scalar. We sum over the mu axis, assuming symmetries."""
    cdef int na = Pnmu_maps.shape[2], nb = Pnmu_maps.shape[3]
    cdef int i1, i2, ip, iF, mu1, iw, nw = len(weights)
    cdef double tmp_sum
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((na,nb),dtype=np.float64)
    for i1 in prange(na, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(nb):
            tmp_sum = 0.
            for iw in xrange(nw):
                ip = pq_inds[iw]
                iF = F_inds[iw]
                tmp_sum = tmp_sum + weights[iw]*creal(Pnmu_maps[ip,0,i1,i2]*F_PQ[iF,n1,i1,i2])
                for mu1 in xrange(1,n1+1):
                    tmp_sum = tmp_sum + weights[iw]*creal(Pnmu_maps[ip,mu1,i1,i2]*F_PQ[iF,n1+mu1,i1,i2]+dpow(-1.,n1-mu1)*conjPnmu_maps[ip,mu1,i1,i2].conjugate()*F_PQ[iF,n1-mu1,i1,i2])
            out[i1,i2] = tmp_sum
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] to_plus_minus(double[:,:,::1] map, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map.shape[0], n2 = map.shape[2]
    cdef int i1, i2
    cdef double outR, outI
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            outR = map[i1,0,i2]
            outI = map[i1,1,i2]
            out[0,i1,i2] = outR+1.0j*outI
            out[1,i1,i2] = outR-1.0j*outI
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] to_plus_minus_complex(complex[:,:,::1] map, int spin, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map.shape[0], n2 = map.shape[2]
    cdef int i1, i2
    cdef complex outR, outI
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            outR = map[i1,0,i2]
            outI = map[i1,1,i2]
            out[0,i1,i2] = -(outR+1.0j*outI)
            if spin%2==0:
                out[1,i1,i2] = -(outR-1.0j*outI)
            else:
                out[1,i1,i2] = (outR-1.0j*outI)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] complex_to_complex(complex[:,::1] map1, complex[:,::1] map2, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map1.shape[1], n2 = map1.shape[0]
    cdef int i1, i2
    cdef complex plus, minus
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            plus = map1[i2,i1]
            minus = map2[i2,i1]
            out[0,i1,i2] = plus+1.0j*minus
            out[1,i1,i2] = plus-1.0j*minus
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=3] complex_to_complex_transpose(complex[:,::1] map1, complex[:,::1] map2, int nthreads):
    """Utility function to turn the Re/Im parts of a map to the + and - maps."""
    cdef int n1 = map1.shape[0], n2 = map1.shape[1]
    cdef int i1, i2
    cdef complex plus, minus
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((2,n1,n2),dtype=np.complex128)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            plus = map1[i1,i2]
            minus = map2[i1,i2]
            out[0,i1,i2] = plus+1.0j*minus
            out[1,i1,i2] = plus-1.0j*minus
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=3] to_real_imag(complex[:,::1] cmapP, complex[:,::1] cmapM, int nthreads):
    """Utility function to turn the +/- parts of a map to the Re and Im maps."""
    cdef int n1 = cmapP.shape[0], n2 = cmapP.shape[1]
    cdef int i1, i2
    cdef complex outP, outM
    cdef np.ndarray[np.float64_t,ndim=3] out = np.zeros((n1,2,n2),dtype=np.float64)
    for i1 in prange(n1, nogil=True, schedule='static', num_threads=nthreads):
        for i2 in xrange(n2):
            outP = cmapP[i1,i2]
            outM = cmapM[i1,i2]
            out[i1,0,i2] = creal(outP+outM)/2.
            out[i1,1,i2] = cimag(outP-outM)/2.
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] compute_productQ_real_all(double[:,:,::1] Q_map, complex[:,:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int mu1_index, int nthreads):
    """Utility function to take the real product of two maps and a scalar. We assume the maps are real."""
    cdef int n1 = Q_map.shape[1], n2 = Q_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef double tmp_out
    cdef np.ndarray[np.float64_t,ndim=2] out = np.zeros((n1,n2),dtype=np.float64)
    # Parallelize if large matrix
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out+weights[iw]*Q_map[pq_inds[iw],i1,i2]*creal(F_PQ[F_inds[iw],mu1_index,i1,i2])
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] compute_productQ_complex_all(double[:,:,::1] Q_map, complex[:,:,:,::1] F_PQ, double[:] weights, int[:] pq_inds, int[:] F_inds, int mu1_index, int nthreads):
    """Utility function to take the complex product of two maps and a scalar. We assume the first map is real."""
    cdef int n1 = Q_map.shape[1], n2 = Q_map.shape[2], nw = len(weights)
    cdef int i1, i2, iw
    cdef complex tmp_out
    cdef np.ndarray[np.complex128_t,ndim=2] out = np.zeros((n1,n2),dtype=np.complex128)
        
    for i1 in prange(n1,nogil=True,schedule='static',num_threads=nthreads):
        for i2 in xrange(n2):
            tmp_out = 0.
            for iw in xrange(nw):
                tmp_out = tmp_out + weights[iw]*Q_map[pq_inds[iw],i1,i2]*F_PQ[F_inds[iw],mu1_index,i1,i2]
            out[i1,i2] = tmp_out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq(double[:,:,::1] flXs, complex[:,::1] P_FPQ, long[:] ls, double[:] r_weights, complex[:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r)."""
    cdef int nlm = P_FPQ.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, l, ilm, ir
    cdef complex tmp_out
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                tmp_out = 0.
                for ir in xrange(nr):
                    tmp_out = tmp_out + 0.5*r_weights[ir]*flXs[l,ipol,inds[ir]]*P_FPQ[ilm,ir]
                out[ipol,ilm] += tmp_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq_complex(complex[:,:,::1] flXs, complex[:,::1] P_FPQ_plus, complex[:,::1] P_FPQ_minus, long[:] ls, double[:] r_weights, complex[:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r). We allow for two complex input maps."""
    cdef int nlm = P_FPQ_plus.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, l, ilm, ir
    cdef complex tmp_out
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                tmp_out = 0.
                for ir in xrange(nr):
                    tmp_out = tmp_out + 0.25*r_weights[ir]*(flXs[l,ipol,inds[ir]]*P_FPQ_plus[ilm,ir]+flXs[l,ipol,inds[ir]].conjugate()*P_FPQ_minus[ilm,ir])
                out[ipol,ilm] += tmp_out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq_deriv(double[:,:,::1] flXs, complex[:,::1] P_FPQ, long[:] ls, double[:] r_weights, complex[:,:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r)."""
    cdef int nlm = P_FPQ.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, ilm, l, ir
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                for ir in xrange(nr):
                    out[ir,ipol,ilm] += 0.5*r_weights[ir]*flXs[l,ipol,inds[ir]]*P_FPQ[ilm,ir]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void integrate_pq_complex_deriv(complex[:,:,::1] flXs, complex[:,::1] P_FPQ_plus, complex[:,::1] P_FPQ_minus, long[:] ls, double[:] r_weights, complex[:,:,::1] out, int[:] inds, int nthreads):
    """Utility function to sum over the radial axis of an array, weighted by f_l^X(r)."""
    cdef int nlm = P_FPQ_plus.shape[0], npol = flXs.shape[1], nr = len(inds)
    cdef int ipol, ilm, l, ir
    with nogil:
        for ipol in xrange(npol):
            for ilm in prange(nlm,schedule='static',num_threads=nthreads):
                l = ls[ilm]
                for ir in xrange(nr):
                    out[ir,ipol,ilm] += 0.5*r_weights[ir]*(flXs[l,ipol,inds[ir]]*P_FPQ_plus[ilm,ir]+flXs[l,ipol,inds[ir]].conjugate()*P_FPQ_minus[ilm,ir])/2.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] outer_product_tspec(complex[:,:,::1] Q4_a, complex[:,:,::1] Q4_b, int nthreads, bint sym):
    """Compute Fisher matrix between two large arrays as an outer product.
    
    If "sym" is specified, we assume a square symmetric matrix."""
    
    cdef int n_out1 = Q4_a.shape[1], n_out2 = Q4_b.shape[1], n_in = Q4_a.shape[2]
    cdef int iab, ia, ib, j
    cdef complex tmp
    cdef np.ndarray[np.float64_t,ndim=2] fish = np.zeros((n_out1,n_out2),dtype=np.float64)
    if sym:
        assert n_out1==n_out2, "Matrix must be square!"

    if sym:
        with nogil:
            for iab in xrange(n_out1*n_out2):
                ia = iab//n_out2
                ib = iab%n_out2
                if ia > ib: continue
                tmp = 0.
                for j in prange(n_in,schedule='static',num_threads=nthreads):
                    tmp += (Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j])+9*(Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j])-3*(Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j])
                fish[ia,ib] += creal(tmp)/24./48.
                if ia!=ib:
                    fish[ib,ia] += creal(tmp)/24./48.
    else:
        with nogil:
            for iab in xrange(n_out1*n_out2):
                ia = iab//n_out2
                ib = iab%n_out2
                tmp = 0.
                for j in prange(n_in,schedule='static',num_threads=nthreads):
                    tmp += (Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j])+9*(Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j])-3*(Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]+Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j])
                fish[ia,ib] += creal(tmp)/24./48.
    
    return fish  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] outer_product_tspec_ideal(complex[:,:,::1] Q4_a, complex[:,:,::1] Q4_b, int nthreads, bint sym):
    """Compute Fisher matrix between two large arrays as an outer product. This is parallelized across templates, not l,m.
    
    We assume a square matrix."""
    
    cdef int n_out1 = Q4_a.shape[1], n_out2 = Q4_b.shape[1], n_in = Q4_a.shape[2]
    cdef int iab, ia, ib, j
    cdef complex tmp
    cdef np.ndarray[np.float64_t,ndim=2] fish = np.zeros((n_out1,n_out2),dtype=np.float64)
        
    if sym:
        assert n_out1==n_out2, "Matrix must be square!"
        with nogil:
            for iab in prange(n_out1*n_out2,schedule='dynamic',num_threads=nthreads):
                ia = iab//n_out2
                ib = iab%n_out2
                if ia > ib: continue
                tmp = 0.
                for j in xrange(n_in):
                    tmp = tmp+Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j]+9*Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+9*Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]-3*Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]-3*Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j]
                fish[ia,ib] = creal(tmp)/24./48.
                if ia!=ib:
                    fish[ib,ia] = creal(tmp)/24./48.
    else:
        with nogil:
            for iab in prange(n_out1*n_out2,schedule='static',num_threads=nthreads):
                ia = iab//n_out2
                ib = iab%n_out2
                tmp = 0.
                for j in xrange(n_in):
                    tmp = tmp+Q4_a[0,ia,j].conjugate()*Q4_b[0,ib,j]+Q4_a[1,ia,j].conjugate()*Q4_b[1,ib,j]+9*Q4_a[2,ia,j].conjugate()*Q4_b[2,ib,j]+9*Q4_a[3,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[0,ia,j].conjugate()*Q4_b[3,ib,j]-3*Q4_a[1,ia,j].conjugate()*Q4_b[2,ib,j]-3*Q4_a[3,ia,j].conjugate()*Q4_b[0,ib,j]-3*Q4_a[2,ia,j].conjugate()*Q4_b[1,ib,j]
                fish[ia,ib] = creal(tmp)/24./48.
    return fish  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.float64_t,ndim=2] outer_product_bspec(complex[:,:,::1] Q3_a, complex[:,:,::1] Q3_b, int nthreads, bint sym):
    """Compute Fisher matrix between two large arrays as an outer product.
    
    If "sym" is specified, we assume a square symmetric matrix."""
    
    cdef int n_out1 = Q3_a.shape[1], n_out2 = Q3_b.shape[1], n_in = Q3_a.shape[2]
    cdef int iab, ia, ib, j
    cdef complex tmp
    cdef np.ndarray[np.float64_t,ndim=2] fish = np.zeros((n_out1,n_out2),dtype=np.float64)
    if sym:
        assert n_out1==n_out2, "Matrix must be square!"

    if sym:
        with nogil:
            for iab in prange(n_out1*n_out2,schedule='static',num_threads=nthreads):
                ia = iab//n_out2
                ib = iab%n_out2
                if ia > ib: continue
                tmp = 0.
                for j in xrange(n_in):
                    tmp = tmp+(Q3_a[0,ia,j].conjugate()*Q3_b[0,ib,j]+Q3_a[1,ia,j].conjugate()*Q3_b[1,ib,j])-(Q3_a[0,ia,j].conjugate()*Q3_b[1,ib,j]+Q3_a[1,ia,j].conjugate()*Q3_b[0,ib,j])
                fish[ia,ib] = creal(tmp)/24.
                if ia!=ib:
                    fish[ib,ia] = creal(tmp)/24.
    else:
        with nogil:
            for iab in prange(n_out1*n_out2,schedule='static',num_threads=nthreads):
                ia = iab//n_out2
                ib = iab%n_out2
                tmp = 0.
                for j in xrange(n_in):
                    tmp = tmp+(Q3_a[0,ia,j].conjugate()*Q3_b[0,ib,j]+Q3_a[1,ia,j].conjugate()*Q3_b[1,ib,j])-(Q3_a[0,ia,j].conjugate()*Q3_b[1,ib,j]+Q3_a[1,ia,j].conjugate()*Q3_b[0,ib,j])
                fish[ia,ib] = creal(tmp)/24.
    
    return fish  

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.complex128_t,ndim=2] apply_ideal_weight(complex[:,:,:,::1] a_map, double[:,:,::1] Ainv, double[:] m_weight, int nthreads):
    """Compute A^-1 x for a list of harmonic-space map x. We additionally add a weight of (1+l>0)."""
    cdef int npol = Ainv.shape[0], nm = a_map.shape[0], nt = a_map.shape[1], nl = a_map.shape[3]
    cdef int ipol, jpol, im, it, il
    cdef np.ndarray[np.complex128_t,ndim=3] out = np.zeros((nm,nt,npol*nl),dtype=np.complex128)

    # Code polarizations explicitly for speed
    if npol==1:
        with nogil:
            for im in xrange(nm):
                for it in prange(nt,schedule='static',num_threads=nthreads):
                    for il in xrange(nl):
                        out[im,it,il] = m_weight[il]*Ainv[0,0,il]*a_map[im,it,0,il]
    else:
        with nogil:
            for im in xrange(nm):
                for it in prange(nt,schedule='static',num_threads=nthreads):
                    for ipol in xrange(npol):
                        for il in xrange(nl):
                            out[im,it,ipol*nl+il] = m_weight[il]*(Ainv[ipol,0,il]*a_map[im,it,0,il]+Ainv[ipol,1,il]*a_map[im,it,1,il]+Ainv[ipol,2,il]*a_map[im,it,2,il])

    return out
    
## Ideal Fisher
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_dotdot(double[:,:,::1] alXs, double[:] tau_arr, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{dot,dot} template."""

    cdef int nl = lmax+1-lmin, nr = len(alXs[0,0]), npol = len(alXs[0]), nmu = len(w_mus)
    cdef int il, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, partial_sum
    cdef double[:] rfactor = np.zeros(nr,dtype=np.float64)
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaAA_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 24./dpow(4.*M_PI,2.)*dpow(384./25.,2.)/2.
    
    # Precompute r-dependent and l-dependent factors
    with nogil:
        for ir in xrange(nr):
            rfactor[ir] = weights[ir]*dpow(tau_arr[ir],4.)
        for il in xrange(nl):
            twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                partial_sum = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        partial_sum = partial_sum + twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*alXs[il+lmin,ipol,ir]*alXs[il+lmin,jpol,jr]
                zetaAA_l[il,ir,jr] = partial_sum

    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True,schedule='dynamic',num_threads=nthreads):
        for jr in xrange(ir+1):
            partial_sum = pref*rfactor[ir]*rfactor[jr]*_zeta_sum(zetaAA_l[:,ir,jr], legs, w_mus, nmu, nl)
            deriv_matrix[ir,jr] = partial_sum
            # Add upper triangle by symmetry
            if jr!=ir:
                deriv_matrix[jr,ir] = partial_sum

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_dotdel(double[:,:,::1] alXs, double[:,:,::1] blXs, double[:,:,::1] clXs,
                                   double[:] tau_arr, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] mu_arr, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{dot,del} template."""

    cdef int nl = lmax+1-lmin, nr = len(alXs[0,0]), npol = len(alXs[0]), nmu = len(w_mus)
    cdef int il, l, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, sins, partial_sum
    cdef double[:] rfactor = np.zeros(nr,dtype=np.float64)
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaAA_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaBB_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaCC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaAB_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaAC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaBC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 4./dpow(4.*M_PI,2.)*dpow(3456./325.,2.)/2.
    
    # Compute Wigner d^l_ss'(theta) functions
    cdef double[:,::1] wig10s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig01s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig11s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig1m1s = np.zeros((nmu,lmax),dtype=np.float64)
    
    for imu in prange(nmu, nogil=True, schedule='dynamic',num_threads=nthreads):
        sins = sqrt(1.-mu_arr[imu]*mu_arr[imu]) # since 0<theta<pi
        wig10s[imu,0] = 1./sqrt(2.)*sins
        wig11s[imu,0] = 0.5*(1.+mu_arr[imu])
        wig1m1s[imu,0] = 0.5*(1.-mu_arr[imu])
        for l in xrange(1,lmax):
            if l==1:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1])/_alpha(l+1,1,-1)
            else:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1]-_alpha(l,1,0)*wig10s[imu,l-2])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1]-_alpha(l,1,1)*wig11s[imu,l-2])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1]-_alpha(l,1,-1)*wig1m1s[imu,l-2])/_alpha(l+1,1,-1)
    
    # Precompute r-dependent and l-dependent factors
    with nogil:
        for ir in xrange(nr):
            rfactor[ir] = weights[ir]*dpow(tau_arr[ir],2.)
        for il in xrange(nl):
            twol_arr[il] = (2.*il+2*lmin+1.)
    
    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl,nogil=True,schedule='static',num_threads=nthreads):
        for ipol in xrange(npol):
            for jpol in xrange(npol):
                partial_sum = twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]
                for ir in xrange(nr):
                    for jr in xrange(nr):
                        zetaAA_l[il,ir,jr] += partial_sum*alXs[il+lmin,ipol,ir]*alXs[il+lmin,jpol,jr]
                        zetaBB_l[il,ir,jr] += partial_sum*blXs[il+lmin,ipol,ir]*blXs[il+lmin,jpol,jr]
                        zetaCC_l[il,ir,jr] += partial_sum*clXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
                        zetaAB_l[il,ir,jr] += partial_sum*alXs[il+lmin,ipol,ir]*blXs[il+lmin,jpol,jr]
                        zetaAC_l[il,ir,jr] += partial_sum*alXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
                        zetaBC_l[il,ir,jr] += partial_sum*blXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
    
    # Compute sum over l, mu for each r, r'
    for ir in prange(nr,nogil=True,schedule='dynamic',num_threads=nthreads):
        for jr in xrange(ir+1):
            partial_sum = pref*rfactor[ir]*rfactor[jr]*_zeta_sum_sym2(zetaAA_l[:,ir,jr], zetaBB_l[:,ir,jr], zetaCC_l[:,ir,jr], 
                                                                            zetaAB_l[:,ir,jr], zetaAC_l[:,ir,jr], zetaBC_l[:,ir,jr], 
                                                                            zetaAB_l[:,jr,ir], zetaAC_l[:,jr,ir], zetaBC_l[:,jr,ir], 
                                                                            legs, wig10s, wig11s, wig1m1s, w_mus, nmu, nl, lmin)
            deriv_matrix[ir,jr] = partial_sum
            # Add upper triangle by symmetry
            if jr!=ir:
                deriv_matrix[jr,ir] = partial_sum
    
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_deldel(double[:,:,::1] blXs, double[:,:,::1] clXs,
                                            double[:] weights, double[:,:,::1] inv_Cl_mat,
                                            double[:,::1] legs, double[:] mu_arr, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{del,del} template."""

    cdef int nl = lmax+1-lmin, nr = len(blXs[0,0]), npol = len(blXs[0]), nmu = len(w_mus)
    cdef int il, l, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, sins, partial_sum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaBB_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaCC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaBC_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 4./dpow(4.*M_PI,2.)*dpow(10368./2575.,2.)/2.
    
    # Compute Wigner d^l_ss'(theta) functions
    cdef double[:,::1] wig10s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig01s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig11s = np.zeros((nmu,lmax),dtype=np.float64)
    cdef double[:,::1] wig1m1s = np.zeros((nmu,lmax),dtype=np.float64)
    for imu in prange(nmu, nogil=True, schedule='static',num_threads=nthreads):
        sins = sqrt(1.-mu_arr[imu]*mu_arr[imu]) # since 0<theta<pi
        wig10s[imu,0] = 1./sqrt(2.)*sins
        wig11s[imu,0] = 0.5*(1.+mu_arr[imu])
        wig1m1s[imu,0] = 0.5*(1.-mu_arr[imu])
        for l in xrange(1,lmax):
            if l==1:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1])/_alpha(l+1,1,-1)
            else:
                wig10s[imu,l] =  ((2.*l+1.)*mu_arr[imu]*wig10s[imu,l-1]-_alpha(l,1,0)*wig10s[imu,l-2])/_alpha(l+1,1,0)
                wig11s[imu,l] =  ((2.*l+1.)*(mu_arr[imu]-1./(l*(l+1.)))*wig11s[imu,l-1]-_alpha(l,1,1)*wig11s[imu,l-2])/_alpha(l+1,1,1)
                wig1m1s[imu,l] = ((2.*l+1.)*(mu_arr[imu]+1./(l*(l+1.)))*wig1m1s[imu,l-1]-_alpha(l,1,-1)*wig1m1s[imu,l-2])/_alpha(l+1,1,-1)
    
    # Precompute l-dependent factors
    with nogil:
        for il in xrange(nl):
            twol_arr[il] = (2.*il+2*lmin+1.)
    
    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True, schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaBB_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*blXs[il+lmin,ipol,ir]*blXs[il+lmin,jpol,jr]
                        zetaCC_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*clXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]
                        zetaBC_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*blXs[il+lmin,ipol,ir]*clXs[il+lmin,jpol,jr]

    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True, schedule='dynamic',num_threads=nthreads):
        for jr in xrange(ir+1):
            partial_sum = pref*weights[ir]*weights[jr]*_zeta_sum_sym3(zetaBB_l[:,ir,jr], zetaCC_l[:,ir,jr], 
                                                                                zetaBC_l[:,ir,jr], zetaBC_l[:,jr,ir], 
                                                                                legs, wig10s, wig11s, wig1m1s, w_mus, nmu, nl, lmin)
            deriv_matrix[ir,jr] = partial_sum
            # Add upper triangle by symmetry
            if jr!=ir:
                deriv_matrix[jr,ir] = partial_sum
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_con(double[:,:,::1] rlXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{con} template."""

    cdef int nl = lmax+1-lmin, nr = len(rlXs[0,0]), npol = len(rlXs[0]), nmu = len(w_mus)
    cdef int il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum, partial_sum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaRR_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 24./dpow(4.*M_PI,2.)*dpow(9./25.,2.)/2.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) a^Y S^-1 a^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                partial_sum = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        partial_sum = partial_sum + twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*rlXs[il+lmin,ipol,ir]*rlXs[il+lmin,jpol,jr]
                zetaRR_l[il,ir,jr] += partial_sum

    # Compute sum over l, mu for each r, r'
    for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
        ir = ijr//nr
        jr = ijr%nr
        deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum(zetaRR_l[:,ir,jr], legs, w_mus, nmu, nl)

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_loc(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the fNL^{loc} template."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(w_mus)
    cdef int il, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPP_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1)*9./25.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPP_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*plXs[il+lmin,jpol,jr]
                        zetaPQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]
                        zetaQQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]

    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True,schedule='static',num_threads=nthreads):
        for jr in xrange(ir,nr):
            deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_symB(zetaPP_l[:,ir,jr], zetaPQ_l[:,ir,jr], zetaPQ_l[:,jr,ir], zetaQQ_l[:,ir,jr], legs, w_mus, nmu, nl)
            if ir!=jr:
                deriv_matrix[jr,ir] = deriv_matrix[ir,jr]

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_binned(double[:,:,::1] flXs, double[:] weights, double[:,:,::1] inv_Cl_mat, double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the fNL^{binned} template, summed over the bins."""

    cdef int nl = lmax+1-lmin, nr = len(flXs[0,0]), npol = len(flXs[0]), nmu = len(w_mus)
    cdef int il, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaFF_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1)*9./25.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaFF_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*flXs[il+lmin,ipol,ir]*flXs[il+lmin,jpol,jr]
    
    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True,schedule='static',num_threads=nthreads):
        for jr in xrange(ir,nr):
            deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_full_symB(zetaFF_l[:,ir,jr], legs, w_mus, nmu, nl)
            if ir!=jr:
                deriv_matrix[jr,ir] = deriv_matrix[ir,jr]

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_feat_res_2d(double[:,:,::1] flXsm3, double[:,:,::1] flXsm2, double[:,:,::1] flXsm1, double[:,:,::1] flXsp0, double[:,:,::1] flXsp1, double[:] weights,
                                                double[:] VQ1, double[:] VQ2, double[:] VQ3, double[:] VQ4,
                                                double[:,:,::1] inv_Cl_mat, double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact ideal Fisher matrix for the fNL^feat-res template (genuinely-convergent, shifted-
    Mellin representation). This is the ONLY ideal-Fisher implementation for this template -- r and u are
    always collapsed into a single combined pair index; there is no separate 'r-only, u pre-summed
    internally' variant. r,r' and u,u' collapse into pair_i, pair_j (no internal u,u' sum), so the output
    is a pure function of (pair_i, pair_j); VQ1..VQ4, weights are evaluated per-pair (using that pair's own
    u, r respectively). Summing the whole matrix gives the total ideal Fisher information; summing only
    the (pair_i, pair_j) blocks sharing a given r (or u) gives that variable's own (N_r, N_r) (or (N_u,
    N_u)) marginal Fisher matrix, e.g. for the optimize_radial_sampling_1d r-optimization workflow. 4 leg-
    power multisets Q1={-3,-3,1}, Q2={-3,-2,0}, Q3={-3,-1,-1}, Q4={-2,-2,-1}; the 4x4 group-pair (m_ab)
    structure was derived by brute-force enumeration of all 6 leg-to-leg bijections per multiset pair
    (divided by 2 for the dmu/2 normalization), checked to exactly reproduce the old, already-validated 3x3
    m_ab formulas for the divergent 2-leg-power representation before being applied here, and independently
    verified against a from-scratch brute-force Python computation (rel diff ~2e-14)."""

    cdef int nl = lmax+1-lmin, npairs = flXsm3.shape[2], npol = flXsm3.shape[1], nmu = len(w_mus)
    cdef int il, ip, jp, ipol, jpol, tid, imu
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((npairs,npairs),dtype=np.float64)
    cdef double[:,:,::1] zeta = np.zeros((nthreads,25,nl),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1)/36.
    cdef double acc, tmp
    cdef double sm3m3, sm3m2, sm3m1, sm3p0, sm3p1, sm2m3, sm2m2, sm2m1, sm2p0, sm2p1, sm1m3, sm1m2, sm1m1, sm1p0, sm1p1, sp0m3, sp0m2, sp0m1, sp0p0, sp0p1, sp1m3, sp1m2, sp1m1, sp1p0, sp1p1
    cdef double m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44

    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    for ip in prange(npairs, nogil=True, schedule='static', num_threads=nthreads):
        tid = threadid()
        for jp in xrange(ip,npairs):

            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm3[il+lmin,ipol,ip]*flXsm3[il+lmin,jpol,jp]
                zeta[tid,0,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm3[il+lmin,ipol,ip]*flXsm2[il+lmin,jpol,jp]
                zeta[tid,1,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm3[il+lmin,ipol,ip]*flXsm1[il+lmin,jpol,jp]
                zeta[tid,2,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm3[il+lmin,ipol,ip]*flXsp0[il+lmin,jpol,jp]
                zeta[tid,3,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm3[il+lmin,ipol,ip]*flXsp1[il+lmin,jpol,jp]
                zeta[tid,4,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm2[il+lmin,ipol,ip]*flXsm3[il+lmin,jpol,jp]
                zeta[tid,5,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm2[il+lmin,ipol,ip]*flXsm2[il+lmin,jpol,jp]
                zeta[tid,6,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm2[il+lmin,ipol,ip]*flXsm1[il+lmin,jpol,jp]
                zeta[tid,7,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm2[il+lmin,ipol,ip]*flXsp0[il+lmin,jpol,jp]
                zeta[tid,8,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm2[il+lmin,ipol,ip]*flXsp1[il+lmin,jpol,jp]
                zeta[tid,9,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm1[il+lmin,ipol,ip]*flXsm3[il+lmin,jpol,jp]
                zeta[tid,10,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm1[il+lmin,ipol,ip]*flXsm2[il+lmin,jpol,jp]
                zeta[tid,11,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm1[il+lmin,ipol,ip]*flXsm1[il+lmin,jpol,jp]
                zeta[tid,12,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm1[il+lmin,ipol,ip]*flXsp0[il+lmin,jpol,jp]
                zeta[tid,13,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsm1[il+lmin,ipol,ip]*flXsp1[il+lmin,jpol,jp]
                zeta[tid,14,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp0[il+lmin,ipol,ip]*flXsm3[il+lmin,jpol,jp]
                zeta[tid,15,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp0[il+lmin,ipol,ip]*flXsm2[il+lmin,jpol,jp]
                zeta[tid,16,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp0[il+lmin,ipol,ip]*flXsm1[il+lmin,jpol,jp]
                zeta[tid,17,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp0[il+lmin,ipol,ip]*flXsp0[il+lmin,jpol,jp]
                zeta[tid,18,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp0[il+lmin,ipol,ip]*flXsp1[il+lmin,jpol,jp]
                zeta[tid,19,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp1[il+lmin,ipol,ip]*flXsm3[il+lmin,jpol,jp]
                zeta[tid,20,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp1[il+lmin,ipol,ip]*flXsm2[il+lmin,jpol,jp]
                zeta[tid,21,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp1[il+lmin,ipol,ip]*flXsm1[il+lmin,jpol,jp]
                zeta[tid,22,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp1[il+lmin,ipol,ip]*flXsp0[il+lmin,jpol,jp]
                zeta[tid,23,il] = twol_arr[il]*tmp
            for il in xrange(nl):
                tmp = 0.
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        tmp = tmp + inv_Cl_mat[ipol,jpol,il+lmin]*flXsp1[il+lmin,ipol,ip]*flXsp1[il+lmin,jpol,jp]
                zeta[tid,24,il] = twol_arr[il]*tmp

            m11 = 0.
            m12 = 0.
            m13 = 0.
            m14 = 0.
            m21 = 0.
            m22 = 0.
            m23 = 0.
            m24 = 0.
            m31 = 0.
            m32 = 0.
            m33 = 0.
            m34 = 0.
            m41 = 0.
            m42 = 0.
            m43 = 0.
            m44 = 0.

            for imu in xrange(nmu):
                sm3m3 = 0.
                sm3m2 = 0.
                sm3m1 = 0.
                sm3p0 = 0.
                sm3p1 = 0.
                sm2m3 = 0.
                sm2m2 = 0.
                sm2m1 = 0.
                sm2p0 = 0.
                sm2p1 = 0.
                sm1m3 = 0.
                sm1m2 = 0.
                sm1m1 = 0.
                sm1p0 = 0.
                sm1p1 = 0.
                sp0m3 = 0.
                sp0m2 = 0.
                sp0m1 = 0.
                sp0p0 = 0.
                sp0p1 = 0.
                sp1m3 = 0.
                sp1m2 = 0.
                sp1m1 = 0.
                sp1p0 = 0.
                sp1p1 = 0.
                for il in xrange(nl):
                    sm3m3 = sm3m3 + zeta[tid,0,il]*legs[imu,il]
                    sm3m2 = sm3m2 + zeta[tid,1,il]*legs[imu,il]
                    sm3m1 = sm3m1 + zeta[tid,2,il]*legs[imu,il]
                    sm3p0 = sm3p0 + zeta[tid,3,il]*legs[imu,il]
                    sm3p1 = sm3p1 + zeta[tid,4,il]*legs[imu,il]
                    sm2m3 = sm2m3 + zeta[tid,5,il]*legs[imu,il]
                    sm2m2 = sm2m2 + zeta[tid,6,il]*legs[imu,il]
                    sm2m1 = sm2m1 + zeta[tid,7,il]*legs[imu,il]
                    sm2p0 = sm2p0 + zeta[tid,8,il]*legs[imu,il]
                    sm2p1 = sm2p1 + zeta[tid,9,il]*legs[imu,il]
                    sm1m3 = sm1m3 + zeta[tid,10,il]*legs[imu,il]
                    sm1m2 = sm1m2 + zeta[tid,11,il]*legs[imu,il]
                    sm1m1 = sm1m1 + zeta[tid,12,il]*legs[imu,il]
                    sm1p0 = sm1p0 + zeta[tid,13,il]*legs[imu,il]
                    sm1p1 = sm1p1 + zeta[tid,14,il]*legs[imu,il]
                    sp0m3 = sp0m3 + zeta[tid,15,il]*legs[imu,il]
                    sp0m2 = sp0m2 + zeta[tid,16,il]*legs[imu,il]
                    sp0m1 = sp0m1 + zeta[tid,17,il]*legs[imu,il]
                    sp0p0 = sp0p0 + zeta[tid,18,il]*legs[imu,il]
                    sp0p1 = sp0p1 + zeta[tid,19,il]*legs[imu,il]
                    sp1m3 = sp1m3 + zeta[tid,20,il]*legs[imu,il]
                    sp1m2 = sp1m2 + zeta[tid,21,il]*legs[imu,il]
                    sp1m1 = sp1m1 + zeta[tid,22,il]*legs[imu,il]
                    sp1p0 = sp1p0 + zeta[tid,23,il]*legs[imu,il]
                    sp1p1 = sp1p1 + zeta[tid,24,il]*legs[imu,il]

                m11 = m11 + w_mus[imu]*(1.0*sm3m3*sm3m3*sp1p1 + 2.0*sm3m3*sm3p1*sp1m3)
                m12 = m12 + w_mus[imu]*(1.0*sm3m3*sm3m2*sp1p0 + 1.0*sm3m3*sm3p0*sp1m2 + 1.0*sm3m2*sm3p0*sp1m3)
                m13 = m13 + w_mus[imu]*(2.0*sm3m3*sm3m1*sp1m1 + 1.0*sm3m1*sm3m1*sp1m3)
                m14 = m14 + w_mus[imu]*(1.0*sm3m2*sm3m2*sp1m1 + 2.0*sm3m2*sm3m1*sp1m2)
                m21 = m21 + w_mus[imu]*(1.0*sm3m3*sm2m3*sp0p1 + 1.0*sm3m3*sm2p1*sp0m3 + 1.0*sm3p1*sm2m3*sp0m3)
                m22 = m22 + w_mus[imu]*(0.5*sm3m3*sm2m2*sp0p0 + 0.5*sm3m3*sm2p0*sp0m2 + 0.5*sm3m2*sm2m3*sp0p0 + 0.5*sm3m2*sm2p0*sp0m3 + 0.5*sm3p0*sm2m3*sp0m2 + 0.5*sm3p0*sm2m2*sp0m3)
                m23 = m23 + w_mus[imu]*(1.0*sm3m3*sm2m1*sp0m1 + 1.0*sm3m1*sm2m3*sp0m1 + 1.0*sm3m1*sm2m1*sp0m3)
                m24 = m24 + w_mus[imu]*(1.0*sm3m2*sm2m2*sp0m1 + 1.0*sm3m2*sm2m1*sp0m2 + 1.0*sm3m1*sm2m2*sp0m2)
                m31 = m31 + w_mus[imu]*(2.0*sm3m3*sm1m3*sm1p1 + 1.0*sm3p1*sm1m3*sm1m3)
                m32 = m32 + w_mus[imu]*(1.0*sm3m3*sm1m2*sm1p0 + 1.0*sm3m2*sm1m3*sm1p0 + 1.0*sm3p0*sm1m3*sm1m2)
                m33 = m33 + w_mus[imu]*(1.0*sm3m3*sm1m1*sm1m1 + 2.0*sm3m1*sm1m3*sm1m1)
                m34 = m34 + w_mus[imu]*(2.0*sm3m2*sm1m2*sm1m1 + 1.0*sm3m1*sm1m2*sm1m2)
                m41 = m41 + w_mus[imu]*(1.0*sm2m3*sm2m3*sm1p1 + 2.0*sm2m3*sm2p1*sm1m3)
                m42 = m42 + w_mus[imu]*(1.0*sm2m3*sm2m2*sm1p0 + 1.0*sm2m3*sm2p0*sm1m2 + 1.0*sm2m2*sm2p0*sm1m3)
                m43 = m43 + w_mus[imu]*(2.0*sm2m3*sm2m1*sm1m1 + 1.0*sm2m1*sm2m1*sm1m3)
                m44 = m44 + w_mus[imu]*(1.0*sm2m2*sm2m2*sm1m1 + 2.0*sm2m2*sm2m1*sm1m2)

            acc = (VQ1[ip]*VQ1[jp]*m11
                 + VQ1[ip]*VQ2[jp]*m12
                 + VQ1[ip]*VQ3[jp]*m13
                 + VQ1[ip]*VQ4[jp]*m14
                 + VQ2[ip]*VQ1[jp]*m21
                 + VQ2[ip]*VQ2[jp]*m22
                 + VQ2[ip]*VQ3[jp]*m23
                 + VQ2[ip]*VQ4[jp]*m24
                 + VQ3[ip]*VQ1[jp]*m31
                 + VQ3[ip]*VQ2[jp]*m32
                 + VQ3[ip]*VQ3[jp]*m33
                 + VQ3[ip]*VQ4[jp]*m34
                 + VQ4[ip]*VQ1[jp]*m41
                 + VQ4[ip]*VQ2[jp]*m42
                 + VQ4[ip]*VQ3[jp]*m43
                 + VQ4[ip]*VQ4[jp]*m44)

            deriv_matrix[ip,jp] = pref*weights[ip]*weights[jp]*acc
            if ip!=jp:
                deriv_matrix[jp,ip] = deriv_matrix[ip,jp]

    return deriv_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_feat_res_2d_complex(double[:,:,:,::1] legre, double[:,:,:,::1] legim,
                        double[:] weights, double[:,::1] Vgre, double[:,::1] Vgim,
                        int[::1] term_group, int[:,::1] term_slots, int nslot,
                        double[:,:,::1] inv_Cl_mat, double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Complex ideal-Fisher derivative for fNL-feat-res (compressed exp-sum rep, sm3 & sm1).
    legre/legim: (nslot=5, lmax+1, npol, npairs) real/imag parts of the complex legs k^{kpow}e^{-k u}
      at complex nodes; slot 0..4 = kpow -3,-2,-1,0,+1. weights: per-pair r-quadrature weight.
    Vgre/Vgim: (ngroup, npairs) complex per-pair group weights = pref0*kappa^{iw}*W_node*gamma_g/n_ord_g.
    term_group[nterm], term_slots[nterm,3]: each distinct ordered leg-triple (slot indices) of every group,
      tagged with its group index (so the 1/n_ord is already folded into Vg).
    The real reduced bispectrum is b=2Re[Z], so F = 2Re<Z,Z> + 2<Z,Z_bar> (symmetric bilinear form):
      <Z,Z>:   s_ab = sum_l (2l+1) invCl leg_a[ip] leg_b[jp] P_l(mu);      weight Vg[gA,ip] Vg[gB,jp].
      <Z,Zbar>: leg_b[jp] -> conj, and Vg[gB,jp] -> conj.
    deriv[ip,jp] = (1/48pi) w[ip] w[jp] (2 Re m_ZZ + 2 Re m_ZZbar); summing gives the ideal Fisher.
    Verified vs brute force (FEAT_NOTES Session 9/9b)."""

    cdef int nl = lmax+1-lmin, npairs = legre.shape[3], npol = legre.shape[2], nmu = len(w_mus)
    cdef int nterm = term_group.shape[0], nsp = nslot*nslot, nrow = 4*nslot*nslot
    cdef int il, ip, jp, ipol, jpol, tid, imu, a, b, cp, tA, tB, gA, gB, i0,i1,i2,j0,j1,j2, c00,c11,c22
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((npairs,npairs),dtype=np.float64)
    # per-thread scratch: compact zeta buffer (rows 0..nrow-1 = ZZr,ZZi,ZBr,ZBi each nsp slot-pairs) x nl,
    # and its BLAS projection S = zeta @ legs^T (nrow x nmu). Allocated at the nslot=5 max (4*25=100 rows).
    cdef double[:,:,::1] zbuf = np.zeros((nthreads,4*25,nl),dtype=np.float64)
    cdef double[:,:,::1] Sbuf = np.zeros((nthreads,4*25,nmu),dtype=np.float64)
    cdef double prefc = 1.0/(48.0*M_PI)
    cdef double ic, ar, ai, br, bi, zzr, zzi, zbr, zbi, tl
    cdef double p12r,p12i,Pr,Pi, Pbr,Pbi, VAr,VAi,VBr,VBi, vr,vi, vbr_,vbi_
    cdef double accZZr, accZBr, mZZr, mZBr
    cdef double s00r,s00i,s11r,s11i,s22r,s22i, b00r,b00i,b11r,b11i,b22r,b22i
    # dgemm (Fortran/col-major) computes S[row,mu]=sum_l zbuf[row,l] legs[mu,l] as one matmul per (ip,jp):
    # row-major legs(nmu x nl)=col-major(nl x nmu); op(A)='T' -> (nmu x nl); op(B=zbuf)='N' -> (nl x nrow).
    cdef char tT = 84, tN = 78   # 'T', 'N'
    cdef int mm = nmu, nn = nrow, kk = nl, ld = nl, ldc = nmu
    cdef double alpha = 1.0, beta = 0.0
    # BLAS dgemm mu-projection is OFF by default. scipy's bundled OpenBLAS is compiled MAX_THREADS=64 and
    # corrupts its thread metadata when called from >64 prange threads -> segfault on >64-core nodes. The
    # inline projection has no such limit and uses all cores. Opt back in (ONLY safe at <=64 threads, where
    # it is ~1.7x faster) via FEAT_USE_BLAS_DGEMM=1.
    cdef int use_blas = 1 if os.environ.get('FEAT_USE_BLAS_DGEMM','0')=='1' else 0

    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    for ip in prange(npairs, nogil=True, schedule='static', num_threads=nthreads):
        tid = threadid()
        for jp in xrange(ip,npairs):
            # --- zeta into the compact buffer (ZZ and ZZbar, re/im), summed over polarizations ---
            for a in xrange(nslot):
                for b in xrange(nslot):
                    cp = a*nslot+b
                    for il in xrange(nl):
                        zzr=0.; zzi=0.; zbr=0.; zbi=0.
                        for ipol in xrange(npol):
                            for jpol in xrange(npol):
                                ic = inv_Cl_mat[ipol,jpol,il+lmin]
                                ar = legre[a,il+lmin,ipol,ip]; ai = legim[a,il+lmin,ipol,ip]
                                br = legre[b,il+lmin,jpol,jp]; bi = legim[b,il+lmin,jpol,jp]
                                zzr = zzr + ic*(ar*br - ai*bi); zzi = zzi + ic*(ar*bi + ai*br)   # leg_a*leg_b
                                zbr = zbr + ic*(ar*br + ai*bi); zbi = zbi + ic*(ai*br - ar*bi)   # leg_a*conj(leg_b)
                        tl = twol_arr[il]
                        zbuf[tid,cp,il]=tl*zzr; zbuf[tid,nsp+cp,il]=tl*zzi
                        zbuf[tid,2*nsp+cp,il]=tl*zbr; zbuf[tid,3*nsp+cp,il]=tl*zbi
            # --- mu-projection Sbuf[row,mu] = sum_l zbuf[row,l] legs[mu,l] ---
            # Default: inline loop (no OpenBLAS -> no >64-thread crash, scales to all cores). Opt-in dgemm
            # (FEAT_USE_BLAS_DGEMM=1) only for large nl AND <=64 threads, where the matmul is the win.
            if nl > 96 and use_blas:
                dgemm(&tT,&tN,&mm,&nn,&kk,&alpha,&legs[0,0],&ld,&zbuf[tid,0,0],&ld,&beta,&Sbuf[tid,0,0],&ldc)
            else:
                for imu in xrange(nmu):
                    for cp in xrange(nrow):
                        zzr=0.
                        for il in xrange(nl):
                            zzr = zzr + zbuf[tid,cp,il]*legs[imu,il]
                        Sbuf[tid,cp,imu]=zzr
            # --- mu integral + term-pair combination (read projected s-values from Sbuf) ---
            mZZr=0.; mZBr=0.
            for imu in xrange(nmu):
                accZZr=0.; accZBr=0.
                for tA in xrange(nterm):
                    gA = term_group[tA]; i0=term_slots[tA,0]; i1=term_slots[tA,1]; i2=term_slots[tA,2]
                    VAr = Vgre[gA,ip]; VAi = Vgim[gA,ip]
                    for tB in xrange(nterm):
                        gB = term_group[tB]; j0=term_slots[tB,0]; j1=term_slots[tB,1]; j2=term_slots[tB,2]
                        VBr = Vgre[gB,jp]; VBi = Vgim[gB,jp]
                        c00=i0*nslot+j0; c11=i1*nslot+j1; c22=i2*nslot+j2
                        # ZZ triple product of projected s = Sbuf[cp] + i Sbuf[nsp+cp]
                        s00r=Sbuf[tid,c00,imu]; s00i=Sbuf[tid,nsp+c00,imu]
                        s11r=Sbuf[tid,c11,imu]; s11i=Sbuf[tid,nsp+c11,imu]
                        s22r=Sbuf[tid,c22,imu]; s22i=Sbuf[tid,nsp+c22,imu]
                        p12r = s00r*s11r - s00i*s11i; p12i = s00r*s11i + s00i*s11r
                        Pr = p12r*s22r - p12i*s22i; Pi = p12r*s22i + p12i*s22r
                        vr = VAr*VBr - VAi*VBi; vi = VAr*VBi + VAi*VBr        # VA*VB
                        accZZr = accZZr + vr*Pr - vi*Pi
                        # ZZbar: s = Sbuf[2*nsp+cp] + i Sbuf[3*nsp+cp]; weight VA*conj(VB)
                        b00r=Sbuf[tid,2*nsp+c00,imu]; b00i=Sbuf[tid,3*nsp+c00,imu]
                        b11r=Sbuf[tid,2*nsp+c11,imu]; b11i=Sbuf[tid,3*nsp+c11,imu]
                        b22r=Sbuf[tid,2*nsp+c22,imu]; b22i=Sbuf[tid,3*nsp+c22,imu]
                        p12r = b00r*b11r - b00i*b11i; p12i = b00r*b11i + b00i*b11r
                        Pbr = p12r*b22r - p12i*b22i; Pbi = p12r*b22i + p12i*b22r
                        vbr_ = VAr*VBr + VAi*VBi; vbi_ = VAi*VBr - VAr*VBi    # VA*conj(VB)
                        accZBr = accZBr + vbr_*Pbr - vbi_*Pbi
                mZZr = mZZr + w_mus[imu]*accZZr
                mZBr = mZBr + w_mus[imu]*accZBr
            deriv_matrix[ip,jp] = prefc*weights[ip]*weights[jp]*(2.*mZZr + 2.*mZBr)
            if ip!=jp:
                deriv_matrix[jp,ip] = deriv_matrix[ip,jp]

    return deriv_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1] fisher_deriv_fNL_feat_res_2d_complex_batch(double[:,:,:,::1] legre, double[:,:,:,::1] legim,
                        double[:] weights, double[:,:,::1] Vgre, double[:,:,::1] Vgim,
                        int[::1] term_group, int[:,::1] term_slots, int nslot,
                        double[:,:,::1] inv_Cl_mat, double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """BATCHED ideal-Fisher SCALAR for fNL-feat-res over many (omega,kappa) at once.

    Identical to fisher_deriv_fNL_feat_res_2d_complex except: (a) the per-pair leg work -- the zeta build
    and the mu-projection Sbuf -- is PARAMETER-INDEPENDENT and computed ONCE per (ip,jp), then (b) the cheap
    term-pair combination (the ONLY (omega,kappa)-dependent step, via the per-param weights Vg) is looped over
    all nparam and reduced directly into a per-param SCALAR Fisher F[p] = sum over the full (ip,jp) matrix.
    This shares the expensive projection across the whole batch and avoids materialising nparam N_pairs^2
    matrices. Term structure (term_group/term_slots/nslot) is common to all params (the sm3/sm1 recipe KEYS
    are omega-independent; only the coefficients folded into Vg change).

    Vgre/Vgim: (nparam, ngroup, npairs). Returns F: (nparam,). For a single param, F[0] equals
    np.sum(fisher_deriv_fNL_feat_res_2d_complex(... that param's Vg ...)) to machine precision."""

    cdef int nl = lmax+1-lmin, npairs = legre.shape[3], npol = legre.shape[2], nmu = len(w_mus)
    cdef int nterm = term_group.shape[0], nsp = nslot*nslot, nrow = 4*nslot*nslot
    cdef int nparam = Vgre.shape[0], ng = Vgre.shape[1], ng2 = ng*ng
    cdef int il, ip, jp, ipol, jpol, tid, imu, a, b, cp, tA, tB, gA, gB, i0,i1,i2,j0,j1,j2, c00,c11,c22, pidx, gg
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zbuf = np.zeros((nthreads,4*25,nl),dtype=np.float64)
    cdef double[:,:,::1] Sbuf = np.zeros((nthreads,4*25,nmu),dtype=np.float64)
    # Per-thread group-level, mu-INTEGRATED form factors U[gA,gB] (ZZ and ZZbar), PARAM-INDEPENDENT: the sole
    # (omega,kappa) dependence is the per-group weight Vg, so we integrate the term/mu structure into
    #   U_ZZ[gA,gB](ip,jp) = sum_mu w_mu sum_{tA in gA, tB in gB} s_i0 s_i1 s_i2      (ZZ, complex)
    # ONCE per (ip,jp), then contract F[p] += w_ip w_jp sum_{gA,gB} 2Re[VA VB U_ZZ] + 2Re[VA conj(VB) U_ZB]
    # for each param -- only ng^2 (<=9) cheap ops per param, so the batch cost is ~independent of nparam.
    cdef double[:,::1] Uzzr = np.zeros((nthreads,ng2),dtype=np.float64)
    cdef double[:,::1] Uzzi = np.zeros((nthreads,ng2),dtype=np.float64)
    cdef double[:,::1] Uzbr = np.zeros((nthreads,ng2),dtype=np.float64)
    cdef double[:,::1] Uzbi = np.zeros((nthreads,ng2),dtype=np.float64)
    # per-thread, per-param accumulator of the scalar Fisher (reduced at the end -> no race in prange).
    # Kahan (compensated) summation over the long per-thread pair loop keeps the reduction at the float64
    # floor, so the batched scalar matches the single-param np.sum(deriv_matrix) as closely as summation allows.
    cdef double[:,::1] Fbuf = np.zeros((nthreads,nparam),dtype=np.float64)
    cdef double[:,::1] Kbuf = np.zeros((nthreads,nparam),dtype=np.float64)   # Kahan compensation
    cdef np.ndarray[np.float64_t,ndim=1] Fout = np.zeros(nparam,dtype=np.float64)
    cdef double prefc = 1.0/(48.0*M_PI)
    cdef double ic, ar, ai, br, bi, zzr, zzi, zbr, zbi, tl, wpair, wmu
    cdef double p12r,p12i,Pr,Pi, Pbr,Pbi, VAr,VAi,VBr,VBi, vr,vi, vbr_,vbi_, uzr,uzi,ubr,ubi
    cdef double accr, val, kterm, kt, ky
    cdef double s00r,s00i,s11r,s11i,s22r,s22i, b00r,b00i,b11r,b11i,b22r,b22i
    cdef char tT = 84, tN = 78
    cdef int mm = nmu, nn = nrow, kk = nl, ld = nl, ldc = nmu
    cdef double alpha = 1.0, beta = 0.0
    cdef int use_blas = 1 if os.environ.get('FEAT_USE_BLAS_DGEMM','0')=='1' else 0

    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    for ip in prange(npairs, nogil=True, schedule='static', num_threads=nthreads):
        tid = threadid()
        for jp in xrange(ip,npairs):
            # --- zeta into the compact buffer (PARAM-INDEPENDENT) ---
            for a in xrange(nslot):
                for b in xrange(nslot):
                    cp = a*nslot+b
                    for il in xrange(nl):
                        zzr=0.; zzi=0.; zbr=0.; zbi=0.
                        for ipol in xrange(npol):
                            for jpol in xrange(npol):
                                ic = inv_Cl_mat[ipol,jpol,il+lmin]
                                ar = legre[a,il+lmin,ipol,ip]; ai = legim[a,il+lmin,ipol,ip]
                                br = legre[b,il+lmin,jpol,jp]; bi = legim[b,il+lmin,jpol,jp]
                                zzr = zzr + ic*(ar*br - ai*bi); zzi = zzi + ic*(ar*bi + ai*br)
                                zbr = zbr + ic*(ar*br + ai*bi); zbi = zbi + ic*(ai*br - ar*bi)
                        tl = twol_arr[il]
                        zbuf[tid,cp,il]=tl*zzr; zbuf[tid,nsp+cp,il]=tl*zzi
                        zbuf[tid,2*nsp+cp,il]=tl*zbr; zbuf[tid,3*nsp+cp,il]=tl*zbi
            # --- mu-projection Sbuf (PARAM-INDEPENDENT) ---
            if nl > 96 and use_blas:
                dgemm(&tT,&tN,&mm,&nn,&kk,&alpha,&legs[0,0],&ld,&zbuf[tid,0,0],&ld,&beta,&Sbuf[tid,0,0],&ldc)
            else:
                for imu in xrange(nmu):
                    for cp in xrange(nrow):
                        zzr=0.
                        for il in xrange(nl):
                            zzr = zzr + zbuf[tid,cp,il]*legs[imu,il]
                        Sbuf[tid,cp,imu]=zzr
            # --- PARAM-INDEPENDENT group form factors U[gA,gB] (mu-integrated), built ONCE per pair ---
            for gg in xrange(ng2):
                Uzzr[tid,gg]=0.; Uzzi[tid,gg]=0.; Uzbr[tid,gg]=0.; Uzbi[tid,gg]=0.
            for imu in xrange(nmu):
                wmu = w_mus[imu]
                for tA in xrange(nterm):
                    gA = term_group[tA]; i0=term_slots[tA,0]; i1=term_slots[tA,1]; i2=term_slots[tA,2]
                    for tB in xrange(nterm):
                        gB = term_group[tB]; j0=term_slots[tB,0]; j1=term_slots[tB,1]; j2=term_slots[tB,2]
                        gg = gA*ng+gB
                        c00=i0*nslot+j0; c11=i1*nslot+j1; c22=i2*nslot+j2
                        # ZZ triple product of projected s = Sbuf[cp] + i Sbuf[nsp+cp]
                        s00r=Sbuf[tid,c00,imu]; s00i=Sbuf[tid,nsp+c00,imu]
                        s11r=Sbuf[tid,c11,imu]; s11i=Sbuf[tid,nsp+c11,imu]
                        s22r=Sbuf[tid,c22,imu]; s22i=Sbuf[tid,nsp+c22,imu]
                        p12r = s00r*s11r - s00i*s11i; p12i = s00r*s11i + s00i*s11r
                        Pr = p12r*s22r - p12i*s22i; Pi = p12r*s22i + p12i*s22r
                        Uzzr[tid,gg] = Uzzr[tid,gg] + wmu*Pr; Uzzi[tid,gg] = Uzzi[tid,gg] + wmu*Pi
                        # ZZbar triple product: s = Sbuf[2*nsp+cp] + i Sbuf[3*nsp+cp]
                        b00r=Sbuf[tid,2*nsp+c00,imu]; b00i=Sbuf[tid,3*nsp+c00,imu]
                        b11r=Sbuf[tid,2*nsp+c11,imu]; b11i=Sbuf[tid,3*nsp+c11,imu]
                        b22r=Sbuf[tid,2*nsp+c22,imu]; b22i=Sbuf[tid,3*nsp+c22,imu]
                        p12r = b00r*b11r - b00i*b11i; p12i = b00r*b11i + b00i*b11r
                        Pbr = p12r*b22r - p12i*b22i; Pbi = p12r*b22i + p12i*b22r
                        Uzbr[tid,gg] = Uzbr[tid,gg] + wmu*Pbr; Uzbi[tid,gg] = Uzbi[tid,gg] + wmu*Pbi
            # --- (omega,kappa)-DEPENDENT contraction with Vg (cheap: ng^2 per param) ---
            wpair = prefc*weights[ip]*weights[jp]
            for pidx in xrange(nparam):
                accr = 0.
                for gA in xrange(ng):
                    VAr = Vgre[pidx,gA,ip]; VAi = Vgim[pidx,gA,ip]
                    for gB in xrange(ng):
                        gg = gA*ng+gB
                        VBr = Vgre[pidx,gB,jp]; VBi = Vgim[pidx,gB,jp]
                        uzr = Uzzr[tid,gg]; uzi = Uzzi[tid,gg]
                        ubr = Uzbr[tid,gg]; ubi = Uzbi[tid,gg]
                        vr = VAr*VBr - VAi*VBi; vi = VAr*VBi + VAi*VBr        # VA*VB
                        accr = accr + (vr*uzr - vi*uzi)                       # Re[VA VB U_ZZ]
                        vbr_ = VAr*VBr + VAi*VBi; vbi_ = VAi*VBr - VAr*VBi    # VA*conj(VB)
                        accr = accr + (vbr_*ubr - vbi_*ubi)                   # Re[VA conj(VB) U_ZB]
                val = wpair*2.*accr
                # accumulate the FULL-matrix sum: diagonal once, off-diagonal twice (symmetric)
                if ip!=jp:
                    kterm = 2.*val
                else:
                    kterm = val
                # Kahan-compensated add of kterm into Fbuf[tid,pidx]
                ky = kterm - Kbuf[tid,pidx]
                kt = Fbuf[tid,pidx] + ky
                Kbuf[tid,pidx] = (kt - Fbuf[tid,pidx]) - ky
                Fbuf[tid,pidx] = kt

    for tid in xrange(nthreads):
        for pidx in xrange(nparam):
            Fout[pidx] += Fbuf[tid,pidx]
    return Fout



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_eq(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:,:,::1] r1lXs, double[:,:,::1] r2lXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the fNL^{eq} template."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(w_mus)
    cdef int il, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPP_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPR1_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPR2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQR1_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQR2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaR1R1_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaR1R2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaR2R2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1.)*27./25.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPP_l[il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] * plXs[il+lmin,jpol,jr]
                        zetaPQ_l[il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] * qlXs[il+lmin,jpol,jr]
                        zetaPR1_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] *r1lXs[il+lmin,jpol,jr]
                        zetaPR2_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] *r2lXs[il+lmin,jpol,jr]
                        zetaQQ_l[il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir] * qlXs[il+lmin,jpol,jr]
                        zetaQR1_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir] *r1lXs[il+lmin,jpol,jr]
                        zetaQR2_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir] *r2lXs[il+lmin,jpol,jr]
                        zetaR1R1_l[il,ir,jr]+= twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*r1lXs[il+lmin,ipol,ir]*r1lXs[il+lmin,jpol,jr]
                        zetaR1R2_l[il,ir,jr]+= twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*r1lXs[il+lmin,ipol,ir]*r2lXs[il+lmin,jpol,jr]
                        zetaR2R2_l[il,ir,jr]+= twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*r2lXs[il+lmin,ipol,ir]*r2lXs[il+lmin,jpol,jr]
                        
    # Compute sum over l, mu for each r, r'
    for ir in prange(nr,nogil=True,schedule='static',num_threads=nthreads):
        for jr in xrange(ir,nr):
            
            deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_eqB(zetaPP_l[:,ir,jr],  zetaPQ_l[:,ir,jr],  zetaPR1_l[:,ir,jr],  zetaPR2_l[:,ir,jr],
                                                                         zetaPQ_l[:,jr,ir],  zetaQQ_l[:,ir,jr],  zetaQR1_l[:,ir,jr],  zetaQR2_l[:,ir,jr],
                                                                         zetaPR1_l[:,jr,ir], zetaQR1_l[:,jr,ir], zetaR1R1_l[:,ir,jr], zetaR1R2_l[:,ir,jr],
                                                                         zetaPR2_l[:,jr,ir], zetaQR2_l[:,jr,ir], zetaR1R2_l[:,jr,ir], zetaR2R2_l[:,ir,jr], legs, w_mus, nmu, nl)
            
            if ir!=jr:
                deriv_matrix[jr,ir] = deriv_matrix[ir,jr]
        
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_orth(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:,:,::1] r1lXs, double[:,:,::1] r2lXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the fNL^{orth} template."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(w_mus)
    cdef int il, ir, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPP_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPR1_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPR2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQR1_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQR2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaR1R1_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaR1R2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaR2R2_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1.)*27./25.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPP_l[il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] * plXs[il+lmin,jpol,jr]
                        zetaPQ_l[il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] * qlXs[il+lmin,jpol,jr]
                        zetaPR1_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] *r1lXs[il+lmin,jpol,jr]
                        zetaPR2_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir] *r2lXs[il+lmin,jpol,jr]
                        zetaQQ_l[il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir] * qlXs[il+lmin,jpol,jr]
                        zetaQR1_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir] *r1lXs[il+lmin,jpol,jr]
                        zetaQR2_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir] *r2lXs[il+lmin,jpol,jr]
                        zetaR1R1_l[il,ir,jr]+= twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*r1lXs[il+lmin,ipol,ir]*r1lXs[il+lmin,jpol,jr]
                        zetaR1R2_l[il,ir,jr]+= twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*r1lXs[il+lmin,ipol,ir]*r2lXs[il+lmin,jpol,jr]
                        zetaR2R2_l[il,ir,jr]+= twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*r2lXs[il+lmin,ipol,ir]*r2lXs[il+lmin,jpol,jr]
                        
    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True,schedule='static',num_threads=nthreads):
        for jr in xrange(ir,nr):
            
            deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_orthB(zetaPP_l[:,ir,jr],  zetaPQ_l[:,ir,jr],  zetaPR1_l[:,ir,jr],  zetaPR2_l[:,ir,jr],
                                                                         zetaPQ_l[:,jr,ir],  zetaQQ_l[:,ir,jr],  zetaQR1_l[:,ir,jr],  zetaQR2_l[:,ir,jr],
                                                                         zetaPR1_l[:,jr,ir], zetaQR1_l[:,jr,ir], zetaR1R1_l[:,ir,jr], zetaR1R2_l[:,ir,jr],
                                                                         zetaPR2_l[:,jr,ir], zetaQR2_l[:,jr,ir], zetaR1R2_l[:,jr,ir], zetaR2R2_l[:,ir,jr], legs, w_mus, nmu, nl)
            if ir!=jr:
                deriv_matrix[jr,ir] = deriv_matrix[ir,jr]
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_fNL_orth2(double[:,:,:,::1] flXs, double[:] weights, double[:,:,::1] inv_Cl_mat, double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the fNL^{orth,2} template."""

    cdef int nl = lmax+1-lmin, nr = flXs[0,0,0].shape[0], npol = flXs[0,0].shape[0], nmu = w_mus.shape[0]
    cdef int il, ir, imu, jr, ipol, jpol, a, b, ab, nbasis=7
    cdef double p = 27./(743./(7.*(20*M_PI*M_PI-193.))-21.)
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    assert len(flXs)==nbasis
    cdef double[:,:,:,:,::1] zeta_l = np.zeros((nbasis,nbasis,nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1.)*27./25.
    cdef int[:,::1] inds_arr
    cdef double[:] coeff_arr

    # List of terms in the basis and their coefficients
    inds_arr = np.asarray([[-2, -2, 4], [-2, -1, 3], [-2, 0, 2], [-2, 1, 1], [-2, 2, 0], [-2, 3, -1], [-2, 4, -2], [-1, -2, 3], [-1, -1, 2], [-1, 0, 1], [-1, 1, 0], [-1, 2, -1], [-1, 3, -2], [0, -2, 2], [0, -1, 1], [0, 0, 0], [0, 1, -1], [0, 2, -2], [1, -2, 1], [1, -1, 0], [1, 0, -1], [1, 1, -2], [2, -2, 0], [2, -1, -1], [2, 0, -2], [3, -2, -1], [3, -1, -2], [4, -2, -2]], dtype=np.int32, order='C')
    coeff_arr = np.asarray([p/27.,(-2*p)/9.,(5*p)/9.,(-20*p)/27.,(5*p)/9.,(-2*p)/9.,p/27.,(-2*p)/9.,-1 - p/3.,1 + (5*p)/9.,1 + (5*p)/9.,-1 - p/3.,(-2*p)/9.,(5*p)/9.,1 + (5*p)/9.,-2 - (20*p)/9.,1 + (5*p)/9.,(5*p)/9.,(-20*p)/27.,1 + (5*p)/9.,1 + (5*p)/9.,(-20*p)/27.,(5*p)/9.,-1 - p/3.,(5*p)/9.,(-2*p)/9.,(-2*p)/9.,p/27.], dtype=np.float64, order='C')
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for ab in prange(nbasis*nbasis, nogil=True, schedule='static',num_threads=nthreads):
        a = ab//nbasis
        b = ab%nbasis
        for il in xrange(nl):
            for ir in xrange(nr):
                for jr in xrange(nr):
                    for ipol in xrange(npol):
                        for jpol in xrange(npol):
                            zeta_l[a,b,il,ir,jr]  += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*flXs[a,il+lmin,ipol,ir] * flXs[b,il+lmin,jpol,jr]
                        
    # Compute sum over l, mu for each r, r'
    for ir in prange(nr, nogil=True,schedule='static',num_threads=nthreads):
        for jr in xrange(ir,nr):
            deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_orth2B(zeta_l[:,:,:,ir,jr], inds_arr, coeff_arr, legs, w_mus, nmu, nl)
            if ir!=jr:
                deriv_matrix[jr,ir] = deriv_matrix[ir,jr]
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_neural_cyclic(double[:,:,:,::1] alpha_lXs, double[:,:,:,::1] beta_lXs, double[:] neural_weights, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the cyclic neural template."""

    cdef int nl = lmax+1-lmin, nr = len(alpha_lXs[0,0,0]), npol = len(alpha_lXs[0,0]), nmu = len(w_mus), nterm = len(neural_weights)
    cdef int iterm, jterm, il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaAA_l, zetaAB_l, zetaBA_l, zetaBB_l
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1.)*9./25.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    for iterm in xrange(nterm):
        for jterm in xrange(nterm):
            zetaAA_l = np.zeros((nl,nr,nr),dtype=np.float64)
            zetaAB_l = np.zeros((nl,nr,nr),dtype=np.float64)
            zetaBA_l = np.zeros((nl,nr,nr),dtype=np.float64)
            zetaBB_l = np.zeros((nl,nr,nr),dtype=np.float64)

            # Compute (2l+1) f^Y S^-1 g^X for each r, r', l
            for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
                for ir in xrange(nr):
                    for jr in xrange(nr):
                        for ipol in xrange(npol):
                            for jpol in xrange(npol):
                                zetaAA_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*alpha_lXs[iterm,il+lmin,ipol,ir]*alpha_lXs[jterm,il+lmin,jpol,jr]
                                zetaAB_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*alpha_lXs[iterm,il+lmin,ipol,ir]*beta_lXs[jterm,il+lmin,jpol,jr]
                                zetaBA_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*beta_lXs[iterm,il+lmin,ipol,ir]*alpha_lXs[jterm,il+lmin,jpol,jr]
                                zetaBB_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*beta_lXs[iterm,il+lmin,ipol,ir]*beta_lXs[jterm,il+lmin,jpol,jr]

            # Compute sum over l, mu for each r, r'
            for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
                ir = ijr//nr
                jr = ijr%nr
                deriv_matrix[ir,jr] += neural_weights[iterm]*neural_weights[jterm]*pref*weights[ir]*weights[jr]*_zeta_sum_symB(zetaBB_l[:,ir,jr], zetaAB_l[:,ir,jr], zetaBA_l[:,ir,jr], zetaAA_l[:,ir,jr], legs, w_mus, nmu, nl)

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_neural(double[:,:,:,::1] alpha_lXs, double[:,:,:,::1] beta_lXs, double[:,:,:,::1] gamma_lXs, double[:] neural_weights, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the general neural template."""

    cdef int nl = lmax+1-lmin, nr = len(alpha_lXs[0,0,0]), npol = len(alpha_lXs[0,0]), nmu = len(w_mus), nterm = len(neural_weights)
    cdef int iterm, jterm, il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaAA_l, zetaAB_l, zetaAC_l, zetaBA_l, zetaBB_l, zetaBC_l, zetaCA_l, zetaCB_l, zetaCC_l
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = dpow(4.*M_PI,-1.)*9./50., pref2, factor
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Define arrays
    zetaAA_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaAB_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaAC_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaBA_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaBB_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaBC_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaCA_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaCB_l = np.zeros((nr,nr,nl),dtype=np.float64)
    zetaCC_l = np.zeros((nr,nr,nl),dtype=np.float64)
            
    for iterm in xrange(nterm):
        for jterm in xrange(nterm):
            zetaAA_l[:,:,:] = 0.
            zetaAB_l[:,:,:] = 0. 
            zetaAC_l[:,:,:] = 0. 
            zetaBA_l[:,:,:] = 0. 
            zetaBB_l[:,:,:] = 0. 
            zetaBC_l[:,:,:] = 0. 
            zetaCA_l[:,:,:] = 0. 
            zetaCB_l[:,:,:] = 0. 
            zetaCC_l[:,:,:] = 0. 

            # Compute (2l+1) f^Y S^-1 g^X for each r, r', l
            for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
                for ir in xrange(nr):
                    for jr in xrange(nr):
                        for ipol in xrange(npol):
                            for jpol in xrange(npol):
                                factor = twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]
                                zetaAA_l[ir,jr,il] += factor*alpha_lXs[iterm,il+lmin,ipol,ir]*alpha_lXs[jterm,il+lmin,jpol,jr]
                                zetaAB_l[ir,jr,il] += factor*alpha_lXs[iterm,il+lmin,ipol,ir]*beta_lXs[jterm,il+lmin,jpol,jr]
                                zetaAC_l[ir,jr,il] += factor*alpha_lXs[iterm,il+lmin,ipol,ir]*gamma_lXs[jterm,il+lmin,jpol,jr]
                                zetaBA_l[ir,jr,il] += factor*beta_lXs[iterm,il+lmin,ipol,ir]*alpha_lXs[jterm,il+lmin,jpol,jr]
                                zetaBB_l[ir,jr,il] += factor*beta_lXs[iterm,il+lmin,ipol,ir]*beta_lXs[jterm,il+lmin,jpol,jr]
                                zetaBC_l[ir,jr,il] += factor*beta_lXs[iterm,il+lmin,ipol,ir]*gamma_lXs[jterm,il+lmin,jpol,jr]
                                zetaCA_l[ir,jr,il] += factor*gamma_lXs[iterm,il+lmin,ipol,ir]*alpha_lXs[jterm,il+lmin,jpol,jr]
                                zetaCB_l[ir,jr,il] += factor*gamma_lXs[iterm,il+lmin,ipol,ir]*beta_lXs[jterm,il+lmin,jpol,jr]
                                zetaCC_l[ir,jr,il] += factor*gamma_lXs[iterm,il+lmin,ipol,ir]*gamma_lXs[jterm,il+lmin,jpol,jr]

            # Compute sum over l, mu for each r, r'
            pref2 = neural_weights[iterm]*neural_weights[jterm]*pref
            for ijr in prange(nr*nr, nogil=True, schedule='static', num_threads=nthreads):
                ir = ijr//nr
                jr = ijr%nr
                deriv_matrix[ir,jr] += pref2*weights[ir]*weights[jr]*_zeta_sum_asymB(zetaAA_l[ir,jr], zetaAB_l[ir,jr], zetaAC_l[ir,jr], zetaBA_l[ir,jr], zetaBB_l[ir,jr], zetaBC_l[ir,jr], zetaCA_l[ir,jr], zetaCB_l[ir,jr], zetaCC_l[ir,jr], legs, w_mus, nmu, nl)
            
    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[:,::1] fisher_deriv_gNL_loc(double[:,:,::1] plXs, double[:,:,::1] qlXs, double[:] weights, double[:,:,::1] inv_Cl_mat,
                                   double[:,::1] legs, double[:] w_mus, int lmin, int lmax, int nthreads):
    """Compute the exact Fisher matrix for the gNL^{loc} template."""

    cdef int nl = lmax+1-lmin, nr = len(plXs[0,0]), npol = len(plXs[0]), nmu = len(w_mus)
    cdef int il, ir, ijr, imu, jr, ipol, jpol
    cdef double XYsum, lsum, musum
    cdef double[:] twol_arr = np.zeros(nl,dtype=np.float64)
    cdef double[:,:,::1] zetaPP_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaPQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zetaQQ_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,:,::1] zeta_l = np.zeros((nl,nr,nr),dtype=np.float64)
    cdef double[:,::1] deriv_matrix = np.zeros((nr,nr),dtype=np.float64)
    cdef double pref = 6./dpow(4.*M_PI,2.)*dpow(9./25.,2.)/2.
    
    # Precompute r-dependent and l-dependent factors
    for il in xrange(nl):
        twol_arr[il] = (2.*il+2*lmin+1.)

    # Compute (2l+1) u^Y S^-1 v^X for each r, r', l
    for il in prange(nl, nogil=True,schedule='static',num_threads=nthreads):
        for ir in xrange(nr):
            for jr in xrange(nr):
                for ipol in xrange(npol):
                    for jpol in xrange(npol):
                        zetaPP_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*plXs[il+lmin,jpol,jr]
                        zetaPQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*plXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]
                        zetaQQ_l[il,ir,jr] += twol_arr[il]*inv_Cl_mat[ipol,jpol,il+lmin]*qlXs[il+lmin,ipol,ir]*qlXs[il+lmin,jpol,jr]

    # Compute sum over l, mu for each r, r'
    for ijr in prange(nr*nr, nogil=True,schedule='static',num_threads=nthreads):
        ir = ijr//nr
        jr = ijr%nr
        deriv_matrix[ir,jr] = pref*weights[ir]*weights[jr]*_zeta_sum_sym(zetaPP_l[:,ir,jr], zetaPQ_l[:,ir,jr], zetaPQ_l[:,jr,ir], zetaQQ_l[:,ir,jr], legs, w_mus, nmu, nl)

    return deriv_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _alpha(int l, int s, int sp) noexcept nogil:
    return sqrt((l**2-s**2)*(l**2-sp**2))/l

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum(double[:] zetaAA_l, double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, lsum
    musum = 0.
    for imu in xrange(nmu):
        lsum = 0.
        for il in xrange(nl):
            lsum += zetaAA_l[il]*legs[imu,il]
        musum += dpow(lsum,4.)*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_symB(double[:] zetaAA_l, double[:] zetaAB_l, double[:] zetaBA_l, double[:] zetaBB_l, 
                          double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the cyclic bispectrum."""
    cdef int il,imu
    cdef double musum, AAsum, ABsum, BAsum, BBsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        ABsum = 0.
        BAsum = 0.
        BBsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
            ABsum += zetaAB_l[il]*legs[imu,il]
            BAsum += zetaBA_l[il]*legs[imu,il]
            BBsum += zetaBB_l[il]*legs[imu,il]
        musum += (AAsum*AAsum*BBsum+2*ABsum*BAsum*AAsum)*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_full_symB(double[:] zetaAA_l, double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the symmetric bispectrum."""
    cdef int il,imu
    cdef double musum, AAsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
        musum += 3*AAsum*AAsum*AAsum*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_eqB(double[:] zetam1m1_l, double[:] zetam1p2_l, double[:] zetam1p1_l, double[:] zetam1p0_l,
                          double[:] zetap2m1_l, double[:] zetap2p2_l, double[:] zetap2p1_l, double[:] zetap2p0_l,
                          double[:] zetap1m1_l, double[:] zetap1p2_l, double[:] zetap1p1_l, double[:] zetap1p0_l,
                          double[:] zetap0m1_l, double[:] zetap0p2_l, double[:] zetap0p1_l, double[:] zetap0p0_l,
                          double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the equilateral bispectrum template."""
    cdef int il,imu
    cdef double musum, m1m1sum, m1p2sum, m1p1sum, m1p0sum, p2m1sum, p2p2sum, p2p1sum, p2p0sum, p1m1sum, p1p2sum, p1p1sum, p1p0sum, p0m1sum, p0p2sum, p0p1sum, p0p0sum
    musum = 0.
    for imu in xrange(nmu):
        m1m1sum = 0.
        m1p0sum = 0.
        m1p1sum = 0.
        m1p2sum = 0.
        p0m1sum = 0.
        p0p0sum = 0.
        p0p1sum = 0.
        p0p2sum = 0.
        p1m1sum = 0.
        p1p0sum = 0.
        p1p1sum = 0.
        p1p2sum = 0.
        p2m1sum = 0.
        p2p0sum = 0.
        p2p1sum = 0.
        p2p2sum = 0.
        for il in xrange(nl):
            m1m1sum += zetam1m1_l[il]*legs[imu,il]
            m1p0sum += zetam1p0_l[il]*legs[imu,il]
            m1p1sum += zetam1p1_l[il]*legs[imu,il]
            m1p2sum += zetam1p2_l[il]*legs[imu,il]
            p0m1sum += zetap0m1_l[il]*legs[imu,il]
            p0p0sum += zetap0p0_l[il]*legs[imu,il]
            p0p1sum += zetap0p1_l[il]*legs[imu,il]
            p0p2sum += zetap0p2_l[il]*legs[imu,il]
            p1m1sum += zetap1m1_l[il]*legs[imu,il]
            p1p0sum += zetap1p0_l[il]*legs[imu,il]
            p1p1sum += zetap1p1_l[il]*legs[imu,il]
            p1p2sum += zetap1p2_l[il]*legs[imu,il]
            p2m1sum += zetap2m1_l[il]*legs[imu,il]
            p2p0sum += zetap2p0_l[il]*legs[imu,il]
            p2p1sum += zetap2p1_l[il]*legs[imu,il]
            p2p2sum += zetap2p2_l[il]*legs[imu,il]
            
        musum += w_mus[imu]*((6*(p1p1sum*p0p0sum*m1m1sum + p1p0sum*p0m1sum*m1p1sum + p1m1sum*p0p1sum*m1p0sum + p1p1sum*p0m1sum*m1p0sum + p1p0sum*p0p1sum*m1m1sum + p1m1sum*p0p0sum*m1p1sum) 
                             -6*(p1m1sum*p0m1sum*m1p2sum + p1m1sum*p0p2sum*m1m1sum + p1p2sum*p0m1sum*m1m1sum)
                             -6*(m1p1sum*m1p0sum*p2m1sum + m1p1sum*m1m1sum*p2p0sum + m1p0sum*m1m1sum*p2p1sum)
                             -12*(p1p0sum*p0p0sum*m1p0sum)
                             -12*(p0p1sum*p0p0sum*p0m1sum)
                             +3*(m1m1sum*m1m1sum*p2p2sum+2*m1p2sum*m1m1sum*p2m1sum)
                             +6*(m1p0sum*m1p0sum*p2p0sum)
                             +6*(p0m1sum*p0m1sum*p0p2sum)
                             +4*(p0p0sum*p0p0sum*p0p0sum)))
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_orth2B(double[:,:,:] zeta_l, int[:,::1] inds_list, double[:] coeff_list, double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the orthogonal2 bispectrum template."""
    cdef int il,imu,x,y,i,j,i1,i2,i3,j1,j2,j3,ncoeff=inds_list.shape[0]
    cdef double musum=0.
    cdef double zs[7][7]
    
    for imu in xrange(nmu):
        # Assemble sum over l for each component
        for x in xrange(7):
            for y in xrange(7):
                zs[x][y] = 0.
                for il in xrange(nl):
                    zs[x][y] += zeta_l[x,y,il]*legs[imu,il]

        # Iterate over terms in the model
        for i in xrange(ncoeff):
            i1 = inds_list[i,0]
            i2 = inds_list[i,1]
            i3 = inds_list[i,2]
            for j in xrange(ncoeff):
                j1 = inds_list[j,0]
                j2 = inds_list[j,1]
                j3 = inds_list[j,2]
        
                # Assemble the cyclic sum
                musum += w_mus[imu]*coeff_list[i]*coeff_list[j]*zs[i1+2][j1+2]*zs[i2+2][j2+2]*zs[i3+2][j3+2]
                
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_orthB(double[:] zetaPP_l, double[:] zetaPQ_l, double[:] zetaPR1_l, double[:] zetaPR2_l,
                          double[:] zetaQP_l, double[:] zetaQQ_l, double[:] zetaQR1_l, double[:] zetaQR2_l,
                          double[:] zetaR1P_l, double[:] zetaR1Q_l, double[:] zetaR1R1_l, double[:] zetaR1R2_l,
                          double[:] zetaR2P_l, double[:] zetaR2Q_l, double[:] zetaR2R1_l, double[:] zetaR2R2_l,
                          double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the orthogonal bispectrum template."""
    cdef int il,imu
    cdef double musum, PPsum, PQsum, PR1sum, PR2sum, QPsum, QQsum, QR1sum, QR2sum, R1Psum, R1Qsum, R1R1sum, R1R2sum, R2Psum, R2Qsum, R2R1sum, R2R2sum
    musum = 0.
    for imu in xrange(nmu):
        PPsum = 0.
        PQsum = 0.
        PR1sum = 0.
        PR2sum = 0.
        QPsum = 0.
        QQsum = 0.
        QR1sum = 0.
        QR2sum = 0.
        R1Psum = 0.
        R1Qsum = 0.
        R1R1sum = 0.
        R1R2sum = 0.
        R2Psum = 0.
        R2Qsum = 0.
        R2R1sum = 0.
        R2R2sum = 0.
        for il in xrange(nl):
            PPsum += zetaPP_l[il]*legs[imu,il]
            PQsum += zetaPQ_l[il]*legs[imu,il]
            PR1sum += zetaPR1_l[il]*legs[imu,il]
            PR2sum += zetaPR2_l[il]*legs[imu,il]
            QPsum += zetaQP_l[il]*legs[imu,il]
            QQsum += zetaQQ_l[il]*legs[imu,il]
            QR1sum += zetaQR1_l[il]*legs[imu,il]
            QR2sum += zetaQR2_l[il]*legs[imu,il]
            R1Psum += zetaR1P_l[il]*legs[imu,il]
            R1Qsum += zetaR1Q_l[il]*legs[imu,il]
            R1R1sum += zetaR1R1_l[il]*legs[imu,il]
            R1R2sum += zetaR1R2_l[il]*legs[imu,il]
            R2Psum += zetaR2P_l[il]*legs[imu,il]
            R2Qsum += zetaR2Q_l[il]*legs[imu,il]
            R2R1sum += zetaR2R1_l[il]*legs[imu,il]
            R2R2sum += zetaR2R2_l[il]*legs[imu,il]

        musum += w_mus[imu]*((54*(R1R1sum*R2R2sum*PPsum + R1R2sum*R2R1sum*PPsum + R1R2sum*R2Psum*PR1sum + R1Psum*R2R1sum*PR2sum + R1R1sum*R2Psum*PR2sum + R1Psum*R2R2sum*PR1sum) 
                             -54*(R1Psum*R2Psum*PQsum + R1Psum*R2Qsum*PPsum + R1Qsum*R2Psum*PPsum)
                             -54*(PR1sum*PR2sum*QPsum + PR1sum*PPsum*QR2sum + PR2sum*PPsum*QR1sum)
                             -144*(R1R2sum*R2R2sum*PR2sum)
                             -144*(R2R1sum*R2R2sum*R2Psum)
                             +27*(PPsum*PPsum*QQsum+2*PQsum*PPsum*QPsum)
                             +72*(PR2sum*PR2sum*QR2sum)
                             +72*(R2Psum*R2Psum*R2Qsum)
                             +64*(R2R2sum*R2R2sum*R2R2sum)))
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_asymB(double[::1] zetaAA_l, double[::1] zetaAB_l, double[::1] zetaAC_l, double[::1] zetaBA_l, double[::1] zetaBB_l, double[::1] zetaBC_l, double[::1] zetaCA_l, double[::1] zetaCB_l, double[::1] zetaCC_l, double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators. This is a specialized version for the bispectrum."""
    cdef int il,imu
    cdef double AAsum, ABsum, ACsum, BAsum, BBsum, BCsum, CAsum, CBsum, CCsum
    cdef double leg_val, weight, musum=0.
    for imu in xrange(nmu):
        weight = w_mus[imu]
        AAsum = 0.
        ABsum = 0.
        ACsum = 0.
        BAsum = 0.
        BBsum = 0.
        BCsum = 0.
        CAsum = 0.
        CBsum = 0.
        CCsum = 0.
        for il in xrange(nl):
            leg_val = legs[imu,il]
            AAsum += zetaAA_l[il]*leg_val
            ABsum += zetaAB_l[il]*leg_val
            ACsum += zetaAC_l[il]*leg_val
            BAsum += zetaBA_l[il]*leg_val
            BBsum += zetaBB_l[il]*leg_val
            BCsum += zetaBC_l[il]*leg_val
            CAsum += zetaCA_l[il]*leg_val
            CBsum += zetaCB_l[il]*leg_val
            CCsum += zetaCC_l[il]*leg_val
        musum += weight*(AAsum*BBsum*CCsum+AAsum*BCsum*CBsum+ABsum*BCsum*CAsum+ABsum*BAsum*CCsum+ACsum*BAsum*CBsum+ACsum*BBsum*CAsum)
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_sym(double[:] zetaAA_l, double[:] zetaAB_l, double[:] zetaBA_l, double[:] zetaBB_l, 
                          double[:,::1] legs, double[:] w_mus, int nmu, int nl) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, AAsum, ABsum, BAsum, BBsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        ABsum = 0.
        BAsum = 0.
        BBsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
            ABsum += zetaAB_l[il]*legs[imu,il]
            BAsum += zetaBA_l[il]*legs[imu,il]
            BBsum += zetaBB_l[il]*legs[imu,il]
        musum += (AAsum*AAsum*AAsum*BBsum+3*ABsum*BAsum*AAsum*AAsum)*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_sym2(double[:] zetaAA_l, double[:] zetaBB_l, double[:] zetaCC_l,
                          double[:] zetaAB_l, double[:] zetaAC_l, double[:] zetaBC_l,
                          double[:] zetaBA_l, double[:] zetaCA_l, double[:] zetaCB_l,
                          double[:,::1] legs, double[:,::1] wig10s, double[:,::1] wig11s, double[:,::1] wig1m1s, 
                          double[:] w_mus, int nmu, int nl, int lmin) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, tmp, AAsum, BBsum, CCsum, CsCsum, ABsum, BAsum, ACsum, CAsum, BCsum, CBsum
    musum = 0.
    for imu in xrange(nmu):
        AAsum = 0.
        BBsum = 0.
        CCsum = 0.
        CsCsum = 0.
        ABsum = 0.
        BAsum = 0.
        ACsum = 0.
        CAsum = 0.
        BCsum = 0.
        CBsum = 0.
        for il in xrange(nl):
            AAsum += zetaAA_l[il]*legs[imu,il]
            BBsum += zetaBB_l[il]*legs[imu,il]
            CCsum += zetaCC_l[il]*wig1m1s[imu,lmin+il-1]
            CsCsum -= zetaCC_l[il]*wig11s[imu,lmin+il-1]
            ABsum += zetaAB_l[il]*legs[imu,il]
            BAsum += zetaBA_l[il]*legs[imu,il]
            ACsum -= zetaAC_l[il]*wig10s[imu,lmin+il-1]
            CAsum -= zetaCA_l[il]*wig10s[imu,lmin+il-1]
            BCsum -= zetaBC_l[il]*wig10s[imu,lmin+il-1]
            CBsum -= zetaCB_l[il]*wig10s[imu,lmin+il-1]
        tmp = AAsum*AAsum*(BBsum*BBsum+BCsum*BCsum+CBsum*CBsum+CsCsum*CsCsum/2.+CCsum*CCsum/2.)
        tmp += 4.*AAsum*(ABsum*BAsum*BBsum+ACsum*BAsum*BCsum+ABsum*CAsum*CBsum+ACsum*CAsum*(CCsum+CsCsum)/2.)
        tmp += ABsum*ABsum*(BAsum*BAsum+CAsum*CAsum)+ACsum*ACsum*(BAsum*BAsum+CAsum*CAsum)
        musum += tmp*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double _zeta_sum_sym3(double[:] zetaBB_l, double[:] zetaCC_l,
                           double[:] zetaBC_l, double[:] zetaCB_l,
                           double[:,::1] legs, double[:,::1] wig10s, double[:,::1] wig11s, double[:,::1] wig1m1s, 
                           double[:] w_mus, int nmu, int nl, int lmin) noexcept nogil:
    """Utility function to sum over l, mu in the exact estimators."""
    cdef int il,imu
    cdef double musum, tmp, BBsum, CCsum, CsCsum, BCsum, CBsum
    musum = 0.
    for imu in xrange(nmu):
        BBsum = 0.
        CCsum = 0.
        CsCsum = 0.
        BCsum = 0.
        CBsum = 0.
        for il in xrange(nl):
            BBsum += zetaBB_l[il]*legs[imu,il]
            CCsum += zetaCC_l[il]*wig1m1s[imu,lmin+il-1]
            CsCsum -= zetaCC_l[il]*wig11s[imu,lmin+il-1]
            BCsum -= zetaBC_l[il]*wig10s[imu,lmin+il-1]
            CBsum -= zetaCB_l[il]*wig10s[imu,lmin+il-1]
        tmp = 6*(BBsum*BBsum*BBsum*BBsum+BCsum*BCsum*BCsum*BCsum+CBsum*CBsum*CBsum*CBsum)+12*BBsum*BBsum*(BCsum*BCsum+CBsum*CBsum)+4*(BCsum*BCsum+CBsum*CBsum)*(CsCsum*CCsum+CCsum*CCsum+CsCsum*CsCsum)
        tmp += 2*BBsum*BBsum*(CsCsum*CsCsum+CCsum*CCsum)+8*BBsum*BCsum*CBsum*(CsCsum+CCsum)+4*BCsum*BCsum*CBsum*CBsum
        tmp += CCsum*CCsum*CCsum*CCsum+CsCsum*CsCsum*CsCsum*CsCsum+4*CsCsum*CsCsum*CCsum*CCsum
        musum += tmp*w_mus[imu]
    return musum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double lensing_isw_sum_sym(double complex[:,::1] u1, double complex[:,::1] v1, double complex[:,::1] s1, int nthreads):
    """Compute the sum over ISW u, v, s maps"""
    cdef double summ=0.
    cdef int i, npix = u1.shape[1]
    for i in prange(npix, nogil=True,schedule='static',num_threads=nthreads):
        summ += creal(u1[0,i]*v1[0,i]*(-v1[0,i].conjugate()*s1[0,i]+v1[0,i]*s1[1,i]))
    return summ      

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double lensing_isw_sum(double complex[:,::1] u1, double complex[:,::1] u2, double complex[:,::1] v1, double complex[:,::1] v2, double complex[:,::1] s1, double complex[:,::1] s2, int nthreads):
    """Compute the sum over ISW u, v, s maps"""
    cdef double summ=0.
    cdef int i, npix = u1.shape[1]
    for i in prange(npix, nogil=True,schedule='static',num_threads=nthreads):
        summ += 2.*creal(u1[0,i]*v1[0,i]*(-v2[0,i].conjugate()*s2[0,i]+v2[0,i]*s2[1,i]))
        summ += 2.*creal(u2[0,i]*v2[0,i]*(-v1[0,i].conjugate()*s1[0,i]+v1[0,i]*s1[1,i]))
        summ += creal(u1[0,i]*v2[0,i]*(-v2[0,i].conjugate()*s1[0,i]+v2[0,i]*s1[1,i]))
        summ += creal(u2[0,i]*v1[0,i]*(-v1[0,i].conjugate()*s2[0,i]+v1[0,i]*s2[1,i]))
    return summ      

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double isw_bispectrum_sum(complex[:,::1] umap, complex[:,::1] vmap, complex[::1] vmap_isw, int nthreads):
    """Compute the sum over U and V maps required for the lensing-ISW bispectrum numerator""" 
    cdef int i, ipol, npol = umap.shape[0], npix = umap.shape[1]
    cdef double out=0.

    # Spin-0
    if npol==1:
        for i in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
            out += 2.*creal(umap[0,i]*vmap[0,i].conjugate()*vmap_isw[i])

    # All spins    
    else:
        for i in prange(npix, nogil=True, schedule='static', num_threads=nthreads):
            out += creal(2.*umap[0,i]*vmap[0,i].conjugate()*vmap_isw[i]+(umap[1,i]+1.0j*umap[2,i])*vmap[1,i].conjugate()*vmap_isw[i]-(umap[1,i]+1.0j*umap[2,i])*vmap[2,i].conjugate()*vmap_isw[i].conjugate())
    return out