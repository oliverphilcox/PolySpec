#cython: language_level=3

from __future__ import print_function
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs, M_PI, sqrt
from cython.parallel import prange
from libc.stdlib cimport malloc, free

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
        cdef int nlm = h_lm.shape[1]
        cdef int npol = flX.shape[1]
        cdef int nr = flX.shape[2]
        cdef int ir, ilm, l
        cdef int[:] ls = self.ls

        # Define output (using fortran ordering for speed)
        cdef np.ndarray[np.complex128_t, ndim=2, mode='fortran'] out = np.zeros((nr, nlm), dtype=np.complex128, order='F')
        # Define fortran-contiguous view
        cdef double complex[::1, :] out_v = out

        cdef double complex h0, h1
        cdef double val0, val1

        # Iterate over l, r and sum over polarizations
        if npol == 1:
            for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                l = ls[ilm]
                h0 = h_lm[0, ilm] * weight
                for ir in range(nr):
                    out_v[ir, ilm] = flX[l, 0, ir] * h0
        else:
            for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                l = ls[ilm]
                h0 = h_lm[0, ilm] * weight
                h1 = h_lm[1, ilm] * weight
                for ir in range(nr):
                    val0 = flX[l, 0, ir]
                    val1 = flX[l, 1, ir]
                    out_v[ir, ilm] = val0 * h0 + val1 * h1
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef np.ndarray[np.complex128_t,ndim=1] apply_fl_weight_single(self, double[:,:,::1] flX, complex[:,::1] h_lm, int index, double weight):
        """Compute w*f_l^X*h^X_lm, summing over polarizations X (assuming only T and/or E contribute). This version is for a single radial component."""

        cdef int nlm = h_lm.shape[1], npol = flX.shape[1]
        cdef int ilm, l
        cdef int[:] ls = self.ls
        cdef np.ndarray[np.complex128_t,ndim=1] out = np.empty(nlm,dtype=np.complex128)
        cdef double complex[::1] out_view = out

        # Iterate over l, r and sum over polarizations
        if npol==1:
            for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                l = ls[ilm]
                out_view[ilm] = flX[l,0,index]*h_lm[0,ilm]*weight
        else:
            for ilm in prange(nlm, nogil=True, schedule='static', num_threads=self.nthreads):
                l = ls[ilm]
                out_view[ilm] = (flX[l,0,index]*h_lm[0,ilm]+flX[l,1,index]*h_lm[1,ilm])*weight  
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
    cpdef double complex fnl_sum_2d_complex(self, double[:] r_weights, double complex[:] u_weights, double[:,:,::1] A_maps, double[:,:,::1] B_maps, double[:,:,::1] C_maps):
        """Compute Sum_ir w_ir Sum_iu w_iu(complex) A(r,u)B(r,u)C(r,u) for real maps A, B, C with a
        complex u-weighting."""
        cdef int ir, iu, ipix, nr = A_maps.shape[0], nu = A_maps.shape[1], npix = A_maps.shape[2]
        cdef double tmp_sum2, tmp_sum_re, tmp_sum_im, u_re, u_im, out_re = 0., out_im = 0.

        for ir in prange(nr, nogil=True, schedule='static', num_threads=self.nthreads):
            tmp_sum_re = 0.
            tmp_sum_im = 0.
            for iu in xrange(nu):
                u_re = u_weights[iu].real
                u_im = u_weights[iu].imag
                tmp_sum2 = 0.
                for ipix in xrange(npix):
                    tmp_sum2 = tmp_sum2 + A_maps[ir,iu,ipix]*B_maps[ir,iu,ipix]*C_maps[ir,iu,ipix]
                tmp_sum_re = tmp_sum_re + u_re*tmp_sum2
                tmp_sum_im = tmp_sum_im + u_im*tmp_sum2
            out_re += r_weights[ir]*tmp_sum_re
            out_im += r_weights[ir]*tmp_sum_im
        return complex(out_re, out_im)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _fnl_sum(self, double[:] r_weights, double[:,::1] A_maps, double[:,::1] B_maps, double[:,::1] C_maps) noexcept nogil:
        """Compute Sum_i w_i A_i(r)B_i(r)C_i(r) for maps A, B, C."""
        cdef int ir, ipix, nr = A_maps.shape[0], npix = A_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in xrange(nr):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum = tmp_sum + A_maps[ir,ipix]*B_maps[ir,ipix]*C_maps[ir,ipix]
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _fnl_sum_linear(self, double[:] r_weights, double[:,::1] A_maps, double[:,::1] B_maps, double[:,::1] C_maps, double[:,::1] sA_maps, double[:,::1] sB_maps, double[:,::1] sC_maps) noexcept nogil:
        """Compute Sum_i w_i A_i(r)B_i(r)C_i(r) for maps A, B, C."""
        cdef int ir, ipix, nr = A_maps.shape[0], npix = A_maps.shape[1]
        cdef double tmp_sum, out = 0.

        for ir in xrange(nr):
            tmp_sum = 0.
            for ipix in xrange(npix):
                tmp_sum += (A_maps[ir,ipix]*sB_maps[ir,ipix]*sC_maps[ir,ipix]+sA_maps[ir,ipix]*B_maps[ir,ipix]*sC_maps[ir,ipix]+sA_maps[ir,ipix]*sB_maps[ir,ipix]*C_maps[ir,ipix])
            out += r_weights[ir]*tmp_sum
        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef double neural_sum(self, double[:] r_weights, double[:] neural_weights, double[:,:,::1] A_maps, double[:,:,::1] B_maps, double[:,:,::1] C_maps):
        """Compute Sum_n W_n Sum_i w_i An_i(r)Bn_i(r)Cn_i(r) for maps A, B, C."""
        cdef int iterm, ir, ipix, nterm = A_maps.shape[0], nr = A_maps.shape[1], npix = A_maps.shape[2]
        cdef double tmp_sum, tmp_sum2 = 0, out = 0.

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
    cpdef void assemble_binned_bispectrum_Q(self, int[:,::1] last_bins, int[:] counts, complex[:] lm_map, double[:,:,:,::1] flXs, int radial_index, double[:] weight, complex[:,:,::1] Q, int q_index):
        """Assemble the binned bispectrum Fisher derivative."""
        cdef int nconfig = last_bins.shape[0], nlm = lm_map.shape[0], npol = flXs.shape[1]
        cdef int ilm, ipol, iconfig, l, binC, bin_id
        cdef double pref

        with nogil:
            for ilm in prange(nlm, schedule='static', num_threads=self.nthreads):
                l = self.ls[ilm]
                for iconfig in xrange(nconfig):
                    binC = last_bins[iconfig,0]
                    bin_id = last_bins[iconfig,1]
                    pref = counts[iconfig]*36./5./last_bins[iconfig,2]*weight[radial_index]
                    for ipol in xrange(npol):
                        Q[q_index+bin_id,ipol,ilm] += pref*lm_map[ilm]*flXs[l,ipol,radial_index,binC]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void assemble_b3_binned(self, int[:] unique_bin_ids, int[:,::1] k_triplet_ids, int[:] bin_degeneracy, double[:] b3_num, int index, double A_pix, double[:] r_weights, double[:,:,::1] proc_maps):
        """Assemble the cubic term of the binned bispectrum output."""
        cdef int nbin = unique_bin_ids.shape[0]
        cdef int ntriplets = k_triplet_ids.shape[0]
        cdef double base_prefactor = 3.0/5.0 * 6.0 * A_pix
        
        # First build an adjacency list for efficient summation
        cdef int max_bin_id = 0
        cdef int i
        for i in range(nbin):
            if unique_bin_ids[i] > max_bin_id:
                max_bin_id = unique_bin_ids[i]
        cdef int *head = <int *> malloc((max_bin_id + 1) * sizeof(int))
        cdef int *tail = <int *> malloc((max_bin_id + 1) * sizeof(int))
        cdef int *next_trip = <int *> malloc(ntriplets * sizeof(int))
        
        if head == NULL or tail == NULL or next_trip == NULL:
            if head != NULL: free(head)
            if tail != NULL: free(tail)
            if next_trip != NULL: free(next_trip)
            raise MemoryError("Failed to allocate adjacency list")

        for i in range(max_bin_id + 1):
            head[i] = -1
            tail[i] = -1

        cdef int bin_id_iter
        for i in range(ntriplets):
            bin_id_iter = k_triplet_ids[i, 3]
            next_trip[i] = -1 # Default to end of chain
            if bin_id_iter <= max_bin_id:
                if head[bin_id_iter] == -1:
                    # First item in list
                    head[bin_id_iter] = i
                    tail[bin_id_iter] = i
                else:
                    # Append to end of list using tail pointer
                    next_trip[tail[bin_id_iter]] = i
                    tail[bin_id_iter] = i

        # Iterate over bins
        cdef int ibin, current_triplet
        cdef int b1, b2, b3, bin_id
        cdef double term, combined_weight
        for ibin in prange(nbin, schedule='static', num_threads=self.nthreads, nogil=True):
            bin_id = unique_bin_ids[ibin]
            
            current_triplet = head[bin_id]
            
            while current_triplet != -1:
                # Load triplet indices
                b1 = k_triplet_ids[current_triplet, 0]
                b2 = k_triplet_ids[current_triplet, 1]
                b3 = k_triplet_ids[current_triplet, 2]
                
                # Compute term and add to output
                term = self._fnl_sum(r_weights, proc_maps[b1], proc_maps[b2], proc_maps[b3])
                combined_weight = base_prefactor / bin_degeneracy[current_triplet]
                b3_num[index + ibin] = b3_num[index + ibin] + (combined_weight * term)
                
                # Move to next triplet
                current_triplet = next_trip[current_triplet]

        # Cleanup
        free(head)
        free(tail)
        free(next_trip)
                        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef void assemble_b1_binned(self, int[:] unique_bin_ids, int[:,::1] k_triplet_ids, int[:] bin_degeneracy, double[:] b1_num, int index, double A_pix, int N_it, double[:] r_weights, double[:,:,::1] proc_maps, double[:,:,::1] sim_proc_maps):
        """
        Assemble the linear term of the binned bispectrum output.
        """
        cdef int nbin = unique_bin_ids.shape[0]
        cdef int ntriplets = k_triplet_ids.shape[0]
        
        # Precompute loop constants
        cdef double base_prefactor = -3.0/5.0*6.0 * A_pix / N_it
        cdef int nthreads = self.nthreads
        
        # First build an adjacency list for efficient summation
        cdef int max_bin_id = 0
        cdef int i
        for i in range(nbin):
            if unique_bin_ids[i] > max_bin_id:
                max_bin_id = unique_bin_ids[i]
        
        cdef int *head = <int *> malloc((max_bin_id + 1) * sizeof(int))
        cdef int *tail = <int *> malloc((max_bin_id + 1) * sizeof(int))
        cdef int *next_trip = <int *> malloc(ntriplets * sizeof(int))
        
        if head == NULL or tail == NULL or next_trip == NULL:
            if head != NULL: free(head)
            if tail != NULL: free(tail)
            if next_trip != NULL: free(next_trip)
            raise MemoryError("Failed to allocate adjacency list")

        for i in range(max_bin_id + 1):
            head[i] = -1
            tail[i] = -1

        cdef int bin_id_iter
        for i in range(ntriplets):
            bin_id_iter = k_triplet_ids[i, 3]
            next_trip[i] = -1 # Default to end of chain
            
            if bin_id_iter <= max_bin_id:
                if head[bin_id_iter] == -1:
                    # First item
                    head[bin_id_iter] = i
                    tail[bin_id_iter] = i
                else:
                    # Append to tail
                    next_trip[tail[bin_id_iter]] = i
                    tail[bin_id_iter] = i

        # Iterate over bins
        cdef int ibin, current_triplet
        cdef int b1, b2, b3, bin_id
        cdef double term, combined_weight
        for ibin in prange(nbin, schedule='static', num_threads=nthreads, nogil=True):
            bin_id = unique_bin_ids[ibin]
            current_triplet = head[bin_id]
            
            while current_triplet != -1:
                # Load triplet indices
                b1 = k_triplet_ids[current_triplet, 0]
                b2 = k_triplet_ids[current_triplet, 1]
                b3 = k_triplet_ids[current_triplet, 2]
                
                # Compute term and add to output
                term = self._fnl_sum_linear(r_weights, proc_maps[b1], proc_maps[b2], proc_maps[b3], sim_proc_maps[b1], sim_proc_maps[b2], sim_proc_maps[b3])
                combined_weight = base_prefactor / bin_degeneracy[current_triplet]
                b1_num[index + ibin] = b1_num[index + ibin] + (combined_weight * term)
                
                # Move to next triplet
                current_triplet = next_trip[current_triplet]

        # Cleanup
        free(head)
        free(tail)
        free(next_trip)
    
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