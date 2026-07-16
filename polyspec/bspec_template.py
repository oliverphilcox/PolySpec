### Code for binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2026)
## This module contains the bispectrum template estimation code

import numpy as np
import time
from scipy.special import p_roots, lpmn
from .cython.k_integrals import *
from .cython.ideal_fisher import *
from .cython.fNL_utils import *

class BSpecTemplate():
    """
    Bispectrum estimation class for measuring the amplitudes of separable primoridal bispectrum templates. 
    We also feed in a function that applies the S^-1 operator (which is ideally beam.mask.C_l^{tot,-1}, where C_l^tot includes the beam and noise). 
    
    Inputs:
    - base: PolySpec class
    - mask: HEALPix mask applied to data. We can optionally specify a vector of three masks for [T, Q, U].
    - applySinv: function which returns S^-1 ~ P^dag Cov^{-1} in harmonic space, when applied to a given input map, where P = Mask * Beam.
    - templates: types of templates to compute e.g. [fNL-loc, fNL-eq, fNL-orth, fNL-orth2, neural-0, neural-2, binned, ..]
    - k_arr, Tl_arr: k-array, plus T- and (optionally) E-mode transfer functions for all ell. Required for all primordial templates.
    - lmin, lmax: minimum/maximum ell (inclusive)
    - ns, As, k_pivot: primordial power spectrum parameters
    - r_values, r_weights: radial sampling points and weights for 1-dimensional integrals
    - C_Tphi, C_Ephi: cross spectrum of temperature/polarization and lensing  [C^Tphi_0, C^Tphi_1, etc.]. Required if 'isw-lensing' is in templates.
    - C_lens_weight: dictionary of lensed power spectra (TT, TE, etc.). Required if 'isw-lensing' is in templates.
    - r_star, r_hor: Comoving distance to last-scattering and the horizon (default: Planck 2018 values).
    - neural_inputs: Input set of neural-network bispectrum templates (only used if "neural-0, neural-1, ..." is in templates). These must take the form ({weights_0, alpha_0, beta_0, [gamma_0]}, {weights_1, alpha_1, beta_1, [gamma_1]}, ...), where alpha/beta/gamma are functions of k and i, for i = 0 ... len(weights)-1. 
    - k_bin_edges: List of Fourier-bin edges for the binned estimator: (k_1, k_2, k_3, ...). If specified, "binned" should be in templates.
    - k_triplet_ids: List of {bin1, bin2, bin3, bin-id}. If specified, "binned" should be in templates.
    - feat_params: Additional parameters for the feature templates.
    """
    def __init__(self, base, mask, applySinv, templates, lmin, lmax,  k_arr=[], Tl_arr=[], r_arr=[], ns=0.96, As=2.1e-9, k_pivot=0.05, r_values = [], r_weights = {}, C_Tphi=[], C_Ephi=[], C_lens_weight = {}, r_star=None, r_hor=None, neural_inputs=None, k_bin_edges=None, k_triplet_ids=None, feat_params=None):
        # Read in attributes
        self.base = base
        self.mask = mask
        self.applySinv = applySinv
        self.pol = self.base.pol
        self.templates = templates
        self.k_arr = k_arr
        self.lmin = lmin
        self.lmax = lmax
        self.ns = ns
        self.As = As
        self.k_pivot = k_pivot
        self.neural_inputs = neural_inputs
        self.k_bin_edges = k_bin_edges
        self.k_triplet_ids = k_triplet_ids
        self.feat_params = feat_params
        
        # Create primordial power spectrum function
        print("Primordial Spectrum: n_s = %.2f, A_s = %.2e, k_pivot = %.2f"%(self.ns, self.As, self.k_pivot));
        self.Pzeta = lambda k: 2.*np.pi**2./k**3*self.As*(k/self.k_pivot)**(self.ns-1)
        
        # Check ell ranges
        assert self.lmax<=base.lmax, "Maximum l can't be larger than HEALPix resolution!"
        assert self.lmin>=2, "Minimum l can't be less than 2"
        if self.lmax>(base.lmax+1)*2/3: print("## Caution: Maximum l is greater than (2/3)*HEALPix-lmax; this might cause boundary effects.")
        
        # Compute filters for ell range of interest
        self.lfilt = (self.base.l_arr>=self.lmin)&(self.base.l_arr<=self.lmax)
        self.ls, self.ms = self.base.l_arr[self.lfilt], self.base.m_arr[self.lfilt]
        self.m_weight = np.asarray(self.base.m_weight[self.lfilt],order='C')
        self.Cinv = np.asarray(self.base.inv_Cl_tot_lm_mat[:,:,self.lfilt],order='C')
        
        # Define beam (in our ell-range)
        self.beam_lm = np.asarray(self.base.beam_lm)[:,self.lfilt]
        
        # Print ell ranges
        print("l-range: %s"%[self.lmin,self.lmax])
        
        # Print polarizations
        if self.pol:
            print("Polarizations: ['T', 'E']")
        else:
            print("Polarizations: ['T']")
        
        # Configure template parameters and limits
        self._configure_templates(templates, C_Tphi, C_Ephi, C_lens_weight)
        
        # Check mask properties
        if not (type(mask)==float or type(mask)==int):
            if len(mask)==1 or len(mask)==3:
                assert len(mask[0])==self.base.Npix, f'Mask has incorrect shape: {mask.shape}'
            else:
                assert len(mask)==self.base.Npix, f'Mask has incorrect shape: {mask.shape}'
        if np.std(self.mask)<1e-12 and np.abs(np.mean(self.mask)-1)<1e-12:
            print("Mask: ones")
            self.ones_mask = True
        else:
            print("Mask: spatially varying")
            self.ones_mask = False
        
        # Define fixed points for radial sampling
        if r_star is None:
            self.r_star = 13882.607764400758 # recombination distance (Planck 2018 cosmology)
        else:
            self.r_star = r_star
        if r_hor is None:
            self.r_hor = 14163.032903769747 # horizon distance (Planck 2018 cosmology)
        else:
            self.r_hor = r_hor
        
        # Check transfer function and initialize arrays
        if self.ints_1d:
            self.Tl_arr = np.asarray(Tl_arr,dtype=np.float64,order='C')
            if not self.pol:
                assert len(self.Tl_arr)==1, "Transfer function should contain only T components"
            else:
                assert len(self.Tl_arr)==2, "Transfer function should contain both T and E components"
            assert len(self.Tl_arr[0])>=self.lmax+1, "Transfer function must contain all ell modes of interest"
            assert self.Tl_arr[0][:2].sum()==0., "Transfer function should return zero for ell = 0, 1"
            assert len(self.Tl_arr[0][0])==len(k_arr), "Transfer function must be computed on the input k-grid"
            if self.pol: assert self.Tl_arr[1][:2].sum()==0., "Transfer function should return zero for ell = 0, 1"
        
        # Initialize timers
        self.timers = {x: 0. for x in ['precomputation','numerator','fisher','optimization','analytic_fisher',
                                       'fish_grfs', 'fish_outer', 'fish_deriv', 'fish_products',
                                       'Sinv','Ainv','map_transforms','fNL_summation','lensing_summation']}
        self.base.time_sht = 0.
        
        # Check sampling points
        if len(r_values)==0 and self.ints_1d:
            print("# No input radial sampling points supplied; these can be computed with the optimize_radial_sampling_1d() function\n") 
            self.N_r = 0
        elif self.ints_1d:
            print("Reading in precomputed radial integration points")
            
            assert len(r_values)>0, "Must supply radial sampling points!"
            for t in templates:
                if (t in self.all_templates_1d) or ('neural' in t):
                    assert t in r_weights.keys(), "Must supply weight for template %s"%t
            self.r_arr = r_values
            self.N_r = len(self.r_arr)
            self.r_weights = r_weights
        else:
            self.N_r = 0
        
        # Precompute k-space integrals if r arrays have been supplied
        if self.N_r>0:
            self._prepare_templates(self.ints_1d)
            
    ### UTILITY FUNCTIONS
    def _configure_templates(self, templates, C_Tphi, C_Ephi, C_lens_weight):
        """Check input templates and log which quantities to compute."""
        
        # Check correct templates are being used and print them
        self.all_templates_1d = ['fNL-loc','fNL-eq','fNL-orth','fNL-orth2','fNL-feat-sharp','fNL-feat-res','binned']
        self.all_templates = self.all_templates_1d+['isw-lensing']
        ii = 0
        neural_inds = []
        for t in templates:
            ii += 1
            if 'neural' in t:
                neural_inds.append(int(t.split('-')[1]))
            else:   
                assert t in self.all_templates, "Unknown template %s supplied!"%t
        print("Templates: %s"%templates)
        
        def _merge_dict(d1,d2):
            """Merge dictionaries and drop duplicates"""
            for key in d2.keys():
                if key in d1.keys():
                    new_list = []
                    for item in d2[key]:
                        if item not in new_list: new_list.append(item)
                    for item in d1[key]:
                        if item not in new_list: new_list.append(item)
                    d1[key] = new_list
                else:
                    new_list = []
                    for item in d2[key]:
                        if item not in new_list: new_list.append(item)
                    d1[key] = new_list
        
        # Store list of quantities to compute
        self.ints_1d = False
        self.to_compute = []
        
        # Check which integrals to compute
        if 'fNL-loc' in templates:
            self.to_compute.append(['f_m1','f_p2'])
            self.ints_1d = True
        
        if 'fNL-eq' in templates:
            self.to_compute.append(['f_m1','f_p0','f_p1','f_p2'])
            self.ints_1d = True
        
        if 'fNL-orth' in templates:
            self.to_compute.append(['f_m1','f_p0','f_p1','f_p2'])
            self.ints_1d = True
        
        if 'fNL-orth2' in templates:
            self.to_compute.append(['f_m2','f_m1','f_p0','f_p1','f_p2','f_p3','f_p4'])
            self.ints_1d = True

        if 'fNL-feat-sharp' in templates:
            print("#TODO")
            self.to_compute.append(['f_osc'])
            self.ints_1d = True

        if 'fNL-feat-res' in templates:
            self.to_compute.append(['f_exp'])
            self.ints_1d = True
            assert 'kres_cs' in self.feat_params, "Must specify kres_cs to use resonant templates!"
            assert 'omega' in self.feat_params, "Must specify omega to use resonant templates!"
            self.omega = self.feat_params['omega']
            kappa = self.feat_params['kres_cs']   # kappa = k_res/c_s

            # Integration scheme for the (non-separable) scalar u-integral factor
            #   F(K) = cosh(pi w/2) Gamma(iw) (K/kappa)^{-iw},  K = k1+k2+k3,
            # which -- since S(u)=S_0 e^{-Ku} factorises exactly -- is the ENTIRE difficulty. Both schemes
            # represent F(K) as a compressed separable exponential sum  F(K) ~ sum_j W_j e^{-K u_j}  with
            # complex nodes u_j on the rotated contour and complex weights W_j (fit by pivoted-QR + lstsq):
            #   'sm3' (default): reconstruct with the m=n bracket    -> legs k^{-3..0}   (no k^{+1}, no r-aliasing)
            #   'sm1'          : reconstruct with the K-raised bracket-> legs k^{-3..+1} (worse-conditioned weights)
            # Both give the same B_res; see FEAT_NOTES.md Sessions 4-6.
            self.feat_scheme = self.feat_params.get('integration_scheme', 'sm3')
            assert self.feat_scheme in ('sm3','sm1'), "integration_scheme must be 'sm3' or 'sm1'"
            self.feat_kpows = [-3,-2,-1,0] if self.feat_scheme=='sm3' else [-3,-2,-1,0,1]

            # e1=k1+k2+k3 range over which the exp-sum must be valid. For k in [kmin,kmax] a triangle has
            # e1 in [3 kmin, 3 kmax] exactly. Overridable via feat_params ('e1_min','e1_max','expsum_tol').
            assert len(self.k_arr)>0, "Must supply k_arr to use resonant templates!"
            Kmin = self.feat_params.get('e1_min', 3.*np.min(self.k_arr))
            Kmax = self.feat_params.get('e1_max', 3.*np.max(self.k_arr))
            tol  = self.feat_params.get('expsum_tol', 1e-6)
            # Build the compressed complex nodes u_j and complex weights W_j. sm1 reconstructs with the
            # K-raised bracket, so its u-factor target is K^{-iw-1} (kpow_shift=-1); sm3 uses K^{-iw} (shift 0).
            kpow_shift = 0 if self.feat_scheme=='sm3' else -1
            # Store the REAL rotated-contour t-grid + the fixed rotation angle phi. The complex node is
            # u = t*e^{i phi}; the rotation is applied only inside the (complex) leg integral. All nodes
            # share this single phi, so t (real, positive, monotone) is the natural stored grid.
            # Optional grid_omega: build the node grid at a (larger) frequency and reuse it for self.omega
            # (one node set for a whole omega-scan; see _build_feat_res_expsum). None -> native per-omega grid.
            grid_omega = self.feat_params.get('grid_omega', None)
            phi_factor = self.feat_params.get('phi_factor', 8.0)   # contour angle phi=arctan(omega_grid/phi_factor)
            self.t_arr, self.feat_phi, self.feat_W = self._build_feat_res_expsum(self.omega, kappa, Kmin, Kmax, kpow_shift=kpow_shift, tol=tol, grid_omega=grid_omega, phi_factor=phi_factor, verb=True)
            self.N_t = len(self.t_arr)
            # Stash the exp-sum build config so (omega,kappa) can be re-set on the SAME node grid later
            # (see set_feat_params / the *_feat_batch drivers). Because grid_omega/phi_factor/Kmin/Kmax/tol
            # are fixed here, re-building for a different omega returns IDENTICAL nodes (only W changes) --
            # which is exactly what lets one node grid (hence one set of legs + filtered maps) serve a whole
            # (omega,kappa) scan.
            self._feat_build_cfg = dict(Kmin=Kmin, Kmax=Kmax, kpow_shift=kpow_shift, tol=tol,
                                        grid_omega=grid_omega, phi_factor=phi_factor)
            # fnl_sum weight per node = W_j (compression amplitude). NO Mellin power is applied anywhere:
            # F(K) is represented directly by sum_j W_j e^{-K u_j}; cosh & Gamma(iw) are folded into W_j.
            self.u_weights = np.asarray(self.feat_W, dtype=np.complex128, order='C')
            # Bracket recipe {sorted kpow-triple: complex coeff} of fnl_sum calls for the chosen scheme.
            self.feat_recipe = self._feat_bracket_recipe(self.omega, self.feat_scheme)
        
        if 'binned' in templates:
            # Check inputs
            assert self.k_bin_edges is not None, "Must specify k-bin-edges to use the binned estimator!"
            assert self.k_triplet_ids is not None, "Must specify k-triplet-ids to use the binned estimator!"
            assert self.k_bin_edges.min() >= self.k_arr.min(), "Must supply transfer function down to minimum bin edge"
            assert self.k_bin_edges.max() <= self.k_arr.max(), "Must supply transfer function up to maximum bin edge"
            assert len(self.k_triplet_ids.shape)==2, "k-triplet-ids must be a 2D array of {bin1, bin2, bin3, bin-id}"
            assert (self.k_bin_edges == np.sort(self.k_bin_edges)).all(), "k-bin-edges must be ordered"
            assert len(self.k_bin_edges) == len(np.unique(self.k_bin_edges)), "k-bin-edges must not contain duplicate values"
            assert len(np.unique(self.k_triplet_ids[:,:3]))==len(self.k_bin_edges)-1, "k-triplet-ids must contain all bins in k-bin-edges"
            assert (self.k_triplet_ids[:,0]<=self.k_triplet_ids[:,1]).all() and (self.k_triplet_ids[:,1]<=self.k_triplet_ids[:,2]).all(), "k-triplet-ids should be ordered"
            self.k_triplet_ids = np.asarray(self.k_triplet_ids)
            self.to_compute.append(['f_bin'])
            self.ints_1d = True
            self.nbin1d = len(self.k_bin_edges)-1
            self.k_bin_means = np.sqrt(self.k_bin_edges[1:]*self.k_bin_edges[:-1])
            
            # Define the binning and degeneracy
            self.nbin3d = len(self.k_triplet_ids)
            self.bin_degeneracy = np.zeros(len(self.k_triplet_ids))
            self.unique_bin_ids = np.unique(self.k_triplet_ids[:,3])
            self.n_binned = len(self.unique_bin_ids)
            for i in range(self.nbin3d):
                ki, kj, kk = self.k_bin_means[self.k_triplet_ids[i,:3]]
                ibin, jbin, kbin = self.k_triplet_ids[i,:3]

                # Check triangle conditions
                if not (ki>=np.abs(kj-kk)-1e-8 and ki<=kj+kk+1e-8):
                    raise Exception("Configuration %d with indices (%d,%d,%d) doesn't satisfy the triangle condition!"%(i,ibin,jbin,kbin))
                    
                if ibin==jbin and jbin==kbin:
                    self.bin_degeneracy[i] = 6
                elif ibin==jbin:
                    self.bin_degeneracy[i] = 2
                elif jbin==kbin:
                    self.bin_degeneracy[i] = 2
                else:
                    self.bin_degeneracy[i] = 1
            print("Using %d Fourier-space bins and %d unique shape bins"%(self.nbin3d,self.n_binned))
        else:
            self.n_binned = 0
    
        if np.any(['neural' in t for t in templates]):
            # Check inputs
            self.neural_inds = neural_inds
            assert self.neural_inputs is not None, "Must supply neural network inputs!"
            assert len(self.neural_inputs)==len(self.neural_inds), "Must supply one set of neural network inputs per template"
            for i in range(len(self.neural_inds)):
                n = self.neural_inds[i]
                assert len(self.neural_inputs[i]) in [3, 4], "Neural-%d network inputs must be of the form (weights, alpha, beta) or (weights, alpha, beta, gamma)"%n
            neural_weights = {self.neural_inds[i]: self.neural_inputs[i][0] for i in range(len(self.neural_inds))}
            for n in self.neural_inds:
                if len(neural_weights[n].shape)==2:
                    if neural_weights[n].shape[1]==1:
                        neural_weights[n] = neural_weights[n].ravel()
                    else:
                        raise Exception("Unknown neural-%d weight shape %s supplied!"%(n,neural_weights[n].shape))
            self.neural_weights = {n: np.asarray(neural_weights[n],dtype=np.float64,order='C') for n in self.neural_inds}
            self.neural_terms = {n: len(self.neural_weights[n]) for n in self.neural_inds}
            self.neural_cyclic = {}
            for i in range(len(self.neural_inds)):
                n = self.neural_inds[i]
                if len(self.neural_inputs[i]) == 3:
                    self.neural_cyclic[n] = True
                    print("Neural-%d: using a cyclic network input with %d terms"%(n,self.neural_terms[n]))
                else:
                    self.neural_cyclic[n] = False
                    print("Neural-%d: Using an input with %d terms"%(n,self.neural_terms[n]))
            self.ints_1d = True
       
        if 'isw-lensing' in templates:
            # Check inputs
            assert len(C_Tphi)>0, "Must supply temperature-lensing cross spectrum!"
            assert len(C_Tphi)>=self.lmax+1, "Must specify C^T-phi(l) up to at least lmax."
            if self.pol:
                assert len(C_Ephi)>0, "Must supply polarization-lensing cross spectrum!"
                assert len(C_Ephi)>=self.lmax+1, "Must specify C^E-phi(l) up to at least lmax."
            if not self.pol:
                assert 'TT' in C_lens_weight.keys(), "Must specify lensed TT power spectrum!"
                assert len(C_lens_weight['TT'])>=self.lmax+1, "Must specify C_lens_weight['TT'](l) up to at least lmax."
            else:
                assert 'TE' in C_lens_weight.keys(), "Must specify lensed TE power spectrum!"
                assert 'EE' in C_lens_weight.keys(), "Must specify lensed EE power spectrum!"
                assert 'BB' in C_lens_weight.keys(), "Must specify lensed BB power spectrum!"
                for k in C_lens_weight.keys():
                    assert len(C_lens_weight[k])>=self.lmax+1, "Must specify C_lens_weight(l) up to at least lmax."
                    
            # Reshape and store
            self.C_Tphi = C_Tphi[:self.lmax+1]
            if self.pol: self.C_Ephi = C_Ephi[:self.lmax+1]
            self.C_lens_weight = {k: C_lens_weight[k][:self.lmax+1] for k in C_lens_weight.keys()}
            self.to_compute.append(['u','v','v-isw'])
        
        # Identify unique components 
        if len(self.to_compute)>0:
            self.to_compute = np.unique(np.concatenate(self.to_compute))
        
        # Create filtering for minimum ls
        self.lminfilt = self.base.l_arr[self.base.l_arr<=self.lmax]>=self.lmin

        # Define array sizes
        if self.n_binned==0:
            self.total_size = len(self.templates)
        else:
            self.total_size = len(self.templates)-1+self.n_binned
        
    def report_timings(self):
        """Report timings for various steps of the computation."""
        print("\n## Timings ##\n")
        
        print("Precomputation: %.2fs"%self.timers['precomputation'])
        if self.timers['numerator']!=0:
            print("Numerator: %.2fs"%self.timers['numerator'])
        if self.timers['fisher']!=0:
            print("Fisher: %.2fs"%self.timers['fisher'])
        if self.timers['optimization']!=0:
            print("Optimization: %.2fs"%self.timers['optimization'])
        
        print("\n# Timing Breakdown")
        if self.timers['Sinv']!=0:
            print("S^-1 filtering: %.2fs"%self.timers['Sinv'])
        if self.timers['map_transforms']!=0:
            print("1-field transforms: %.2fs"%self.timers['map_transforms'])
        if self.timers['numerator']!=0:
            if np.any([('fNL' in t) for t in self.templates]):
                print("fNL -- 3-field summation: %.2fs"%self.timers['fNL_summation'])
            if 'isw-lensing' in self.templates:
                print("Lensing -- 3-field summation: %.2fs"%self.timers['lensing_summation'])
        if (self.timers['fisher']!=0 or self.timers['optimization']!=0):
            if self.timers['analytic_fisher']!=0:
                print("Analytic Fisher Matrices: %.2fs"%self.timers['analytic_fisher'])
            if self.timers['fish_grfs']!=0:
                print("Fisher -- creating GRFs: %.2fs"%self.timers['fish_grfs'])
            if self.timers['Ainv']!=0:
                print("Fisher -- A^-1 filtering: %.2fs"%self.timers['Ainv'])
            print("Fisher -- 3-field derivatives: %.2fs"%self.timers['fish_deriv'])
            print("Outer product: %.2fs"%self.timers['fish_outer'])  
            
        print("\n## Harmonic Transforms ##")
        print("Forward: %d"%self.base.n_SHTs_forward)
        print("Reverse: %d"%self.base.n_SHTs_reverse)
        print("Time: %.2fs"%self.base.time_sht)
        print("\n")
        
    def reset_timings(self):
        """Reset all the timers to zero."""
        for f in self.timers.keys():
            self.timers[f] = 0.
        self.base.n_SHTs_forward = 0
        self.base.n_SHTs_reverse = 0
        self.base.time_sht = 0.
    
    def _timer_func(counter):
        """Decorator to compute the executation time of a function and add it to a counter."""
        def _timer_func_int(func): 
            def wrap_func(self,*args, **kwargs): 
                t1 = time.time() 
                result = func(self,*args, **kwargs)
                t2 = time.time() 
                self.timers[counter] += t2-t1
                return result 
            return wrap_func 
        return _timer_func_int
        
    @_timer_func('precomputation')
    def _prepare_templates(self, ints_1d=True):
        """Compute necessary k-integrals over the transfer functions for template estimation.

        This fills arrays such as flXs_m1 and flXs_p2 arrays. Note that values outside the desired ell & field range will be set to zero.
        """
        # Print dimensions of k and r
        print("N_k: %d"%len(self.k_arr))
        if ints_1d: print("N_r: %d"%self.N_r)
        
        # Clear saved quantities, if necessary
        if hasattr(self, 't0_num'): delattr(self, 't0_num')
        if ints_1d:
            if hasattr(self, 'flXs_m2'): delattr(self, 'flXs_m2')
            if hasattr(self, 'flXs_m1'): delattr(self, 'flXs_m1')
            if hasattr(self, 'flXs_p0'): delattr(self, 'flXs_p0')
            if hasattr(self, 'flXs_p1'): delattr(self, 'flXs_p1')
            if hasattr(self, 'flXs_p2'): delattr(self, 'flXs_p2')
            if hasattr(self, 'flXs_p3'): delattr(self, 'flXs_p3')
            if hasattr(self, 'flXs_p4'): delattr(self, 'flXs_p4')
            if hasattr(self, 'flXs_osc'): delattr(self, 'flXs_osc')
            if hasattr(self, 'flXs_exp'): delattr(self, 'flXs_exp')
            if hasattr(self, 'flXs_bin'): delattr(self, 'flXs_bin')
            if hasattr(self, 'alpha_lXs'): delattr(self, 'alpha_lXs')
            if hasattr(self, 'beta_lXs'): delattr(self, 'beta_lXs')
            if hasattr(self, 'gamma_lXs'): delattr(self, 'gamma_lXs')
            
        # Precompute all spherical Bessel functions on a regular grid
        print("Precomputing Bessel functions")
        max_kr = max(self.k_arr)*max(self.r_arr)
        
        x_arr = list(np.arange(0,self.lmax*2,0.01))+list(np.arange(self.lmax*2,max_kr,0.1))
        x_arr = np.asarray(x_arr,dtype=np.float64)
        
        # Compute Bessel function in range of interest in Cython
        jlxs = np.zeros((self.lmax-self.lmin+1,len(x_arr)),dtype=np.float64,order='C')
        compute_bessel(x_arr,self.lmin,self.lmax,jlxs,self.base.nthreads)
        if np.isnan(jlxs).any(): raise Exception("Spherical Bessel calculation failed!")
        
        # Interpolate to the values of interest
        print("Interpolating Bessel functions")
        if ints_1d:
            jlkr = interpolate_jlkr(x_arr, self.k_arr, self.r_arr, jlxs, self.base.nthreads)
        
        # Set up arrays
        Pzeta_arr = self.Pzeta(self.k_arr)
        
        if 'f_m2' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,-2) integrals")
            self.flXs_m2 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, (2.+2.)/3., self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_m2)
            
        if 'f_m1' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,-1) integrals")
            self.flXs_m1 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral(self.k_arr, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_m1)
            
        if 'f_p0' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,0) integrals")
            self.flXs_p0 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, (2.-0.)/3., self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_p0)
        
        if 'f_p1' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,+1) integrals")
            self.flXs_p1 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, (2.-1.)/3, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_p1)
            
        if 'f_p2' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,+2) integrals")
            self.flXs_p2 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            q_integral(self.k_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_p2)
            
        if 'f_p3' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,+3) integrals")
            self.flXs_p3 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, (2.-3.)/3, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_p3)
            
        if 'f_p4' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^X(r,+4) integrals")
            self.flXs_p4 = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, (2.-4.)/3, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_p4)

        if 'f_exp' in self.to_compute and ints_1d:

            print("Computing exponential f_l^X(r,u) integrals (complex rotated-contour nodes)")
            # Legs are pure k^{kpow} power laws (ns=1 slope, no As); the full As^2 amplitude dependence is
            # instead applied explicitly via the (2 pi^2 As)^2 prefactor in Bl_numerator, so every term
            # (regardless of which k-powers it mixes) carries exactly As^2.
            # The decay e^{-k u_j} is COMPLEX (u_j on the rotated contour); real/imag parts are stored as
            # separate real arrays (1j applied only at the numerator, per the Re/Im expansion).
            self.flXs_exp_re = {}
            self.flXs_exp_im = {}
            # complex node u = t*e^{i phi}: real/imag decay rates fed to the complex leg integral
            u_re = np.ascontiguousarray(self.t_arr*np.cos(self.feat_phi))
            u_im = np.ascontiguousarray(self.t_arr*np.sin(self.feat_phi))
            for kpow in self.feat_kpows:
                self.flXs_exp_re[kpow] = np.zeros((self.lmax+1,1+2*self.pol,self.N_r,self.N_t),dtype=np.float64,order='C')
                self.flXs_exp_im[kpow] = np.zeros((self.lmax+1,1+2*self.pol,self.N_r,self.N_t),dtype=np.float64,order='C')
                q_integral_exp_complex(self.k_arr, float(kpow), u_re, u_im, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_exp_re[kpow], self.flXs_exp_im[kpow])
        
        if 'f_bin' in self.to_compute and ints_1d:
            
            # Compute integrals in Cython
            print("Computing f_l^{X,a}(r) integrals")
            self.flXs_bin = np.zeros((self.lmax+1,1+2*self.pol,self.N_r,self.nbin1d),dtype=np.float64,order='C')
            p_integral_bin(self.k_arr, Pzeta_arr, np.asarray(self.k_bin_edges, dtype=np.float64, order='C'), self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_bin)
        
        if self.neural_inputs is not None:
            
            # Compute neural integrals in Cython
            self.alpha_lXs, self.beta_lXs, self.gamma_lXs = {},{},{}
            for j,n in enumerate(self.neural_inds):
                self.alpha_lXs[n] = np.zeros((self.neural_terms[n],self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
                self.beta_lXs[n]  = np.zeros((self.neural_terms[n],self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
                if not self.neural_cyclic[n]:
                    self.gamma_lXs[n] = np.zeros((self.neural_terms[n],self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
                print("Computing f_l^X[alpha_%d,beta_%d,gamma_%d](r) integrals"%(n,n,n))
                for i in range(self.neural_terms[n]):
                    alphas = np.asarray(np.ravel([self.neural_inputs[j][1](np.float32(kk),i) for kk in self.k_arr]), dtype=np.float64)
                    betas = np.asarray(np.ravel([self.neural_inputs[j][2](np.float32(kk),i) for kk in self.k_arr]), dtype=np.float64)
                    f_integral(self.k_arr, alphas, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.alpha_lXs[n][i])
                    f_integral(self.k_arr, betas, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.beta_lXs[n][i])
                    if not self.neural_cyclic[n]:
                        gammas = np.asarray(np.ravel([self.neural_inputs[j][3](np.float32(kk),i) for kk in self.k_arr]), dtype=np.float64)
                        f_integral(self.k_arr, gammas, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.gamma_lXs[n][i])
            
        if ints_1d: del jlkr
        
        # Define Cython utility class
        self.utils = fNL_utils(self.base.nthreads, self.N_r, self.base.l_arr.astype(np.int32),self.base.m_arr.astype(np.int32),
                                self.ls.astype(np.int32), self.ms.astype(np.int32))
        
        print("Precomputation complete")
        
    ### MAP TRANSFORMATIONS
    @_timer_func('map_transforms')
    def _compute_weighted_maps(self, h_lm_filt, flX_arr, spin=0):
        """
        Compute [Sum_lm {}_sY_lm(n) f_l^X(i) h_lm^X] maps for each sampling point i, given the relevant weightings. 
        These are used in the bispectrum numerators and Fisher matrices.
        """
        if not (hasattr(self, 'r_arr') or hasattr(self, 'rtau_arr')):
            raise Exception("Radial arrays have not been computed!")
        
        # Sum over polarizations (only filling non-zero elements)
        # Note; we use Fortran-indexing for efficient memory access
        summ = np.zeros((flX_arr.shape[2], len(self.lminfilt)), order='F', dtype=np.complex128)
        summ[:, self.lminfilt] = self.utils.apply_fl_weights(flX_arr, h_lm_filt, 1.)
        
        # Compute SHTs
        if spin != 0:
            return self.base.to_map_vec(summ, output_spin=spin, lmax=self.lmax)[0]
        else:
            return self.base.to_map_vec(summ, output_spin=spin, lmax=self.lmax)
        
    @_timer_func('map_transforms')
    def _compute_weighted_map_single(self, h_lm_filt, flX_arr, radial_index, spin=0):
        """
        Compute [Sum_lm {}_sY_lm(n) f_l^X(i) h_lm^X] maps for a single sampling point i, given the relevant weightings. These are used in the bispectrum numerators and Fisher matrices.
        """
        if not (hasattr(self,'r_arr') or hasattr(self,'rtau_arr')):
            raise Exception("Radial arrays have not been computed!")
        
        # Sum over polarizations (only filling non-zero elements)
        summ = np.zeros((1,len(self.lminfilt)),order='C',dtype=np.complex128)
        summ[0,self.lminfilt] = self.utils.apply_fl_weight_single(flX_arr, h_lm_filt, radial_index, 1.)
        return self.base.to_map(summ, lmax=self.lmax)

    @_timer_func('map_transforms')
    def _compute_lensing_U_map(self, h_lm_filt):
        """
        Compute lensing U map from a given data vector. These are used in the ISW-lensing bispectrum numerators.
        
        The U^T map is also used in the point-source estimator. If "isw-lensing" is not in self.to_compute, we only compute U^T.
        
        We return [U^T, U^E, U^B].
        """
        
        # Output array
        U = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
        
        # Compute X = T piece for point-source + lensing estimation
        inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
        inp_lm[self.lminfilt] = h_lm_filt[0]
        U[0] = self.base.to_map(inp_lm[None],lmax=self.lmax)[0]
        
        # Compute X = E, B if implementing lensing estimators
        if 'isw-lensing' in self.templates:
            
            if self.pol:
            
                # Compute X = E piece
                inp_lm[self.lminfilt] = h_lm_filt[1]
                U[1] = self.base.to_map_spin(inp_lm, inp_lm, spin=2, lmax=self.lmax)[0]
                
                # Compute X = B piece
                inp_lm[self.lminfilt] = h_lm_filt[2]
                U[2] = self.base.to_map_spin(inp_lm, inp_lm, spin=2, lmax=self.lmax)[0]
                
        # Return output
        return U

    @_timer_func('map_transforms')
    def _compute_lensing_V_map(self, h_lm_filt):
        """
        Compute the lensing V^{lens,X}_{lambda} maps from a given data vector. These are used in the ISW-lensing bispectrum numerators.
        """
        
        # Output array
        V = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
            
        if not self.pol:
            # Apply C_l^{TT} filtering and compute V^T
            pref = np.sqrt(self.ls*(self.ls+1.))*self.C_lens_weight['TT'][self.ls]
            inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm[self.lminfilt] = pref*h_lm_filt[0]
            V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
            del pref, inp_lm
    
        else:
            # Output array
            V = np.zeros((1+2*self.pol,self.base.Npix),dtype=np.complex128,order='C')
            
            # Spin-0, X = T
            pref = np.sqrt(self.ls*(self.ls+1.))
            wienerT = (self.C_lens_weight['TT'][self.ls]*h_lm_filt[0]+self.C_lens_weight['TE'][self.ls]*h_lm_filt[1])
            inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm[self.lminfilt] = pref*wienerT
            V[0] = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
            del inp_lm
            
            # Spin-2
            pref_p = np.sqrt((self.ls+2.)*(self.ls-1.))
            pref_m = np.sqrt((self.ls-2.)*(self.ls+3.))
            wienerE = (self.C_lens_weight['TE'][self.ls]*h_lm_filt[0]+self.C_lens_weight['EE'][self.ls]*h_lm_filt[1])
            wienerB = self.C_lens_weight['BB'][self.ls]*h_lm_filt[2]
            inp_lm_re = np.zeros(len(self.lminfilt),dtype=np.complex128)
            inp_lm_im = np.zeros(len(self.lminfilt),dtype=np.complex128)
            
            # X = E,B(+)
            inp_lm_re[self.lminfilt] = pref_p*wienerE
            inp_lm_im[self.lminfilt] = 1.0j*pref_p*wienerB
            V[1] = self.base.to_map_spin(inp_lm_re+inp_lm_im,-inp_lm_re+inp_lm_im,spin=1,lmax=self.lmax)[0] # (h^R_lm + i h^I_lm)(+1)Y_lm
            
            # X = E,B(-)
            inp_lm_re[self.lminfilt] = pref_m*wienerE
            inp_lm_im[self.lminfilt] = 1.0j*pref_m*wienerB
            V[2] = self.base.to_map_spin(inp_lm_re+inp_lm_im,-inp_lm_re+inp_lm_im,spin=3,lmax=self.lmax)[0] # (h^R_lm + i h^I_lm)(+3)Y_lm
            
            del pref_p, pref_m, inp_lm_re, inp_lm_im
            
        # Return output
        return V

    @_timer_func('map_transforms')
    def _compute_isw_V_map(self, h_lm_filt):
        """
        Compute the ISW-lensing V^{ISW}_{+1} map from a given data vector. This is used in the ISW-lensing bispectrum numerators.
        """
        
        # Output array
        V = np.zeros(self.base.Npix,dtype=np.complex128,order='C')
        
        # Apply C_l^{Xphi} filtering, summing over X = T,E
        inp_lm = np.zeros(len(self.lminfilt),dtype=np.complex128)
        if not self.pol:
            inp_lm[self.lminfilt] = np.sqrt(self.ls*(self.ls+1.))*self.C_Tphi[self.ls]*h_lm_filt[0]
        else:
            inp_lm[self.lminfilt] = np.sqrt(self.ls*(self.ls+1.))*(self.C_Tphi[self.ls]*h_lm_filt[0]+self.C_Ephi[self.ls]*h_lm_filt[1])

        # Compute V_{+} map    
        V = self.base.to_map_spin(-inp_lm,inp_lm,spin=1,lmax=self.lmax)[1] # h_lm (-1)Y_lm
        
        # Return output
        return V

    def _filter_pair(self, input_maps, filtering = 'F_m1', radial_index=None):
        """Compute the processed field with a given filtering for a pair of input maps."""
        
        if   filtering=='F_m2':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_m2, radial_index) for imap in input_maps],order='C')     
        
        elif filtering=='F_m1':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_m1, radial_index) for imap in input_maps],order='C')     
        
        elif filtering=='F_p0':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_p0, radial_index) for imap in input_maps],order='C')     
        
        elif filtering=='F_p1':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_p1, radial_index) for imap in input_maps],order='C')     
        
        elif filtering=='F_p2':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_p2, radial_index) for imap in input_maps],order='C')     
        
        elif filtering=='F_p3':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_p3, radial_index) for imap in input_maps],order='C')     

        elif filtering=='F_p4':
            return np.asarray([self._compute_weighted_map_single(imap, self.flXs_p4, radial_index) for imap in input_maps],order='C')    

        elif filtering=='F_exp':
            # Complex leg maps (real/imag legs filtered separately, recombined with 1j).
            out = {}
            for kpow in self.feat_kpows:
                fr = np.asarray(self.flXs_exp_re[kpow][:,:,radial_index,:], order='C')
                fi = np.asarray(self.flXs_exp_im[kpow][:,:,radial_index,:], order='C')
                maps = [ (self._compute_weighted_maps(imap, fr)[:,None,:]
                          + 1j*self._compute_weighted_maps(imap, fi)[:,None,:]) for imap in input_maps ]
                out[kpow] = np.asarray(maps, order='C')
            return out

        elif filtering=='F_bin':
            return np.asarray([self._compute_weighted_maps(imap, np.asarray(self.flXs_bin[:,:,radial_index,:], order='C')) for imap in input_maps],order='C')
        
        elif filtering=='U':
            return np.asarray([self._compute_lensing_U_map(imap) for imap in input_maps], order='C')        
            
        elif filtering=='V':
            return np.asarray([self._compute_lensing_V_map(imap) for imap in input_maps], order='C')        
        
        elif filtering=='V-ISW':
            return np.asarray([self._compute_isw_V_map(imap) for imap in input_maps], order='C')        
        
        elif 'neural-alpha' in filtering:
            n = int(filtering.split('-')[2])
            return np.asarray([[self._compute_weighted_map_single(imap, self.alpha_lXs[n][i], radial_index) for imap in input_maps] for i in range(self.neural_terms[n])], order='C')
        
        elif 'neural-beta' in filtering:
            n = int(filtering.split('-')[2])
            return np.asarray([[self._compute_weighted_map_single(imap, self.beta_lXs[n][i], radial_index) for imap in input_maps] for i in range(self.neural_terms[n])], order='C')

        elif 'neural-gamma' in filtering:
            n = int(filtering.split('-')[2])
            return np.asarray([[self._compute_weighted_map_single(imap, self.gamma_lXs[n][i], radial_index) for imap in input_maps] for i in range(self.neural_terms[n])], order='C')

        else:
            raise Exception("Filtering %s is not implemented!"%filtering)
    
    def _apply_all_filters(self, input_map):
        """Compute the processed fields with all relevant filterings for a single input map."""
        
        # Output array
        output = {}
        
        # Compute local maps
        if 'f_m1' in self.to_compute:
            output['f_m1'] = self._compute_weighted_maps(input_map, self.flXs_m1)
              
        if 'f_p2' in self.to_compute:
            output['f_p2'] = self._compute_weighted_maps(input_map, self.flXs_p2)
        
        # Compute equilateral maps
        if 'f_p1' in self.to_compute:
            output['f_p1'] = self._compute_weighted_maps(input_map, self.flXs_p1)
            
        if 'f_p0' in self.to_compute:
            output['f_p0'] = self._compute_weighted_maps(input_map, self.flXs_p0) 

        # Compute other maps
        if 'f_m2' in self.to_compute:
            output['f_m2'] = self._compute_weighted_maps(input_map, self.flXs_m2)

        if 'f_p3' in self.to_compute:
            output['f_p3'] = self._compute_weighted_maps(input_map, self.flXs_p3)

        if 'f_p4' in self.to_compute:
            output['f_p4'] = self._compute_weighted_maps(input_map, self.flXs_p4)

        # Compute feature maps
        if 'f_osc' in self.to_compute:
            output['f_osc'] = self._compute_weighted_maps(input_map, self.flXs_osc)
            
        if 'f_exp' in self.to_compute:
            # Complex leg maps: since _compute_weighted_maps is LINEAR in flX, the real/imag map parts are
            # obtained by filtering with the real/imag leg arrays separately (1j applied at the numerator).
            output['f_exp_re'] = {}
            output['f_exp_im'] = {}
            for kpow in self.feat_kpows:
                fr = np.asarray(self.flXs_exp_re[kpow].reshape(self.flXs_exp_re[kpow].shape[0], self.flXs_exp_re[kpow].shape[1], -1), order='C')
                fi = np.asarray(self.flXs_exp_im[kpow].reshape(self.flXs_exp_im[kpow].shape[0], self.flXs_exp_im[kpow].shape[1], -1), order='C')
                output['f_exp_re'][kpow] = self._compute_weighted_maps(input_map, fr).reshape(self.N_r, self.N_t, self.base.Npix)
                output['f_exp_im'][kpow] = self._compute_weighted_maps(input_map, fi).reshape(self.N_r, self.N_t, self.base.Npix)
            
        # Compute binned maps
        if 'f_bin' in self.to_compute:
            # Flatten all bins to one dimension
            flXs_flat = self.flXs_bin.transpose(0, 1, 3, 2).reshape(self.flXs_bin.shape[0], self.flXs_bin.shape[1], -1)
            # Assemble output and reshape
            output['f_bin'] = self._compute_weighted_maps(input_map, flXs_flat).reshape(self.nbin1d, self.N_r, self.base.Npix)
             
        # Compute lensing maps
        if 'u' in self.to_compute:
            output['u'] = self._compute_lensing_U_map(input_map)        
            
        if 'v' in self.to_compute:
            output['v'] = self._compute_lensing_V_map(input_map)
            
        if 'v-isw' in self.to_compute:
            output['v-isw'] = self._compute_isw_V_map(input_map)
            
        if self.neural_inputs is not None:
            for n in self.neural_inds:
                output['neural-alpha-%d'%n] = np.zeros((self.neural_terms[n], len(self.alpha_lXs[n][0,0,0]),self.base.Npix),order='C',dtype=np.float64)
                output['neural-beta-%d'%n] = np.zeros((self.neural_terms[n], len(self.beta_lXs[n][0,0,0]),self.base.Npix),order='C',dtype=np.float64)
                if not self.neural_cyclic[n]:
                    output['neural-gamma-%d'%n] = np.zeros((self.neural_terms[n],len(self.gamma_lXs[n][0,0,0]),self.base.Npix),order='C',dtype=np.float64)
                for i in range(self.neural_terms[n]):
                    output['neural-alpha-%d'%n][i] = self._compute_weighted_maps(input_map, self.alpha_lXs[n][i])
                    output['neural-beta-%d'%n][i] = self._compute_weighted_maps(input_map, self.beta_lXs[n][i])
                    if not self.neural_cyclic[n]:
                        output['neural-gamma-%d'%n][i] = self._compute_weighted_maps(input_map, self.gamma_lXs[n][i])
            
        return output
    
    ### SIMULATION FUNCTIONS
    def _process_sim(self, sim, input_type='map'):
        """
        Process a single input simulation. This is used for the linear term of the bispectrum estimator.
        
        We return a set of weighted maps for this simulation (filtered by e.g. p_l^X).
        """
        # Transform to Fourier space and normalize appropriately
        if input_type=='Sinv_map':
            assert sim.shape[1]==self.lminfilt.sum(), "S^-1.sim has the wrong shape!"
            h_sim_lm = sim.copy()
        else:
            t_init = time.time()
            h_sim_lm = np.asarray(self.applySinv(sim, input_type=input_type, lmax=self.lmax)[:,self.lminfilt],order='C')
            self.timers['Sinv'] += time.time()-t_init
        
        # Compute processed maps
        proc_maps = self._apply_all_filters(h_sim_lm)
        return proc_maps

    def load_sims(self, load_sim, N_sims, verb=False, preload=True, input_type='map'):
        """
        Load in and preprocess N_sim Monte Carlo simulations used in the linear term of the bispectrum estimator.

        The input is a function which loads the simulation in map- or harmonic-space given an index (0 to N_sims-1).

        If preload=False, the simulation products will not be stored in memory, but instead accessed when necessary. This greatly reduces memory usage, but is less CPU efficient if many datasets are analyzed together.
        
        These can alternatively be generated with a fiducial spectrum using the generate_sims script.
        """
        
        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)

        if preload:
            self.preload = True

            # Check we have initialized correctly
            if self.ints_1d and (not hasattr(self,'r_arr')):
                raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
            
            #  Define list of maps
            self.proc_maps = []

            # Iterate over simulations and preprocess appropriately    
            for ii in range(self.N_it):
                if ii%5==0 and verb: print("Processing bias simulation %d of %d"%(ii+1,self.N_it))

                # Load and process simulation
                self.proc_maps.append(self._process_sim(load_sim(ii), input_type=input_type))
        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            self.load_sim_data = lambda ii: self._process_sim(load_sim(ii), input_type=input_type)
            
    def generate_sims(self, N_sims, Cl_input=[], preload=True, verb=False):
        """
        Generate Monte Carlo simulations used in the linear term of the bispectrum generator. 
        These are pure GRFs. By default, they are generated with the input survey mask.
        
        If preload=True, we create N_sims simulations and store the relevant transformations into memory.
        If preload=False, we store only the function used to generate the sims, which will be processed later. This is cheaper on memory, but less CPU efficient if many datasets are analyzed together.
        
        We can alternatively load custom simulations using the load_sims script.
        """

        self.N_it = N_sims
        print("Using %d Monte Carlo simulations"%self.N_it)
        
        # Define input power spectrum (with noise)
        if len(Cl_input)==0:
            Cl_input = self.base.Cl_tot

        if preload:
            self.preload = True

            # Check we have initialized correctly
            if self.ints_1d and (not hasattr(self,'r_arr')):
                raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
            
            # Define lists of maps
            self.proc_maps = []
            
            # Iterate over simulations
            for ii in range(self.N_it):
                if ii%5==0 and verb: print("Generating bias simulation %d of %d"%(ii+1,self.N_it))
                
                # Generate simulation and compute P, Q maps
                if self.ones_mask:
                    sim_lm = self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic', lmax=self.lmax, deconvolve_beam=False)
                    self.proc_maps.append(self._process_sim(sim_lm, input_type='harmonic'))
                else:
                    sim = self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False)
                    self.proc_maps.append(self._process_sim(sim))

        else:
            self.preload = False
            if verb: print("No preloading; simulations will be loaded and accessed at runtime.")
            
            # Simply save iterator and continue (simulations will be processed in serial later) 
            if self.ones_mask:
                self.load_sim_data = lambda ii: self._process_sim(self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, output_type='harmonic', lmax=self.lmax, deconvolve_beam=False), input_type='harmonic')
            else:
                self.load_sim_data = lambda ii: self._process_sim(self.mask*self.base.generate_data(int(1e5)+ii, Cl_input=Cl_input, deconvolve_beam=False))
    
    ### FISHER MATRIX FUNCTIONS
    @_timer_func('fish_outer')
    def _assemble_fish(self, Q3_a, Q3_b, sym=False):
        """Compute Fisher matrix between two Q arrays as an outer product. This is parallelized across the l,m axis."""
        return outer_product_bspec(Q3_a, Q3_b, self.base.nthreads, sym)

    def _weight_Q_maps(self, tmp_Q, weighting='Ainv'):
        """Apply inplace weighting to a Q map to form output array. This includes factors of S^-1.P if necessary."""
        
        for index in range(self.total_size):
            if weighting=='Ainv':
                # Construct l-space map down to l=0
                full_Q = np.zeros((1+2*self.pol,len(self.lminfilt)),dtype=np.complex128)
                full_Q[:,self.lminfilt] = self.beam_lm*tmp_Q[index]
                # Compute S^-1.P.Q
                t_init = time.time()
                if self.ones_mask:
                    tmp_Q[index] = self.applySinv(full_Q,input_type='harmonic', lmax=self.lmax)[:,self.lminfilt]
                else:
                    tmp_Q[index] = self.applySinv(self.mask*self.base.to_map(full_Q,lmax=self.lmax), lmax=self.lmax)[:,self.lminfilt]
                self.timers['Sinv'] += time.time()-t_init
            elif weighting=='Sinv':
                tmp_Q[index] = self.m_weight*tmp_Q[index]   

    @_timer_func('fish_deriv')
    def _transform_maps(self, map12, flXs, weights, spin=0, lm_map=None):
        """Compute Sum_i w_i M_LM f^X_L(i) for real-space map M(n). We optionally average over spins."""
        if spin==0:
            if lm_map is None:
                lm_map = np.asarray(self.base.to_lm_vec(map12,lmax=self.lmax)[:,self.lminfilt],order='C')
            return self.utils.radial_sum(lm_map, weights, flXs)
        elif spin==1:
            if lm_map is None:
                lm_map = np.asarray(self.base.to_lm_vec([map12,map12.conjugate()],spin=1,lmax=self.lmax)[:,:,self.lminfilt],order='C')
            return self.utils.radial_sum_spin1(lm_map, weights, flXs)
        else:
            raise Exception(f"Wrong spin s = {spin}!")

    def _transform_maps_complex(self, inner, outer_re, outer_im, weight):
        """Complex version of _transform_maps (spin 0): compute
            Sum_i weight_i outer^X_l(i) [SHT(inner)]_lm(i)
        for a COMPLEX real-space map inner = inner_re + i*inner_im, COMPLEX per-node weight, and a COMPLEX
        outer leg outer = outer_re + i*outer_im (real/imag parts passed separately). SHTs are done on the
        real and imaginary map parts SEPARATELY (each a genuine real map) and recombined with 1j -- never a
        complex map into a real SHT. The complex (weight*outer) is folded into a real g = g_re + i*g_im and
        the radial sum split into two real radial_sum calls. Returns a complex (npol, nlm) a_lm array."""
        L = (np.asarray(self.base.to_lm_vec(np.ascontiguousarray(inner.real), lmax=self.lmax)[:,self.lminfilt], order='C')
             + 1j*np.asarray(self.base.to_lm_vec(np.ascontiguousarray(inner.imag), lmax=self.lmax)[:,self.lminfilt], order='C'))
        wr = weight.real[None,None,:]; wi = weight.imag[None,None,:]
        g_re = np.ascontiguousarray(wr*outer_re - wi*outer_im)
        g_im = np.ascontiguousarray(wr*outer_im + wi*outer_re)
        ones = np.ones(inner.shape[0], dtype=np.float64)
        return self.utils.radial_sum(L, ones, g_re) + 1j*self.utils.radial_sum(L, ones, g_im)

    def _compute_fisher_derivatives(self, templates, verb=False, input_derivatives={}):
        """Compute the derivative of the ideal Fisher matrix with respect to the weights for each template of interest."""

        # Output array
        output = {}
        
        # Compute arrays
        # NB: using exact Gauss-Legendre integration in mu
        [mus, w_mus] = p_roots(2*self.lmax+1)
        ls = np.arange(self.lmin,self.lmax+1)
        legs = np.asarray([lpmn(0,self.lmax,mus[i])[0][0,self.lmin:] for i in range(len(mus))])
            
        t_init = time.time()
        for template in templates:

            # Load from input dictionary if possible
            if template in input_derivatives.keys():
                if verb: print("\tLoading derivative %s from input dictionary!"%template)
                output[template] = input_derivatives[template]
                continue

            # Compute derivative matrices from scratch
            if template=='fNL-loc':
                if verb: print("\tComputing fNL-loc Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_loc(self.flXs_m1, self.flXs_p2, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
               
            elif template=='fNL-eq':
                if verb: print("\tComputing fNL-eq Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_eq(self.flXs_m1, self.flXs_p2, self.flXs_p1, self.flXs_p0, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
            
            elif template=='fNL-orth':
                if verb: print("\tComputing fNL-orth Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_orth(self.flXs_m1, self.flXs_p2, self.flXs_p1, self.flXs_p0, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))

            elif template=='fNL-orth2':
                if verb: print("\tComputing fNL-orth2 Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_orth2(np.asarray([self.flXs_m2, self.flXs_m1, self.flXs_p0, self.flXs_p1, self.flXs_p2, self.flXs_p3, self.flXs_p4], order='C',dtype=np.float64), self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))

            elif template=='binned':
                if verb: print("\tComputing binned Fisher matrix derivative exactly")
                flXs_bin_sum = np.asarray(self.flXs_bin.sum(axis=3), order='C', dtype=np.float64)
                deriv_matrix = np.asarray(fisher_deriv_fNL_binned(flXs_bin_sum, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'),
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))

            elif template=='fNL-feat-res':
                if verb: print("\tComputing fNL-feat-res Fisher matrix derivative exactly")
                # There is only one ideal-Fisher implementation for this template: compute_ideal_fisher_2d_feat_res,
                # which collapses (r,u) into a single pair index (see its docstring, and the cubic-numerator-term
                # comment in Bl_numerator for the genuinely-convergent Mellin-shift derivation). The (N_r, N_r)
                # matrix needed by the optimize_radial_sampling_1d r-optimization workflow below is just this
                # template's r-marginal of the full pair-pair matrix (sum over all u,u' at fixed r,r').
                deriv_matrix_pairs, r_pairs, t_pairs = self.compute_ideal_fisher_2d_feat_res()
                deriv_matrix = deriv_matrix_pairs.reshape(self.N_r, self.N_t, self.N_r, self.N_t).sum(axis=(1,3))
                # Opt-in diagnostic stash of the full (N_pairs, N_pairs) derivative matrix + pair indices
                # (set self._save_feat_pairs=True before calling). Off by default -> no memory cost in production.
                if getattr(self, '_save_feat_pairs', False):
                    self._feat_pairs_matrix = (deriv_matrix_pairs, r_pairs, t_pairs)
            
            elif 'neural' in template:
                n = int(template.split('-')[1])
                if verb: print("\tComputing neural-%d Fisher matrix derivative exactly"%n)
                if not self.neural_cyclic[n]:
                    deriv_matrix = np.asarray(fisher_deriv_neural(self.alpha_lXs[n], self.beta_lXs[n], self.gamma_lXs[n], self.neural_weights[n], self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                        legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                else:
                    deriv_matrix = np.asarray(fisher_deriv_neural_cyclic(self.alpha_lXs[n], self.beta_lXs[n], self.neural_weights[n], self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                        legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                
            else:
                raise Exception("Template %s not implemented!"%template)
                
            output[template] = np.sum(deriv_matrix), deriv_matrix
            
        self.timers['analytic_fisher'] += time.time()-t_init

        return output
    def _feat_res_fisher_deriv_complex(self, flXs_re, flXs_im, weights_pairs, W_pairs):
        """Build the (N_pairs, N_pairs) ideal-Fisher derivative matrix for fNL-feat-res (sm3/sm1), given
        the COMPLEX legs (real/imag parts) on (r,u) pairs. Shared by compute_ideal_fisher_2d_feat_res and
        _build_and_optimize_ru_grid. Uses fisher_deriv_fNL_feat_res_2d_complex (F = 2Re<Z,Z> + 2<Z,Zbar>);
        verified vs brute force (FEAT_NOTES Session 9/9b).
          flXs_re/im: dict {kpow: (lmax+1, npol, N_pairs)}; weights_pairs: (N_pairs,) r-quadrature weight;
          W_pairs: (N_pairs,) compression node-weight W_node per pair. Summing the matrix gives the ideal Fisher."""
        npol = 1+2*self.pol; Npairs = len(weights_pairs)
        # leg slots 0..4 = kpow -3,-2,-1,0,+1 (sm3 leaves slot 4 zero) -- PARAM-INDEPENDENT
        legre = np.zeros((5, self.lmax+1, npol, Npairs), dtype=np.float64)
        legim = np.zeros((5, self.lmax+1, npol, Npairs), dtype=np.float64)
        for kpow in self.feat_kpows:
            legre[kpow+3] = flXs_re[kpow]; legim[kpow+3] = flXs_im[kpow]
        # complex per-pair group weights (pref0 kappa^{iw} gamma_g/n_ord folded in) + ordered-term list
        Vgre, Vgim, term_group, term_slots, nslot = self._feat_ideal_vg(W_pairs)
        inv_Cl_mat, legs, w_mus = self._feat_ideal_mu()
        args = (np.ascontiguousarray(legre), np.ascontiguousarray(legim), np.ascontiguousarray(weights_pairs),
                np.ascontiguousarray(Vgre), np.ascontiguousarray(Vgim), term_group, term_slots, nslot,
                inv_Cl_mat, legs, np.ascontiguousarray(w_mus), self.lmin, self.lmax, self.base.nthreads)
        # The routine BLAS-projects (dgemm) inside a prange over pairs, so pin BLAS to 1 thread to avoid
        # nesting the BLAS thread pool inside the pair-parallel threads (oversubscription).
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api='blas'):
                dm = fisher_deriv_fNL_feat_res_2d_complex(*args)
        except ImportError:
            dm = fisher_deriv_fNL_feat_res_2d_complex(*args)
        return np.asarray(dm)

    def compute_ideal_fisher_2d_feat_res(self):
        """
        Compute the ideal Fisher matrix for the fNL-feat-res template with r and u collapsed into a single
        combined (r,u) 'pair' index (compressed exp-sum representation; complex rotated-contour nodes).
        Requires self.flXs_exp_re/im[kpow] (shape (lmax+1, npol, N_r, N_u)), e.g. via _prepare_templates()
        or optimize_radial_sampling_1d().

        The (r,t) pairs are the full outer product of self.r_arr and self.t_arr (pair p = ir*N_t + it).
        Summing the full matrix gives the total ideal Fisher; r- or t-marginal blocks give those variables'
        own marginal Fisher matrices (used by the optimize_radial_sampling_1d r-workflow).

        Returns: deriv_matrix (N_pairs,N_pairs); r_pairs, t_pairs (N_pairs,).
        """
        assert hasattr(self, 'flXs_exp_re'), "Must first compute flXs_exp_re/im, e.g. via _prepare_templates() or optimize_radial_sampling_1d()!"
        N_r, N_t = self.N_r, self.N_t
        npol = 1+2*self.pol
        flXs_re = {p: np.asarray(self.flXs_exp_re[p].reshape(self.lmax+1, npol, N_r*N_t), order='C') for p in self.feat_kpows}
        flXs_im = {p: np.asarray(self.flXs_exp_im[p].reshape(self.lmax+1, npol, N_r*N_t), order='C') for p in self.feat_kpows}
        weights_pairs = np.asarray(np.repeat(self.quad_weights_1d, N_t), order='C')   # r-quadrature weight per pair
        W_pairs = np.asarray(np.tile(self.feat_W, N_r))                                # compression W per pair (per node)
        r_pairs = np.repeat(self.r_arr, N_t)
        t_pairs = np.tile(self.t_arr, N_r)
        deriv_matrix = self._feat_res_fisher_deriv_complex(flXs_re, flXs_im, weights_pairs, W_pairs)
        return deriv_matrix, r_pairs, t_pairs

    def _feat_ideal_vg(self, W_pairs):
        """Build the complex per-pair group weights Vg = pref0*kappa^{iw}*W_node*gamma_g/n_ord_g and the
        ordered-term list, for the CURRENT (omega,kappa,recipe). Returns (Vgre, Vgim, term_group, term_slots,
        nslot). Only Vgre/Vgim depend on (omega,kappa); term_group/term_slots/nslot are scheme-fixed (the sm3/
        sm1 recipe KEYS are omega-independent), so a batch reuses the term structure and re-builds only Vg."""
        from itertools import permutations
        omega = self.omega; kappa = self.feat_params['kres_cs']
        C = (2*np.pi**2*self.As)**2*omega**2/(4.*(omega+1j))*kappa**(1j*omega)   # pref0 * kappa^{iw}
        Npairs = len(W_pairs)
        groups = list(self.feat_recipe.items()); ng = len(groups)
        Vgre = np.zeros((ng, Npairs), dtype=np.float64); Vgim = np.zeros((ng, Npairs), dtype=np.float64)
        term_slots = []; term_group = []
        for gi, (key, gam) in enumerate(groups):
            nord = len(set(permutations(key)))
            Vg = C*W_pairs*(gam/nord)
            Vgre[gi] = Vg.real; Vgim[gi] = Vg.imag
            for o in set(permutations(key)):
                term_slots.append([o[0]+3, o[1]+3, o[2]+3]); term_group.append(gi)
        term_slots = np.ascontiguousarray(np.array(term_slots, dtype=np.int32))
        term_group = np.ascontiguousarray(np.array(term_group, dtype=np.int32))
        nslot = len(self.feat_kpows)   # active leg slots (sm3:4, sm1:5); slot=kpow+3, kpows contiguous from -3
        return np.ascontiguousarray(Vgre), np.ascontiguousarray(Vgim), term_group, term_slots, nslot

    def _feat_ideal_mu(self):
        """Gauss-Legendre mu quadrature + inv-Cl for the ideal Fisher (param-independent). The integrand
        (product of three degree<=lmax Legendre polys) has degree <=3*lmax, so ceil((3*lmax+1)/2) nodes are
        exact. Returns (inv_Cl_mat, legs, w_mus). (n_mu is NOT reducible: halving it gave a 61% error --
        the integrand is genuinely degree ~3*lmax and undersampling aliases catastrophically.)"""
        n_mu = int(np.ceil((3*self.lmax+1)/2.))
        [mus, w_mus] = p_roots(n_mu)
        legs = np.ascontiguousarray(np.asarray([lpmn(0,self.lmax,mus[i])[0][0,self.lmin:] for i in range(len(mus))]))
        inv_Cl_mat = np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat, order='C')
        return inv_Cl_mat, legs, np.ascontiguousarray(w_mus)

    def compute_ideal_fisher_feat_batch(self, feat_params_list, verb=False):
        """BATCHED ideal Fisher for a SET of (omega,kappa) sharing the current node + radial grid.

        The expensive per-pair leg work (the zeta build + mu-projection) is (omega,kappa)-INDEPENDENT and done
        ONCE inside fisher_deriv_fNL_feat_res_2d_complex_batch; only the cheap per-param term-combination
        (weighted by Vg) is looped. Returns F: (nparam,) real ideal-Fisher scalars, F[i] matching
        np.sum(compute_ideal_fisher_2d_feat_res()) run at feat_params_list[i] to machine precision.

        feat_params_list: list of (omega, kappa) tuples (or dicts with 'omega','kres_cs'). Requires
        self.flXs_exp_re/im (legs) already built on a grid_omega that covers every omega in the list."""
        assert hasattr(self, 'flXs_exp_re'), "Must first compute flXs_exp_re/im (e.g. prepare a common r-grid)!"
        N_r, N_t = self.N_r, self.N_t
        npol = 1+2*self.pol
        # PARAM-INDEPENDENT leg arrays on (r,u) pairs (slots 0..4 = kpow -3..+1)
        Npairs = N_r*N_t
        legre = np.zeros((5, self.lmax+1, npol, Npairs), dtype=np.float64)
        legim = np.zeros((5, self.lmax+1, npol, Npairs), dtype=np.float64)
        for kp in self.feat_kpows:
            legre[kp+3] = np.asarray(self.flXs_exp_re[kp].reshape(self.lmax+1, npol, Npairs), order='C')
            legim[kp+3] = np.asarray(self.flXs_exp_im[kp].reshape(self.lmax+1, npol, Npairs), order='C')
        weights_pairs = np.asarray(np.repeat(self.quad_weights_1d, N_t), order='C')
        inv_Cl_mat, legs, w_mus = self._feat_ideal_mu()

        # per-param Vg (only piece that depends on (omega,kappa)); term structure common to all params
        params = [p if isinstance(p, (tuple, list)) else (p['omega'], p['kres_cs']) for p in feat_params_list]
        Vgre_list = []; Vgim_list = []; term_group = None; term_slots = None; nslot = None
        for (om, ka) in params:
            self.set_feat_params(om, ka, verb=verb)
            W_pairs = np.asarray(np.tile(self.feat_W, N_r))
            vgr, vgi, term_group, term_slots, nslot = self._feat_ideal_vg(W_pairs)
            Vgre_list.append(vgr); Vgim_list.append(vgi)
        Vgre = np.ascontiguousarray(np.stack(Vgre_list, axis=0))   # (nparam, ng, Npairs)
        Vgim = np.ascontiguousarray(np.stack(Vgim_list, axis=0))

        args = (np.ascontiguousarray(legre), np.ascontiguousarray(legim), np.ascontiguousarray(weights_pairs),
                Vgre, Vgim, term_group, term_slots, nslot,
                inv_Cl_mat, legs, w_mus, self.lmin, self.lmax, self.base.nthreads)
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api='blas'):
                F = fisher_deriv_fNL_feat_res_2d_complex_batch(*args)
        except ImportError:
            F = fisher_deriv_fNL_feat_res_2d_complex_batch(*args)
        return np.asarray(F)

    def _greedy_optimize_fisher_matrix(self, deriv_matrix, tolerance=1e-3, verb=False, label=''):
        """
        Generic greedy (Smith & Zaldarriaga 06) optimizer for an ideal Fisher derivative matrix. This is
        index-agnostic -- it works identically whether the matrix is indexed by r alone (as in
        optimize_radial_sampling_1d, which implements this same algorithm inline, template-by-template)
        or by (r,u) pairs (as in optimize_ru_sampling): greedily selects a minimal subset of indices
        (with optimal quadratic weights) that reproduces the full matrix's total Fisher information to
        within 'tolerance' (relative).

        Returns: inds (list of selected indices into the input array), w_opt (weights for those
        indices), init_score (total Fisher of the unoptimized matrix), fish (total Fisher after
        optimization).
        """
        N = len(deriv_matrix)
        init_score = np.sum(deriv_matrix)
        inds_init = np.arange(N)
        inds = []

        def _compute_score(w_vals, full_score=False):
            if full_score:
                return np.sum(G_mat), np.sum(np.outer(w_vals,w_vals)*deriv_matrix[inds][:,inds])
            else:
                return np.sum(G_mat)

        if len(inds)!=0:
            notinds = [i for i in np.arange(N) if i not in inds]
            inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
            G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
            w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
            score = _compute_score(w_vals)
            if verb: print("Unoptimized relative score: %.2e"%(score/init_score))
        else:
            score = init_score
            w_vals = []
            G_mat = np.eye(0)

        if score/init_score >= tolerance and (np.diag(G_mat)>0).all():
            if len(inds)==0:
                next_ind = np.argsort(np.sum(deriv_matrix,axis=1)**2/np.diag(deriv_matrix))[-1]
            else:
                next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
            inds.append(next_ind)
            score_old = score

            for iteration in range(len(inds), N):
                inds[-1] = next_ind
                notinds = [i for i in np.arange(N) if i not in inds]
                try:
                    inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                except np.linalg.LinAlgError:
                    print("Singular matrix; exiting!")
                    inds = inds[:-1]
                    break
                G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                score = _compute_score(w_vals)
                if verb: print("Iteration %d, relative score: %.2e"%(iteration, score/init_score))
                if score<0:
                    print("## Score is negative; this indicates a numerical error!")
                    break
                if score/init_score < tolerance:
                    break
                score_old = score
                next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
                inds.append(next_ind)

        if len(G_mat)==0:
            raise Exception("Failed to converge; this indicates a bug!")

        w_opt = np.asarray(w_vals.copy())
        score, fish = _compute_score(w_opt, full_score=True)
        if verb: print("Ideal %s Fisher: %.4e (initial), %.4e (optimized). Relative score: %.2e"%(label, init_score, fish, score/init_score))

        return inds, w_opt, init_score, fish

    @staticmethod
    def _feat_bracket_recipe(omega, scheme):
        """Return {sorted (kpow1,kpow2,kpow3): complex coeff} giving the fnl_sum decomposition of the
        m=n bracket S_0 (scheme='sm3') or its K-raised form K*S_0 (scheme='sm1', legs +1 on one factor).
        p_n(k)=k^{-n} -> leg kpow=-n. Ordered-perm coeffs: {-3,-3,0}:1, {-3,-2,-1}:i*w, {-2,-2,-2}:-(w^2+i*w).
        Collapsing ordered perms to a sorted multiset with summed coeff is exactly the fNL-eq/orth
        fnl_sum convention (coeff = template_coeff x #orderings). Verified vs S_0/K*S_0 to machine precision."""
        from itertools import permutations
        base = []
        for p in set(permutations((-3,-3,0))): base.append((p, 1.0+0j))
        for p in set(permutations((-3,-2,-1))): base.append((p, 1j*omega))
        base.append(((-2,-2,-2), -(omega**2+1j*omega)))
        if scheme == 'sm3':
            terms = base
        elif scheme == 'sm1':
            terms = []
            for (p,c) in base:
                for leg in range(3):
                    q = list(p); q[leg] += 1; terms.append((tuple(q), c))
        else:
            raise ValueError("integration_scheme must be 'sm3' or 'sm1', got %r"%scheme)
        recipe = {}
        for (p,c) in terms:
            key = tuple(sorted(p)); recipe[key] = recipe.get(key, 0) + c
        return recipe

    def _build_feat_res_expsum(self, omega, kappa, Kmin, Kmax, kpow_shift=0, tol=1e-6, n_train=400,
                               n_val=1500, n_base=700, n_stat=500, grid_omega=None, phi_factor=8.0, verb=False):
        """Build the compressed exponential-sum basis for the fNL-feat-res u-integral. The estimator needs
            cosh(pi w/2) Gamma(iw) K^{-iw+kpow_shift}  ~  sum_j W_j e^{-K u_j},  u_j=e^{x_j+i phi}/mu.
        kpow_shift=0 for sm3 (reconstruct with the m=n bracket S_0, u-factor F=cosh Gamma K^{-iw}); kpow_shift=-1
        for sm1 (reconstruct with the K-raised bracket K*S_0, so the u-factor is F/K=cosh Gamma K^{-iw-1} -- the
        extra K is supplied by the raised bracket). Fit the kappa-INDEPENDENT scaled target on Chebyshev-in-logK
        points using overcomplete rotated-contour candidates; select a minimal node subset by pivoted QR (ID);
        refit complex weights by lstsq; grow N_t until max validation rel.err < tol. Returns (u_nodes, W) complex.
        kappa=k_res/c_s enters only as the external phase kappa^{iw} applied at consumption (not here).

        grid_omega (default None -> omega): frequency that defines the NODE grid (contour angle phi=arctan/8,
        stationary-phase band, N_osc, and the pivoted-QR node count grown until grid_omega's own fit < tol).
        The complex WEIGHTS W are always fit for the actual target `omega` on that node set. Setting
        grid_omega>omega lets one node set (built for the largest frequency in a scan) be reused for all
        smaller omega -- the dense grid over-resolves the smoother small-w target, giving equal/better
        accuracy and much smaller |W| (verified: w=50 grid reproduces w=10 to ~1e-7 with max|W|~1)."""
        from scipy.linalg import qr, lstsq
        from scipy.special import loggamma
        wg = omega if grid_omega is None else grid_omega   # frequency defining the node grid
        # phi = contour rotation angle (numerical gauge; the exact answer is phi-independent). Default
        # arctan(wg/8); phi_factor tunes it (larger factor -> smaller angle) for the phi-invariance check.
        mu = np.sqrt(Kmin*Kmax); phi = np.arctan(wg/phi_factor); sphi = np.sin(phi)
        # overcomplete candidate nodes on the rotated contour, denser in the (grid-freq) stationary-phase band
        x_min = np.log(1e-12*mu/Kmax); x_max = np.log(32*mu/(Kmin*np.cos(phi)))
        xs_lo = np.log(wg*mu/(Kmax*sphi)); xs_hi = np.log(wg*mu/(Kmin*sphi))
        x = np.unique(np.concatenate([np.linspace(x_min, x_max, n_base),
                                      np.linspace(max(xs_lo,x_min), min(xs_hi,x_max), n_stat)]))
        u_cand = np.exp(x + 1j*phi)/mu
        # log-safe scaled target cosh(pi w/2) Gamma(iw) K^{-iw+kpow_shift} (O(1) magnitude), kappa-independent
        def _tgt(w, K):
            lc = np.pi*w/2 + np.log1p(np.exp(-np.pi*w)) - np.log(2.)
            return np.exp(lc + loggamma(1j*w)) * K**(-1j*w+kpow_shift)
        tc = np.cos((2*np.arange(n_train)+1)/(2*n_train)*np.pi)
        Ktr = np.exp(0.5*(np.log(Kmin)+np.log(Kmax)) + 0.5*(np.log(Kmax)-np.log(Kmin))*tc)
        Kva = np.exp(np.linspace(np.log(Kmin), np.log(Kmax), n_val))
        Atr = np.exp(-Ktr[:,None]*u_cand[None,:]); Ava = np.exp(-Kva[:,None]*u_cand[None,:])
        ftr_g = _tgt(wg, Ktr); fva_g = _tgt(wg, Kva)          # grid-defining target (frequency wg)
        ftr_o = _tgt(omega, Ktr); fva_o = _tgt(omega, Kva)    # actual target (frequency omega)
        _, _, piv = qr(Atr, mode='economic', pivoting=True)
        Nosc = int(np.ceil(wg/(2*np.pi)*np.log(Kmax/Kmin)))
        # grow the node count until the GRID-DEFINING (wg) fit converges -> fixes the node set
        W = None; Nt = max(2, Nosc-2)
        for Nt in range(max(2,Nosc-2), min(len(u_cand), 20*Nosc+80)):
            J = piv[:Nt]; Wg, *_ = lstsq(Atr[:,J], ftr_g)
            errg = np.max(np.abs(Ava[:,J]@Wg - fva_g)/np.maximum(np.abs(fva_g), 1e-300))
            if errg < tol: break
        # fit the returned WEIGHTS for the actual target omega on that (wg-defined) node set
        J = piv[:Nt]; W, *_ = lstsq(Atr[:,J], ftr_o)
        err = np.max(np.abs(Ava[:,J]@W - fva_o)/np.maximum(np.abs(fva_o), 1e-300))
        if verb:
            gtag = "" if grid_omega is None else " [grid_omega=%.3g]"%wg
            print("\tfNL-feat-res exp-sum: N_t=%d (N_osc=%d)%s, fit rel.err=%.2e, max|W|=%.2e"%(Nt,Nosc,gtag,err,np.max(np.abs(W))))
        if err >= tol: print("\tWARNING: fNL-feat-res exp-sum did not reach tol=%.1e (got %.2e); increase candidate density or tol"%(tol,err))
        # All selected nodes lie on the single ray arg(u)=phi, so the real t-grid t=|u|=e^x/mu fully
        # describes them (u = t e^{i phi}). Return the real t-grid + phi (the rotated-contour parameter).
        u_sel = u_cand[piv[:Nt]]
        return np.ascontiguousarray(np.abs(u_sel)), phi, W

    def set_feat_params(self, omega, kappa, verb=False):
        """Re-set the fNL-feat-res (omega, kappa) on the EXISTING node grid, WITHOUT recomputing the
        k-integral legs or any filtered maps. Only the (omega,kappa)-dependent 'assembly' quantities change:
          - self.feat_W / self.u_weights : exp-sum weights, refit for `omega` on the SAME nodes
          - self.feat_recipe             : bracket coefficients (omega-dependent)
          - self.omega, feat_params['kres_cs']
        The node grid (t_arr, phi, N_t) is UNCHANGED -- asserted below -- so any previously-computed legs
        (flXs_exp_re/im) and filtered maps (f_exp_re/im) remain exactly valid. This is the primitive that
        the *_feat_batch drivers use to sweep many (omega,kappa) while sharing all the expensive work.
        Requires the class to have been built with a fixed grid_omega covering the whole scan (so the node
        set does not depend on omega)."""
        assert hasattr(self, '_feat_build_cfg'), "set_feat_params requires an fNL-feat-res template!"
        cfg = self._feat_build_cfg
        t_new, phi_new, W_new = self._build_feat_res_expsum(
            omega, kappa, cfg['Kmin'], cfg['Kmax'], kpow_shift=cfg['kpow_shift'], tol=cfg['tol'],
            grid_omega=cfg['grid_omega'], phi_factor=cfg['phi_factor'], verb=verb)
        assert len(t_new) == self.N_t and np.array_equal(t_new, self.t_arr) and phi_new == self.feat_phi, \
            ("set_feat_params changed the node grid -- the class must be built with a fixed grid_omega "
             ">= every omega in the scan so the nodes are omega-independent (got N_t %d vs %d)."
             % (len(t_new), self.N_t))
        self.omega = omega
        self.feat_params = dict(self.feat_params); self.feat_params['kres_cs'] = kappa
        self.feat_W = W_new
        self.u_weights = np.asarray(W_new, dtype=np.complex128, order='C')
        self.feat_recipe = self._feat_bracket_recipe(omega, self.feat_scheme)

    def _feat_prefactor(self):
        """Complex scalar pref0 * kappa^{i omega} multiplying the fNL-feat-res bracket (cubic & linear)."""
        omega = self.omega; kappa = self.feat_params['kres_cs']
        return (2*np.pi**2*self.As)**2*omega**2/(4.*(omega+1j))*kappa**(1j*omega)

    def _feat_num_cubic(self, proc_maps):
        """Cubic (3-field) fNL-feat-res numerator term from pre-filtered data maps `proc_maps`
        (keys f_exp_re/f_exp_im). Reads the CURRENT (omega,kappa,W,recipe); returns b3 (real scalar).
        Identical arithmetic to the inline Bl_numerator branch -- shared by the single & batched paths."""
        prefactor = self._feat_prefactor()
        fre, fim = proc_maps['f_exp_re'], proc_maps['f_exp_im']
        rw = self.r_weights['fNL-feat-res']
        bracket = 0.+0.j
        for (ka, kb, kc), coeff in self.feat_recipe.items():
            bracket += coeff*self.utils.fnl_sum_2d_fullcomplex(
                rw, self.u_weights, fre[ka], fim[ka], fre[kb], fim[kb], fre[kc], fim[kc])
        return 1./3.*(prefactor*bracket).real*self.base.A_pix

    def _feat_num_linear_one(self, proc_maps, this_proc_maps):
        """One simulation's contribution to the fNL-feat-res linear (mean-field) term, using pre-filtered
        data maps `proc_maps` and sim maps `this_proc_maps`. Reads the CURRENT (omega,kappa,W,recipe).
        Returns the raw per-sim value b1_contrib = -(1/3) Re[C*bracket] A_pix (the caller averages over
        sims, i.e. divides the SUM over sims by N_it). Identical arithmetic to the inline Bl_numerator
        branch -- shared by the single & batched paths."""
        C = self._feat_prefactor()
        rw = self.r_weights['fNL-feat-res']
        d_re, d_im = proc_maps['f_exp_re'], proc_maps['f_exp_im']
        s_re, s_im = this_proc_maps['f_exp_re'], this_proc_maps['f_exp_im']
        bracket = 0.+0.j
        for key, gam in self.feat_recipe.items():
            for s in range(3):
                o1, o2 = [j for j in range(3) if j != s]
                kd, ka, kb = key[s], key[o1], key[o2]
                bracket += gam*self.utils.fnl_sum_2d_fullcomplex(
                    rw, self.u_weights, d_re[kd], d_im[kd], s_re[ka], s_im[ka], s_re[kb], s_im[kb])
        return -1./3.*(C*bracket).real*self.base.A_pix

    def _feat_q_deriv(self, Qs_obs, Fexp_maps, radial_index):
        """Accumulate the fNL-feat-res Q-derivative for the CURRENT (omega,kappa) into Qs_obs (shape
        (2, npol, nlm)), for a single radial index, from the SHARED filtered maps Fexp_maps and legs. See the
        long derivation note in compute_fisher_contribution for the 2/3 factor and the Re/Im expansion.
        Shared by the single & batched MC-Fisher paths -- the (omega,kappa) dependence enters only through the
        CURRENT prefactor C, weights feat_W, and recipe (Fexp_maps/legs are node-only, hence batch-shareable)."""
        C = self._feat_prefactor()
        rw = self.r_weights['fNL-feat-res'][radial_index]
        leg_re = {kp: np.asarray(self.flXs_exp_re[kp][:,:,radial_index,:], order='C') for kp in self.feat_kpows}
        leg_im = {kp: np.asarray(self.flXs_exp_im[kp][:,:,radial_index,:], order='C') for kp in self.feat_kpows}
        expansion = [(('re','re','re'), 1., 0.), (('re','im','im'), -1., 0.),
                     (('im','re','im'), -1., 0.), (('im','im','re'), -1., 0.),
                     (('re','re','im'), 0., -1.), (('re','im','re'), 0., -1.),
                     (('im','re','re'), 0., -1.), (('im','im','im'), 0., 1.)]
        wnode = np.ascontiguousarray((2./3.)*rw*np.ones(self.N_t, dtype=np.float64))
        for index in [0,1]:
            Mre = {kp: np.ascontiguousarray(Fexp_maps[kp][index,:,0,:].real) for kp in self.feat_kpows}
            Mim = {kp: np.ascontiguousarray(Fexp_maps[kp][index,:,0,:].imag) for kp in self.feat_kpows}
            def _themap(kp, typ): return Mre[kp] if typ=='re' else Mim[kp]
            inner_acc = {}   # (kpow, type) -> accumulated real inner map (N_t, Npix)
            for key, gam in self.feat_recipe.items():
                cc = C*gam*self.feat_W          # complex per-node array (includes W_node)
                ccr = cc.real; cci = cc.imag
                for types, fr, fi in expansion:
                    coeff = fr*ccr + fi*cci     # per-node real coefficient of this triple product
                    for s in range(3):
                        o1, o2 = [j for j in range(3) if j != s]
                        inner = _themap(key[o1], types[o1])*_themap(key[o2], types[o2])
                        ok = (key[s], types[s])
                        contrib = coeff[:,None]*inner
                        inner_acc[ok] = contrib if ok not in inner_acc else inner_acc[ok]+contrib
            for (kp_s, t_s), inner in inner_acc.items():
                leg = leg_re[kp_s] if t_s=='re' else leg_im[kp_s]
                Qs_obs[index] += self._transform_maps(np.ascontiguousarray(inner), leg, wnode)

    def _feat_q_plan(self):
        """Scheme-fixed plan for the batched MC-Fisher Q-derivative with SHARED outer SHTs. Returns
        (products, tuples_by_prod, outers) where `products` is the list of DISTINCT 2-leg product factors
        ((kp_a,type_a),(kp_b,type_b)) whose map SHT is param-independent, `tuples_by_prod[j]` lists the
        (key, outer=(kp,type), fr, fi) contributions of product j, and `outers` the distinct outer legs.
        Cached (depends only on scheme via recipe keys + kpows)."""
        if getattr(self, '_feat_q_plan_cache', None) is not None:
            return self._feat_q_plan_cache
        expansion = [(('re','re','re'), 1., 0.), (('re','im','im'), -1., 0.),
                     (('im','re','im'), -1., 0.), (('im','im','re'), -1., 0.),
                     (('re','re','im'), 0., -1.), (('re','im','re'), 0., -1.),
                     (('im','re','re'), 0., -1.), (('im','im','im'), 0., 1.)]
        products = []; prod_index = {}; tuples_by_prod = {}; outers = set()
        for key in self.feat_recipe:
            for types, fr, fi in expansion:
                for s in range(3):
                    o1, o2 = [j for j in range(3) if j != s]
                    pr = tuple(sorted([(key[o1], types[o1]), (key[o2], types[o2])]))
                    if pr not in prod_index:
                        prod_index[pr] = len(products); products.append(pr); tuples_by_prod[prod_index[pr]] = []
                    outer = (key[s], types[s])
                    tuples_by_prod[prod_index[pr]].append((key, outer, fr, fi))
                    outers.add(outer)
        self._feat_q_plan_cache = (products, tuples_by_prod, sorted(outers))
        return self._feat_q_plan_cache

    def _feat_q_batch(self, Qs, Fexp_maps, radial_index, params):
        """BATCHED fNL-feat-res Q-derivative for many (omega,kappa), SHARING the outer SHTs. Each distinct
        2-leg product map is SHT'd ONCE (param-independent) and scattered, in harmonic space, into each
        param's per-outer accumulator (only the per-node real coefficient coeff=fr*Re(cc)+fi*Im(cc),
        cc=pref0 kappa^{iw} gamma feat_W, is param-dependent). Then one radial_sum per (param,outer).
        Accumulates into Qs (shape (2, nparam, npol, nlm)). Because SHT(sum coeff*P) is only reordered
        (not changed) vs sum coeff*SHT(P), results match the per-param _feat_q_deriv path to the SHT
        round-off floor (~1e-13), not bitwise. Holds nparam*len(outers) harmonic accumulators."""
        products, tuples_by_prod, outers = self._feat_q_plan()
        kpows = self.feat_kpows; N_t = self.N_t
        rw = self.r_weights['fNL-feat-res'][radial_index]
        wnode = np.ascontiguousarray((2./3.)*rw*np.ones(N_t, dtype=np.float64))
        leg_re = {kp: np.asarray(self.flXs_exp_re[kp][:,:,radial_index,:], order='C') for kp in kpows}
        leg_im = {kp: np.asarray(self.flXs_exp_im[kp][:,:,radial_index,:], order='C') for kp in kpows}
        for index in [0, 1]:
            Mre = {kp: np.ascontiguousarray(Fexp_maps[kp][index,:,0,:].real) for kp in kpows}
            Mim = {kp: np.ascontiguousarray(Fexp_maps[kp][index,:,0,:].imag) for kp in kpows}
            _themap = lambda kp, typ: (Mre[kp] if typ == 're' else Mim[kp])
            # per-param cc[key] = pref0 kappa^{iw} gamma_key feat_W  (the ONLY (omega,kappa) dependence)
            ccs = []
            for (om, ka) in params:
                self.set_feat_params(om, ka)
                C = self._feat_prefactor()
                ccs.append({key: C*self.feat_recipe[key]*self.feat_W for key in self.feat_recipe})
            inner_lm = [dict() for _ in params]   # ip -> {outer: (N_t, nlm) complex}
            for j, (fa, fb) in enumerate(products):
                Pj = _themap(fa[0], fa[1])*_themap(fb[0], fb[1])           # (N_t, npix), shared
                PLj = np.asarray(self.base.to_lm_vec(Pj, lmax=self.lmax)[:, self.lminfilt], order='C')  # shared SHT
                for (key, outer, fr, fi) in tuples_by_prod[j]:
                    for ip in range(len(params)):
                        cc = ccs[ip][key]
                        coeff = fr*cc.real + fi*cc.imag                    # (N_t,) real
                        contrib = coeff[:, None]*PLj
                        inner_lm[ip][outer] = contrib if outer not in inner_lm[ip] else inner_lm[ip][outer] + contrib
            for ip in range(len(params)):
                for outer, ilm in inner_lm[ip].items():
                    leg = leg_re[outer[0]] if outer[1] == 're' else leg_im[outer[0]]
                    Qs[index, ip] += self.utils.radial_sum(np.ascontiguousarray(ilm), wnode, leg)

    def _build_and_optimize_ru_grid(self, r_init, r_quad_weights, u_min=0.01, u_safety_factor=15, pts_per_period=15, tolerance=1e-3, verb=False, label='fNL-feat-res (r,u)'):
        """
        Core (r,u) ragged-grid build + ideal Fisher computation + greedy prune, given an explicit
        array of r points and their quadrature weights (r^2 dr convention). Factored out of
        optimize_ru_sampling so a pre-thinned/localized r-candidate set (e.g. from
        optimize_ru_sampling_staged), or a deliberately over-resolved grid (for a joint-convergence
        check), can reuse the same pair-grid/Fisher/prune logic without going through the
        raw-grid-from-reduce_r construction. Per-r u_arr density need not follow any fixed formula --
        this only requires r_init (arbitrary, can repeat r values with different u ranges) and a
        matching per-r u_max_r/N_u_r pair; the ragged-grid loop below is what currently derives those
        from (u_safety_factor, pts_per_period), but nothing downstream assumes that particular rule.

        Returns: r_opt, u_opt, weights_opt, init_score. init_score is deriv_matrix.sum() -- the total
        ideal Fisher of the *unpruned* input grid -- so calling this with tolerance=1.0 (skip pruning)
        and comparing init_score across grids of different density is a direct joint (r,u) convergence
        check.
        """
        t_init = time.time()

        omega = self.omega
        kappa = self.feat_params['kres_cs']
        npol = 1+2*self.pol
        period = 2.*np.pi/omega

        r_init = np.asarray(r_init)
        r_quad_weights = np.asarray(r_quad_weights)
        N_r = len(r_init)

        # Precompute Bessel functions once, over the full r-range
        max_kr = max(self.k_arr)*max(r_init)
        x_arr = np.asarray(list(np.arange(0,self.lmax*2,0.01))+list(np.arange(self.lmax*2,max_kr,0.1)),dtype=np.float64)
        jlxs = np.zeros((self.lmax-self.lmin+1,len(x_arr)),dtype=np.float64,order='C')
        compute_bessel(x_arr,self.lmin,self.lmax,jlxs,self.base.nthreads)
        jlkr_all = interpolate_jlkr(x_arr, self.k_arr, r_init, jlxs, self.base.nthreads)

        # Build the ragged (r,u) grid, computing legs per-r (each r needs its own u_arr)
        r_list, u_list, wt_list, uwt_list = [], [], [], []
        flXs_pieces = {p: [] for p in [-3,-2,-1,0,1]}
        for ir in range(N_r):
            r_val = r_init[ir]
            u_max_r = u_safety_factor*r_val/self.lmin
            N_u_r = max(3, int(pts_per_period*np.log(u_max_r/u_min)/period)+1)
            u_arr_r = np.geomspace(u_min, u_max_r, N_u_r)

            log_u = np.log(u_arr_r)
            dlnu = np.zeros(N_u_r)
            dlnu[:-1] += 0.5*np.diff(log_u)
            dlnu[1:] += 0.5*np.diff(log_u)
            du_r = dlnu*u_arr_r

            jlkr_r = np.ascontiguousarray(jlkr_all[:,[ir],:])
            for p in [-3,-2,-1,0,1]:
                out = np.zeros((self.lmax+1, npol, 1, N_u_r), dtype=np.float64, order='C')
                q_integral_exp(self.k_arr, float(p), u_arr_r, self.Tl_arr, jlkr_r, self.lmin, self.lmax, self.base.nthreads, out)
                flXs_pieces[p].append(out[:,:,0,:])

            r_list.append(np.full(N_u_r, r_val))
            u_list.append(u_arr_r)
            wt_list.append(np.full(N_u_r, r_quad_weights[ir]))
            uwt_list.append(du_r.astype(np.complex128)*u_arr_r**(1j*omega))

        r_pairs = np.concatenate(r_list)
        u_pairs = np.concatenate(u_list)
        weights_pairs = np.asarray(np.concatenate(wt_list), order='C')
        u_weight_pairs = np.concatenate(uwt_list)
        N_pairs = len(r_pairs)
        if verb: print("# Ragged (r,u) grid: N_r = %d, N_pairs = %d"%(N_r, N_pairs))

        flXs_full = {p: np.asarray(np.concatenate(flXs_pieces[p], axis=-1), order='C') for p in [-3,-2,-1,0,1]}

        prefactor = -0.25*(omega+3j)*(2*np.pi**2*self.As)**2*omega**2*np.cosh(np.pi*omega/2)*kappa**(1j*omega)
        coeff1 = 1./(1j*omega*(1j*omega-1)*(1j*omega-3))
        coeff2 = 1./(1j*omega*(1j*omega-1))
        coeff3 = 1./(1j*omega)
        VQ1 = np.asarray(2.*np.real(prefactor*(3.*coeff1)*u_weight_pairs), order='C')
        VQ2 = np.asarray(2.*np.real(prefactor*(24.*coeff1+6.*coeff2)*u_weight_pairs), order='C')
        VQ3 = np.asarray(2.*np.real(prefactor*(18.*coeff1+6.*coeff2)*u_weight_pairs), order='C')
        VQ4 = np.asarray(2.*np.real(prefactor*(36.*coeff1+15.*coeff2+3.*coeff3)*u_weight_pairs), order='C')

        [mus, w_mus] = p_roots(2*self.lmax+1)
        legs = np.asarray([lpmn(0,self.lmax,mus[i])[0][0,self.lmin:] for i in range(len(mus))])

        if verb: print("# Computing ideal Fisher matrix on the ragged grid (%d x %d)"%(N_pairs,N_pairs))
        deriv_matrix = np.asarray(fisher_deriv_fNL_feat_res_2d(flXs_full[-3], flXs_full[-2], flXs_full[-1], flXs_full[0], flXs_full[1],
                            weights_pairs, VQ1, VQ2, VQ3, VQ4,
                            np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'),
                            legs, w_mus, self.lmin, self.lmax, self.base.nthreads))

        inds, w_opt, init_score, fish = self._greedy_optimize_fisher_matrix(deriv_matrix, tolerance=tolerance, verb=verb, label=label)
        if verb: print("\nScore threshold met with %d of %d pairs"%(len(inds), N_pairs))

        r_opt = r_pairs[inds]
        u_opt = u_pairs[inds]
        weights_opt = w_opt*weights_pairs[inds]

        print("\n%s optimization complete after %.2f seconds (%d -> %d pairs)"%(label, time.time()-t_init, N_pairs, len(inds)))

        return r_opt, u_opt, weights_opt, init_score

    def optimize_ru_sampling(self, reduce_r=1, u_min=0.01, u_safety_factor=15, pts_per_period=15, tolerance=1e-3, verb=False):
        """
        Jointly optimize the (r,u) sampling for the fNL-feat-res template, generalizing
        optimize_radial_sampling_1d to the pair-collapsed architecture. This is necessary because the
        'good' u range for a given r scales with r itself -- the u-integral for a leg only decays once u
        exceeds ~r/l_min for the dominant l=l_min contribution (empirically confirmed: convergence for a
        single r happens once u_max reaches ~10-20x r/l_min, independent of omega) -- so a single, shared
        u_arr across all r (as used by the ints_1d/ints_2d-with-full-outer-product pathway) is wasteful:
        small r never needs a large u_max, while large r genuinely does.

        Step 1: build an initial, ragged (r,u) pair grid -- the same fiducial r_raw construction as
        optimize_radial_sampling_1d, but with u_arr(r) = geomspace(u_min, u_safety_factor*r/lmin, N_u(r)),
        N_u(r) chosen for pts_per_period samples per u^{i*omega} oscillation period in ln(u). Note this
        u_max(r)/N_u(r) rule is just a convenient, empirically-motivated *starting point* -- the grid it
        produces is already non-regular (ragged: more u points at large r than small r), and there is no
        requirement that production use exactly this rule; any initial (r,u) point set can be fed to
        _build_and_optimize_ru_grid directly. What matters is verifying the initial grid is fine enough,
        jointly in (r,u), that the ideal Fisher has converged -- see the joint-convergence check
        recommended alongside this method (compare init_score across independently-refined r- and
        u-density before trusting the pruned result).
        Step 2: compute the ideal Fisher matrix directly on this ragged pair grid (no shared-u tensor
        product, so N_pairs = sum_r N_u(r), not N_r * max(N_u)).
        Step 3: greedily prune via _greedy_optimize_fisher_matrix (the same Smith & Zaldarriaga algorithm
        as optimize_radial_sampling_1d), now operating on (r,u) pairs instead of r alone.

        This does an O(N_pairs^2) dense Fisher build, which only stays tractable for N_r up to a few
        tens (N_pairs up to a few thousand); it stalls well before N_pairs~2e4 (empirically, this did
        not converge within 10 minutes at N_r=52). For a fine production r-grid (e.g. reduce_r~2),
        use optimize_ru_sampling_staged instead.

        Returns:
            - r_opt, u_opt: (N_pairs_opt,) arrays of the optimized (r,u) points
            - weights_opt: (N_pairs_opt,) their combined (r^2 dr times quadratic-optimization) weights
            - ideal_fisher: total ideal Fisher (sum over the full, unoptimized ragged grid)
        """
        assert 'fNL-feat-res' in self.templates, "optimize_ru_sampling is only implemented for fNL-feat-res!"

        # Fiducial r-grid: same construction as optimize_radial_sampling_1d
        r_raw = np.asarray(list(np.arange(1,self.r_star*0.95,50*reduce_r))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,2.5*reduce_r))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r)))
        r_init = 0.5*(r_raw[1:]+r_raw[:-1])
        r_quad_weights = r_init**2*np.diff(r_raw)
        if verb: print("# Fiducial r-grid: N_r = %d"%len(r_init))

        return self._build_and_optimize_ru_grid(r_init, r_quad_weights, u_min=u_min, u_safety_factor=u_safety_factor, pts_per_period=pts_per_period, tolerance=tolerance, verb=verb, label='fNL-feat-res (r,u)')

    def optimize_ru_sampling_staged(self, reduce_r_coarse=80, reduce_r_fine=2, refine_window=1.5, u_min=0.01, u_safety_factor=15, pts_per_period=15, coarse_tolerance=1e-3, tolerance=1e-3, verb=False):
        """
        Two-stage version of optimize_ru_sampling for when the target (fine, production-resolution)
        r-grid is too large for a direct joint (r,u) Fisher build+prune. optimize_ru_sampling's dense
        Fisher build costs O(N_pairs^2); this was empirically found to stall (not converge within 10
        minutes) already at N_pairs~2e4 (N_r=52, reduce_r=20), well short of the coarsen=2 production
        target (N_r in the hundreds+). This method avoids ever building that full-size dense matrix.

        Stage A: run the existing optimize_ru_sampling at a coarse r-grid (reduce_r_coarse -- the same
        scale already validated to work directly, e.g. reduce_r=80 giving N_r=14, N_pairs~5e3) to find
        which r *regions* carry the bulk of the Fisher information (5339->18 pairs, 99.9% retained, was
        the benchmark result at this scale).

        Stage B: build the *fine* r-grid (reduce_r_fine, the actual target resolution), but keep only
        fine-grid points within +/- refine_window * (local coarse-grid spacing) of each distinct r
        selected by Stage A. This keeps the Stage-B candidate set small (localized around a handful of
        important regions, not spanning the whole fine grid), so the joint Fisher build + greedy prune
        stays in the already-validated tractable regime, now at full target r-resolution.

        Caveat: this assumes Fisher information in r is concentrated in a few localized regions (well
        supported by Stage-A's own strong compression, and by the physical picture -- last scattering
        and reionization/ISW shells -- but not independently proven optimal for the fine grid). Treat
        the result as a good, cheap, production-usable approximation, not a certified global optimum --
        confirm with a joint (r,u) convergence check (e.g. rerun Stage B with a larger refine_window
        and/or higher pts_per_period/u_safety_factor and check init_score is stable) before trusting it
        at full scale, per the same logic as optimize_ru_sampling's docstring.

        Returns: r_opt, u_opt, weights_opt, init_score (init_score is Stage B's ragged-candidate-set
        total, i.e. restricted to the localized candidate region, not the full fine grid).
        """
        assert 'fNL-feat-res' in self.templates, "optimize_ru_sampling_staged is only implemented for fNL-feat-res!"

        if verb: print("## Stage A: coarse (r,u) optimization to find important r regions (reduce_r=%d)"%reduce_r_coarse)
        r_opt_c, u_opt_c, w_opt_c, fisher_c = self.optimize_ru_sampling(reduce_r=reduce_r_coarse, u_min=u_min, u_safety_factor=u_safety_factor, pts_per_period=pts_per_period, tolerance=coarse_tolerance, verb=verb)
        r_important = np.unique(r_opt_c)
        if verb: print("# Stage A selected %d distinct r regions"%len(r_important))

        if verb: print("\n## Stage B: building fine r-grid (reduce_r=%d) and localizing candidates near Stage-A regions"%reduce_r_fine)
        r_raw_fine = np.asarray(list(np.arange(1,self.r_star*0.95,50*reduce_r_fine))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,2.5*reduce_r_fine))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r_fine)))
        r_fine = 0.5*(r_raw_fine[1:]+r_raw_fine[:-1])
        r_fine_quad_weights = r_fine**2*np.diff(r_raw_fine)
        if verb: print("# Fine r-grid: N_r = %d"%len(r_fine))

        # Local window width at each Stage-A r: use the *coarse* grid's local spacing as the window scale
        r_raw_coarse = np.asarray(list(np.arange(1,self.r_star*0.95,50*reduce_r_coarse))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,2.5*reduce_r_coarse))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r_coarse)))
        coarse_spacing = np.diff(r_raw_coarse)
        coarse_centers = 0.5*(r_raw_coarse[1:]+r_raw_coarse[:-1])

        mask = np.zeros(len(r_fine), dtype=bool)
        for r_val in r_important:
            i_c = np.argmin(np.abs(coarse_centers-r_val))
            window = refine_window*coarse_spacing[i_c]
            mask |= (np.abs(r_fine-r_val) <= window)
        r_candidate = r_fine[mask]
        r_candidate_weights = r_fine_quad_weights[mask]
        if verb: print("# Candidate fine-r set near Stage-A regions: N_r = %d (of %d fine points)"%(len(r_candidate), len(r_fine)))

        if verb: print("\n## Stage B: joint (r,u) Fisher build + prune on the localized candidate set")
        return self._build_and_optimize_ru_grid(r_candidate, r_candidate_weights, u_min=u_min, u_safety_factor=u_safety_factor, pts_per_period=pts_per_period, tolerance=tolerance, verb=verb, label='fNL-feat-res (r,u) [staged]')

    ### NUMERATOR
    @_timer_func('numerator')
    def Bl_numerator(self, data, include_linear_term=True, verb=False, input_type='map', return_cubic=False):
        """
        Compute the numerator of the quasi-optimal bispectrum estimator for all templates.

        We optionally include the linear terms, which can reduce the estimator variance.
        """
        # Check we have initialized correctly
        if self.ints_1d and (not hasattr(self,'r_arr')):
            raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
        
        # Check if simulations have been supplied
        if not hasattr(self, 'preload') and include_linear_term:
            raise Exception("Need to generate or specify bias simulations!")

        # Check input data format
        if self.pol:
            assert len(data)==3, "Data must contain T, Q, U components!"
        else:
            if input_type=='map':
                assert (len(data)==1 and len(data[0])==self.base.Npix) or len(data)==self.base.Npix, "Data must contain T only!"

        # Apply S^-1 to data and transform to harmonic space
        if input_type == 'Sinv_map':
            assert data.shape[1]==self.lminfilt.sum(), "S^-1.data has the wrong shape!"
            h_data_lm = data.copy()
        else:
            t_init = time.time()
            h_data_lm = np.asarray(self.applySinv(data, input_type=input_type, lmax=self.lmax)[:,self.lminfilt], order='C')
            self.timers['Sinv'] += time.time()-t_init
           
        # Compute all relevant weighted maps
        proc_maps = self._apply_all_filters(h_data_lm)
        
        # Define 3- and 1-field arrays
        b3_num = np.zeros(self.total_size)
        if include_linear_term:
            b1_num = np.zeros(self.total_size)
            
        if verb: print("# Assembling bispectrum numerator (3-field term)")
        index = 0
        for t in self.templates:
            
            if t=='fNL-loc':
                # fNL-local template
                print("Computing fNL-local template")
                
                t_init = time.time()
                b3_num[index] = 3./5.*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], proc_maps['f_m1'], proc_maps['f_p2'])*self.base.A_pix
                index += 1
                self.timers['fNL_summation'] += time.time()-t_init
                
            elif t=='fNL-eq':
                # fNL-eq template
                print("Computing fNL-eq template")
                
                t_init = time.time()
                summ  = 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p1'], proc_maps['f_p0'], proc_maps['f_m1'])
                summ -= 3*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], proc_maps['f_m1'], proc_maps['f_p2'])
                summ -= 2*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], proc_maps['f_p0'], proc_maps['f_p0'])
                b3_num[index] = 3./5.*summ*self.base.A_pix
                index += 1
                self.timers['fNL_summation'] += time.time()-t_init

            elif t=='fNL-orth':
                # fNL-eq template
                print("Computing fNL-orth template")
                
                t_init = time.time()
                summ  = 18*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p1'], proc_maps['f_p0'], proc_maps['f_m1'])
                summ -= 9*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], proc_maps['f_m1'], proc_maps['f_p2'])
                summ -= 8*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], proc_maps['f_p0'], proc_maps['f_p0'])
                b3_num[index] = 3./5.*summ*self.base.A_pix
                index += 1
                self.timers['fNL_summation'] += time.time()-t_init

            elif t=='fNL-orth2':
                # fNL-eq template
                print("Computing fNL-orth2 template")

                t_init = time.time()
                p = 27./(743./(7.*(20*np.pi**2.-193.))-21.)
                summ = -(2+20./9.*p)*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], proc_maps['f_p0'], proc_maps['f_p0'])
                summ += (6+10./3.*p)*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], proc_maps['f_p0'], proc_maps['f_p1'])
                summ -= 20./9.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], proc_maps['f_p1'], proc_maps['f_p1'])
                summ -= (3+p)*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], proc_maps['f_m1'], proc_maps['f_p2'])
                summ += 10./3.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], proc_maps['f_p0'], proc_maps['f_p2'])
                summ -= 4./3.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], proc_maps['f_m1'], proc_maps['f_p3'])
                summ += 1./9.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], proc_maps['f_m2'], proc_maps['f_p4'])
                b3_num[index] = 3./5.*summ*self.base.A_pix
                index += 1
                self.timers['fNL_summation'] += time.time()-t_init

            elif t=='fNL-feat-res':
                # fNL-feat-res template, cubic (3-field) numerator term. Compressed exponential-sum scheme
                # (FEAT_NOTES.md Sessions 4-6). The u-integral factorises exactly: S(u)=S_0 e^{-Ku}, so
                #   B_res/A_res = 2 Re[ pref0 * kappa^{iw} * F_hat(K) * S_0 ],   K = k1+k2+k3,
                #   pref0 = (2 pi^2 As)^2 w^2 / (4(w+i)),   F_hat(K) = sum_j W_j e^{-K u_j}  (~ cosh Gamma K^{-iw}),
                # with complex nodes u_j and complex weights W_j (cosh & Gamma(iw) folded into W_j). In the
                # estimator, e^{-K u_j}=prod_a e^{-k_a u_j} is separable and S_0 is the m=n bracket (sm3) or
                # its K-raised form (sm1). The bracket -> fnl_sum recipe (see _feat_bracket_recipe) uses the
                # standard fNL-eq/orth convention (coeff = template_coeff x #orderings). Legs are COMPLEX;
                # the complex triple product is contracted by fnl_sum_2d_fullcomplex (Re/Im expanded, 1j at
                # the end). u_weights = W_j (NO u^{iw} power -- F is represented directly).
                print("Computing fNL-feat-res template (%s, exp-sum, N_t=%d)"%(self.feat_scheme, self.N_t))

                t_init = time.time()
                # Assembly extracted to _feat_num_cubic (shared by the batched driver). Normalization:
                # verified (FEAT_NOTES Session 6) that (prefactor*bracket) reproduces the PREVIOUS m=n+1
                # code's complex `summ` to machine precision (ratio 1.0, phase ~1e-16, all w,kappa), so the
                # previous (correct) (1/3)*summ.real*A_pix wrapper carries over UNCHANGED.
                b3_num[index] = self._feat_num_cubic(proc_maps)
                index += 1
                #TODO: add b1_num
                self.timers['fNL_summation'] += time.time()-t_init
            
            elif 'neural' in t:
                # Neural-network input template
                n = int(t.split('-')[1])
                print("Computing neural-%d template"%n)
                
                t_init = time.time()
                if not self.neural_cyclic[n]:
                    b3_num[index] = 3./5.*self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], proc_maps['neural-alpha-%d'%n], proc_maps['neural-beta-%d'%n], proc_maps['neural-gamma-%d'%n])*self.base.A_pix 
                else: 
                    b3_num[index] = 3./5.*self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], proc_maps['neural-alpha-%d'%n], proc_maps['neural-beta-%d'%n], proc_maps['neural-beta-%d'%n])*self.base.A_pix
                index += 1
                self.timers['fNL_summation'] += time.time()-t_init
                
            elif t=='isw-lensing':
                # ISW-Lensing template
                print("Computing ISW-lensing template")
                
                t_init = time.time()
                b3_num[index] = 0.5*isw_bispectrum_sum(proc_maps['u'], proc_maps['v'], proc_maps['v-isw'], self.base.nthreads)*self.base.A_pix
                index += 1
                self.timers['lensing_summation'] += time.time()-t_init

            elif t=='binned':
                
                # Binned bispectra
                print("Computing binned template")
                t_init = time.time()
                self.utils.assemble_b3_binned(self.unique_bin_ids.astype(np.int32), self.k_triplet_ids.astype(np.int32), self.bin_degeneracy.astype(np.int32), b3_num, index, self.base.A_pix, self.r_weights[t], proc_maps['f_bin'])
                index += len(self.unique_bin_ids)
                self.timers['fNL_summation'] += time.time()-t_init

        if include_linear_term:

            # Iterate over simulations
            for isim in range(self.N_it):
                if verb: print("# Assembling bispectrum linear term for simulation %d of %d"%(isim+1,self.N_it))

                # Load processed bias simulations
                if self.preload:
                    this_proc_maps = self.proc_maps[isim]
                else:
                    this_proc_maps = self.load_sim_data(isim)

                # Compute templates
                index = 0
                for t in self.templates:
                    if t=='fNL-loc':
                        t_init = time.time()
                        
                        # Sum over permutations
                        summ  = 2.*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], this_proc_maps['f_m1'], this_proc_maps['f_p2'])
                        summ += self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], this_proc_maps['f_m1'], proc_maps['f_p2'])
                        b1_num[index] += -3./5.*summ*self.base.A_pix/self.N_it
                        index += 1
                        self.timers['fNL_summation'] += time.time()-t_init
                        
                    elif t=='fNL-eq':
                        t_init = time.time()

                        # Sum over permutations
                        summ  = 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p1'], this_proc_maps['f_p0'], this_proc_maps['f_m1'])
                        summ += 6*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_p1'], proc_maps['f_p0'], this_proc_maps['f_m1'])
                        summ += 6*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_p1'], this_proc_maps['f_p0'], proc_maps['f_m1'])
                        summ -= 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], this_proc_maps['f_m1'], this_proc_maps['f_p2'])
                        summ -= 3*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], this_proc_maps['f_m1'], proc_maps['f_p2'])
                        summ -= 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], this_proc_maps['f_p0'], this_proc_maps['f_p0'])
                        b1_num[index] += -3./5.*summ*self.base.A_pix/self.N_it
                        index += 1
                        self.timers['fNL_summation'] += time.time()-t_init

                    elif t=='fNL-orth':
                        t_init = time.time()
                        
                        # Sum over permutations
                        summ  = 18*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p1'], this_proc_maps['f_p0'], this_proc_maps['f_m1'])
                        summ += 18*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_p1'], proc_maps['f_p0'], this_proc_maps['f_m1'])
                        summ += 18*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_p1'], this_proc_maps['f_p0'], proc_maps['f_m1'])
                        summ -= 18*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], this_proc_maps['f_m1'], this_proc_maps['f_p2'])
                        summ -= 9*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], this_proc_maps['f_m1'], proc_maps['f_p2'])
                        summ -= 24*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], this_proc_maps['f_p0'], this_proc_maps['f_p0'])
                        b1_num[index] += -3./5.*summ*self.base.A_pix/self.N_it
                        index += 1
                        self.timers['fNL_summation'] += time.time()-t_init

                    elif t=='fNL-orth2':
                        t_init = time.time()

                        # Sum over permutations
                        summ  = 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p1'], this_proc_maps['f_p0'], this_proc_maps['f_m1'])
                        summ += 6*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_p1'], proc_maps['f_p0'], this_proc_maps['f_m1'])
                        summ += 6*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_p1'], this_proc_maps['f_p0'], proc_maps['f_m1'])
                        summ -= 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], this_proc_maps['f_m1'], this_proc_maps['f_p2'])
                        summ -= 3*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], this_proc_maps['f_m1'], proc_maps['f_p2'])
                        summ -= 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], this_proc_maps['f_p0'], this_proc_maps['f_p0'])
                        
                        # Sum over permutations
                        p = 27./(743./(7.*(20*np.pi**2.-193.))-21.)
                        summ  = -3.*(2+20./9.*p)*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_p0'], this_proc_maps['f_p0'], this_proc_maps['f_p0'])
                        
                        summ += (6+10./3.*p)*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], this_proc_maps['f_p0'], this_proc_maps['f_p1'])
                        summ += (6+10./3.*p)*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], proc_maps['f_p0'], this_proc_maps['f_p1'])
                        summ += (6+10./3.*p)*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], this_proc_maps['f_p0'], proc_maps['f_p1'])
                        
                        summ -= 20./9.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], this_proc_maps['f_p1'], this_proc_maps['f_p1'])
                        summ -= 40./9.*p*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m2'], proc_maps['f_p1'], this_proc_maps['f_p1'])
                        
                        summ -= 2*(3+p)*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m1'], this_proc_maps['f_m1'], this_proc_maps['f_p2'])
                        summ -= (3+p)*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m1'], this_proc_maps['f_m1'], proc_maps['f_p2'])
                        
                        summ += 10./3.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], this_proc_maps['f_p0'], this_proc_maps['f_p2'])
                        summ += 10./3.*p*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m2'], proc_maps['f_p0'], this_proc_maps['f_p2'])
                        summ += 10./3.*p*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m2'], this_proc_maps['f_p0'], proc_maps['f_p2'])
                        
                        summ -= 4./3.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], this_proc_maps['f_m1'], this_proc_maps['f_p3'])
                        summ -= 4./3.*p*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m2'], proc_maps['f_m1'], this_proc_maps['f_p3'])
                        summ -= 4./3.*p*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m2'], this_proc_maps['f_m1'], proc_maps['f_p3'])
                        
                        summ += 2./9.*p*self.utils.fnl_sum(self.r_weights[t], proc_maps['f_m2'], this_proc_maps['f_m2'], this_proc_maps['f_p4'])
                        summ += 1./9.*p*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['f_m2'], this_proc_maps['f_m2'], proc_maps['f_p4'])


                        b1_num[index] += -3./5.*summ*self.base.A_pix/self.N_it
                        index += 1
                        self.timers['fNL_summation'] += time.time()-t_init

                    elif t=='fNL-feat-res':
                        # Linear (mean-field) term for the compressed exp-sum rep (sm3/sm1). Same recipe and
                        # (1/3)Re[C*.]A_pix wrapper as the cubic numerator, but with 2 of the 3 legs replaced by
                        # a simulation and summed over the 3 data-placements (which leg is data), averaged over
                        # N_it sims. Complex data/sim maps -> fnl_sum_2d_fullcomplex (verified). Both sim legs use
                        # the SAME sim (this_proc_maps); the sim-average estimates the disconnected <a a> piece.
                        t_init = time.time()
                        # Assembly extracted to _feat_num_linear_one (shared by the batched driver); the
                        # per-sim value already carries -(1/3)Re[C*bracket]A_pix, averaged here over N_it.
                        b1_num[index] += self._feat_num_linear_one(proc_maps, this_proc_maps)/self.N_it
                        index += 1
                        self.timers['fNL_summation'] += time.time()-t_init

                    elif 'neural' in t:
                        n = int(t.split('-')[1])
                        t_init = time.time()
                        
                        # Sum over permutations
                        if not self.neural_cyclic[n]:
                            summ  = self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], this_proc_maps['neural-alpha-%d'%n], this_proc_maps['neural-beta-%d'%n], proc_maps['neural-gamma-%d'%n])
                            summ += self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], this_proc_maps['neural-alpha-%d'%n], this_proc_maps['neural-gamma-%d'%n], proc_maps['neural-beta-%d'%n])
                            summ += self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], this_proc_maps['neural-beta-%d'%n], this_proc_maps['neural-gamma-%d'%n], proc_maps['neural-alpha-%d'%n])
                            b1_num[index] += -3./5.*summ*self.base.A_pix/self.N_it
                        else:
                            summ  = 2.*self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], this_proc_maps['neural-alpha-%d'%n], this_proc_maps['neural-beta-%d'%n], proc_maps['neural-beta-%d'%n])
                            summ += self.utils.neural_sum(self.r_weights[t], self.neural_weights[n], this_proc_maps['neural-beta-%d'%n], this_proc_maps['neural-beta-%d'%n], proc_maps['neural-alpha-%d'%n])
                            b1_num[index] += -3./5.*summ*self.base.A_pix/self.N_it
                        index += 1
                        self.timers['fNL_summation'] += time.time()-t_init
                    
                    elif t=='isw-lensing':
                        t_init = time.time()

                        # Sum over 3 permutations
                        summ =  isw_bispectrum_sum(proc_maps['u'], this_proc_maps['v'], this_proc_maps['v-isw'], self.base.nthreads)
                        summ += isw_bispectrum_sum(this_proc_maps['u'], proc_maps['v'], this_proc_maps['v-isw'], self.base.nthreads)
                        summ += isw_bispectrum_sum(this_proc_maps['u'], this_proc_maps['v'], proc_maps['v-isw'], self.base.nthreads)
                        b1_num[index] += -0.5*summ*self.base.A_pix/self.N_it
                        index += 1
                        self.timers['lensing_summation'] += time.time()-t_init

                    elif t=='binned':

                        t_init = time.time()
                        self.utils.assemble_b1_binned(self.unique_bin_ids.astype(np.int32), self.k_triplet_ids.astype(np.int32), self.bin_degeneracy.astype(np.int32), b1_num, index, self.base.A_pix, self.N_it, self.r_weights[t], proc_maps['f_bin'], this_proc_maps['f_bin'])
                        index += len(self.unique_bin_ids)
                        self.timers['fNL_summation'] += time.time()-t_init
                
        if include_linear_term:
            b_num = b3_num+b1_num
        else:
            b_num = b3_num

        if return_cubic:
            return b_num, b3_num
        else:
            return b_num

    @_timer_func('numerator')
    def Bl_numerator_feat_batch(self, data, feat_params_list, include_linear_term=True, verb=False, input_type='map', return_cubic=False):
        """BATCHED fNL-feat-res numerator for a SET of (omega,kappa) sharing the current node + radial grid.

        The (omega,kappa)-INDEPENDENT work is done ONCE and reused across the whole batch:
          - S^-1.data and its filtered feature maps (the data SHTs), and
          - each bias simulation's filtered feature maps (the sim SHTs, the dominant cost of the linear term).
        Only the cheap complex triple-product 'assembly' (fnl_sum_2d_fullcomplex, weighted by the per-omega W
        and recipe and the pref0*kappa^{iw} scalar) is looped over the batch. Returns b_num (nparam,) and, if
        return_cubic, also b3_num (nparam,) -- each element equal to the corresponding single-param
        Bl_numerator run to machine precision.

        Assumes 'fNL-feat-res' is the (only) template. feat_params_list: list of (omega,kappa) or dicts."""
        assert self.templates == ['fNL-feat-res'], "Bl_numerator_feat_batch supports a lone fNL-feat-res template"
        if self.ints_1d and (not hasattr(self, 'r_arr')):
            raise Exception("Need radial integration points (prepare a common r-grid) first!")
        if include_linear_term and (not hasattr(self, 'preload')):
            raise Exception("Need to generate or specify bias simulations!")
        params = [p if isinstance(p, (tuple, list)) else (p['omega'], p['kres_cs']) for p in feat_params_list]
        nparam = len(params)

        # S^-1.data -> filtered feature maps (ONE set of data SHTs, shared by every param)
        if input_type == 'Sinv_map':
            h_data_lm = data.copy()
        else:
            t_init = time.time()
            h_data_lm = np.asarray(self.applySinv(data, input_type=input_type, lmax=self.lmax)[:, self.lminfilt], order='C')
            self.timers['Sinv'] += time.time()-t_init
        if verb: print("# Batched feat numerator: filtering data (shared across %d params)" % nparam)
        proc_maps = self._apply_all_filters(h_data_lm)

        # cubic (3-field) term, per param
        b3_num = np.zeros(nparam)
        for ip, (om, ka) in enumerate(params):
            self.set_feat_params(om, ka)
            b3_num[ip] = self._feat_num_cubic(proc_maps)

        # linear (mean-field) term: build each sim's maps ONCE, then loop params
        b1_num = np.zeros(nparam)
        if include_linear_term:
            for isim in range(self.N_it):
                if verb: print("# Batched feat numerator: linear term, sim %d of %d" % (isim+1, self.N_it))
                this_proc_maps = self.proc_maps[isim] if self.preload else self.load_sim_data(isim)
                for ip, (om, ka) in enumerate(params):
                    self.set_feat_params(om, ka)
                    b1_num[ip] += self._feat_num_linear_one(proc_maps, this_proc_maps)/self.N_it

        b_num = b3_num + b1_num if include_linear_term else b3_num
        if return_cubic:
            return b_num, b3_num
        return b_num

    ### OPTIMIZATION
    @_timer_func('optimization')
    def optimize_radial_sampling_1d(self, reduce_r=1, tolerance=1e-3, N_split=None, split_index=None, initial_r_points=None, verb=False, ideal_only=False, input_derivatives={}, r_fine_lo=None, r_fine_hi=None):
        """
        Compute the 1D radial sampling points and weights via optimization (as in Smith & Zaldarriaga 06), up to some tolerance in the Fisher distance.
        Optimization will be done for each template, analytically computing the 'distance' between template approximations
        
        Main Inputs:
            - reduce_r: Downsample the number of points in the starting radial integral grid (default: 1)
            - tolerance: Convergence threshold for the optimization (default 1e-3). This indicates the approximate error in the Fisher matrix induced by the optimization.
            
        For large problems, it is too expensive to optimize the whole matrix at once. Instead, we can split the optimization into N_split pieces, each of which is optimized separately.
        Following this, we perform a final optimization of all N_split components, using the union of all previously obtained radial points. 
        
        Additional Inputs (for chunked computations):
            - N_split (optional): Number of chunks to split the optimization into. If None, no splitting is performed.
            - split_index (optional): Index of the chunk to optimize. 
            - initial_r_points (optional): Starting set of radial points (used for the final optimization step).
            - ideal_only (optional): Return the ideal Fisher matrix predictions without performing optimization.
            
        Also, we can load precomputed Fisher derivatives via the input_derivatives input. These must be computed for the same reduce_r value!
        """
        assert self.ints_1d, "No 1D optimization is required for these templates!"
        t_init = time.time()

        # Check precision parameters
        assert reduce_r>0, "reduce_r parameter must be positive"
        
        if reduce_r<0.5:
            print("## Caution: very dense r-sampling requested; computation may be very slow") 
        if reduce_r>3:
            print("## Caution: very sparse r-sampling requested; computation may be inaccurate")
        
        # Create radial array. Fine-sampling window [flo, fhi] (dense 2.5*reduce_r spacing); coarse
        # 50*reduce_r elsewhere. Defaults bracket recombination generously (r_star*0.95, r_hor*1.05);
        # r_fine_lo/r_fine_hi override them (e.g. shrink to the actual r-support to cut N_r -- verified
        # negligible for fNL-feat-res, whose support is a ~400-wide spike at r_star). Tails span [1, r_hor+5000].
        flo = self.r_star*0.95 if r_fine_lo is None else r_fine_lo
        fhi = self.r_hor*1.05 if r_fine_hi is None else r_fine_hi
        r_raw = np.asarray(list(np.arange(1,flo,50*reduce_r))+list(np.arange(flo,fhi,2.5*reduce_r))+list(np.arange(fhi,self.r_hor+5000,50*reduce_r)))

        #r_raw = np.asarray(list(np.geomspace(1,999,1000))+list(np.arange(1000,self.r_star*0.95,50*reduce_r))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,5*reduce_r))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r)))
        
        r_init = 0.5*(r_raw[1:]+r_raw[:-1])
        self.quad_weights_1d = r_init**2*np.diff(r_raw)
        r_weights = {}

        # Partition the radial indices if required or read in precomputed points
        if initial_r_points is not None:
            assert split_index is None, "Cannot specify both initial_r_points and index_split"
            assert len(initial_r_points)==len(np.unique(initial_r_points)), "initial_r_points cannot contain repeated points"
            inds = np.asarray([np.where(np.abs(r_init - r) < 1e-10)[0][0] for r in initial_r_points])
            r_init = r_init[inds]
            self.quad_weights_1d = self.quad_weights_1d[inds]
        else:
            if N_split is not None:
                print("Partitioning sampling grid into %d pieces"%(N_split))                
                r_init = r_init[split_index::N_split]
                self.quad_weights_1d = self.quad_weights_1d[split_index::N_split]
        
        # Precompute k-integrals with initial r grid
        self.r_arr = r_init
        self.N_r = len(r_init)
        print("# Computing k integrals with fiducial radial grid")
        self._prepare_templates(ints_1d=True)
        
        # Reorder templates (keeping only those that require 1D optimization)
        ordered_templates = [tem for tem in self.templates if (tem in self.all_templates_1d) or ('neural' in tem)]
        
        # Create list of radial indices in the optimized representation
        inds = []
        inds_init = np.arange(self.N_r)
        
        # Compute all Fisher matrix derivatives of interest
        if verb: print("Computing all Fisher matrix derivatives")
        derivs = self._compute_fisher_derivatives(ordered_templates, verb=verb, input_derivatives=input_derivatives)
        self.derivatives = derivs
        
        # Save ideal Fisher matrices
        if not hasattr(self, 'ideal_fisher'):
            self.ideal_fisher = {}
        for t in ordered_templates:
            self.ideal_fisher[t] = derivs[t][0]

        # Optionally exit and return ideal Fisher matrices
        if ideal_only:
            if verb: print("## Not performing optimization!")
            return self.ideal_fisher
            
        for template in ordered_templates:
            
            if verb: print("\nRunning optimization for template %s"%template)
            
            # Compute Fisher matrix derivative
            init_score, deriv_matrix = derivs[template][0], derivs[template][1]
            if verb: print("Initial score: %.2e"%init_score)
            
            def _compute_score(w_vals, full_score=False):
                """
                Compute the Fisher distance between templates given weights w_vals. This optionally computes the gradients.
                """
                if full_score:
                    return np.sum(G_mat), np.sum(np.outer(w_vals,w_vals)*deriv_matrix[inds][:,inds])
                else:
                    return np.sum(G_mat)
                
            def _test_inds(inds, score_old, w_vals):
                """Test the current set of indices"""
                score = _compute_score(w_vals)
                return score, w_vals
            
            # Check zeroth iteration
            if len(inds)!=0:
                # Compute quadratic weights
                notinds = [i for i in np.arange(self.N_r) if i not in inds]
                inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                
                # Compute score
                score, w_vals = _test_inds(inds, init_score, w_vals)
                if verb: print("Unoptimized relative score: %.2e"%(score/init_score))
            else:
                score = init_score
                w_vals = []
                G_mat = np.eye(0) # dummy variable
                
            # Set up iteration
            if score/init_score >= tolerance and (np.diag(G_mat)>0).all():

                # Define starting indices
                if len(inds)==0:
                    next_ind = np.argsort(np.sum(deriv_matrix,axis=1)**2/np.diag(deriv_matrix))[-1]
                else:
                    next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
                inds.append(next_ind)
                
                # Set-up memory
                w_vals_old = w_vals
                score_old = score
                    
                # Iterate until convergence
                for iit,iteration in enumerate(range(len(inds),self.N_r)):
                    
                    # Define indices  
                    inds[-1] = next_ind
                    notinds = [i for i in np.arange(self.N_r) if i not in inds]
                    
                    # Set-up weights
                    try:
                        inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
                    except:
                        print("Singular matrix; exiting!")
                        inds = inds[:-1]
                        break
                    G_mat = deriv_matrix[notinds][:,notinds]-deriv_matrix[inds][:,notinds].T@inv_deriv@deriv_matrix[inds][:,notinds]
                    
                    # Compute optimal quadratic weights
                    w_vals = (1+np.sum(inv_deriv@deriv_matrix[inds][:,notinds],axis=1))
                      
                    # Compute score
                    score, w_vals = _test_inds(inds, score_old, w_vals)
                    if verb: print("Iteration %d, relative score: %.2e, old score: %.2e"%(iteration, score/init_score, score_old/init_score))
                    
                    # Check for numerical errors
                    if score<0:
                        print("## Score is negative; this indicates a numerical error!")
                        break
                    
                    # Finish if converged
                    if score/init_score < tolerance:
                        break

                    # Check for errors
                    
                    # Update memory when score is accepted
                    w_vals_old = w_vals
                    score_old = score
                    
                    # Compute indices for next iteration
                    next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
                    inds.append(next_ind)
                    
            if len(G_mat)==0:
                raise Exception("Failed to converge; this indicates a bug!")
                
            if verb: print("\nScore threshold met with %d indices"%len(inds))
            w_opt = np.asarray(w_vals.copy())
            
            # Check final Fisher matrix
            score, fish = _compute_score(w_opt, full_score=True)
            if verb: print("Ideal %s Fisher: %.4e (initial), %.4e (optimized). Relative score: %.2e\n"%(template, init_score, fish, score/init_score))

            # Store attributes
            r_weights[template] = w_opt*self.quad_weights_1d[inds]
        
        # Store attributes
        self.r_arr = r_init[inds]
        self.N_r = len(self.r_arr)
        self.r_weights = {}
        for template in ordered_templates:
            # add weights, padding with zeros
            self.r_weights[template] = np.zeros(len(w_opt))
            self.r_weights[template][:len(r_weights[template])] = r_weights[template]

        # Precompute k-space integrals with new radial integration
        if verb: print("Computing k integrals with optimized radial grid")
        self._prepare_templates(ints_1d=True)

        print("\nOptimization complete after %.2f seconds"%(time.time()-t_init))
        
        return self.r_arr, self.r_weights
        
    ### NORMALIZATION
    @_timer_func('fisher')
    def compute_fisher_contribution(self, seed, verb=False):
        """
        This computes the contribution to the Fisher matrix from a single pair of GRF simulations, created internally.
        """
        # Check we have initialized correctly
        if self.ints_1d and (not hasattr(self,'r_arr')):
            raise Exception("Need to supply radial integration points or run optimize_radial_sampling_1d()!")
        
        print("Computing Fisher matrix with seed %d"%seed)
        
        # Initialize output
        fish = np.zeros((self.total_size, self.total_size),dtype='complex')

        # Compute two random realizations with known power spectrum, removing the beam
        if verb: print("# Generating GRFs")
        t_init = time.time()
        a_maps = []
        for ii in range(2):
            if self.ones_mask:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic',lmax=self.lmax, deconvolve_beam=True))
            else:
                # we can't truncate to l<=lmax here, since we need to apply the mask!
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic', deconvolve_beam=True))
        self.timers['fish_grfs'] += time.time()-t_init
        
        # Define Q map code
        def compute_Q3(weighting):
            """
            Assemble and return an array of Q3 maps in real- or harmonic-space, for S^-1 or A^-1 weighting. 

            Schematically Q ~ (A[x]B[y,z]_lm + perms., and we dynamically compute each permutation of (A B)_lm.

            The outputs are either Q_i or S^-1.P.Q_i.
            """
            # Weight maps by S^-1.P or A^-1
            if verb: print("Weighting maps")
            
            t_init = time.time()
            if weighting=='Sinv':
                # Compute S^-1.P.a
                if self.ones_mask:
                    Uinv_a_lms = [np.asarray(self.applySinv(self.base.beam_lm[:,self.base.l_arr<=self.lmax]*a_lm, input_type='harmonic', lmax=self.lmax)[:,self.lminfilt],order='C') for a_lm in a_maps]
                else:
                    Uinv_a_lms = [np.asarray(self.applySinv(self.mask*self.base.to_map(self.base.beam_lm*a_lm), lmax=self.lmax)[:,self.lminfilt],order='C') for a_lm in a_maps]
                self.timers['Sinv'] += time.time()-t_init
            elif weighting=='Ainv':
                # Compute A^-1.a
                if self.ones_mask:
                    Uinv_a_lms = [np.asarray(self.base.applyAinv(a_lm, input_type='harmonic', lmax=self.lmax)[:,self.lminfilt],order='C') for a_lm in a_maps]
                else:
                    Uinv_a_lms = [np.asarray(self.base.applyAinv(a_lm, input_type='harmonic')[:,self.lfilt],order='C') for a_lm in a_maps]
                self.timers['Ainv'] += time.time()-t_init 

            
            # Define output arrays (Q11, Q22)
            Qs = np.zeros((2,self.total_size,1+2*self.pol,np.sum(self.lfilt)),dtype=np.complex128,order='C')
            # Iterate over radial indices
            for radial_index in range(self.N_r):
                if verb: print("Using radial index %d of %d"%(radial_index+1, self.N_r))
                
                # Filter maps
                if (verb and radial_index==0): print("Computing filtered maps")
                if 'f_p4' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[+4] maps")
                    Fp4_maps = self._filter_pair(Uinv_a_lms, 'F_p4', radial_index)   
                if 'f_p3' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[+3] maps")
                    Fp3_maps = self._filter_pair(Uinv_a_lms, 'F_p3', radial_index)   
                if 'f_p2' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[+2] maps")
                    Fp2_maps = self._filter_pair(Uinv_a_lms, 'F_p2', radial_index)   
                if 'f_p1' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[+1] maps")
                    Fp1_maps = self._filter_pair(Uinv_a_lms, 'F_p1', radial_index)   
                if 'f_p0' in self.to_compute:        
                    if (verb and radial_index==0): print("Creating F[0] maps")
                    Fp0_maps = self._filter_pair(Uinv_a_lms, 'F_p0', radial_index)
                    
                if 'f_m1' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[-1] maps")
                    Fm1_maps = self._filter_pair(Uinv_a_lms, 'F_m1', radial_index)   
                if 'f_m2' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[-2] maps")
                    Fm2_maps = self._filter_pair(Uinv_a_lms, 'F_m2', radial_index)
                if 'f_exp' in self.to_compute:
                    if (verb and radial_index==0): print("Creating F[exp] maps")
                    Fexp_maps = self._filter_pair(Uinv_a_lms, 'F_exp', radial_index)
                if 'f_bin' in self.to_compute:
                    if (verb and radial_index==0): print("Creating binned F maps")
                    F_bin_maps = self._filter_pair(Uinv_a_lms, 'F_bin', radial_index)[:,:,None,:]
                    
                if self.neural_inputs is not None:
                    if (verb and radial_index==0): print("Creating neural maps")
                    neural_alphas, neural_betas, neural_gammas = {},{},{}
                    for n in self.neural_inds:
                        neural_alphas[n] = self._filter_pair(Uinv_a_lms, 'neural-alpha-%d'%n, radial_index)
                        neural_betas[n] = self._filter_pair(Uinv_a_lms, 'neural-beta-%d'%n, radial_index)
                        if not self.neural_cyclic[n]:
                            neural_gammas[n] = self._filter_pair(Uinv_a_lms, 'neural-gamma-%d'%n, radial_index)
                
                if radial_index==0:
                    if 'u' in self.to_compute:
                        if verb: print("Creating U maps")
                        U_maps = self._filter_pair(Uinv_a_lms, 'U')
                    
                    if 'v' in self.to_compute:
                        if verb: print("Creating V maps")
                        V_maps = self._filter_pair(Uinv_a_lms, 'V')
                    
                    if 'v-isw' in self.to_compute:
                        if verb: print("Creating ISW V maps")
                        V_isw_maps = self._filter_pair(Uinv_a_lms, 'V-ISW')

                # Compute products (with symmetries)
                q_index = 0
                for t in self.templates:
                    
                    def get_map(i,index):
                        if i==-2:
                            return Fm2_maps[index]
                        elif i==-1:
                            return Fm1_maps[index]
                        elif i==0:
                            return Fp0_maps[index]
                        elif i==1:
                            return Fp1_maps[index]
                        elif i==2:
                            return Fp2_maps[index]
                        elif i==3:
                            return Fp3_maps[index]
                        elif i==4:
                            return Fp4_maps[index]
                        else:
                            raise Exception()

                    def mult(i,j,index):
                        return self.utils.multiply(get_map(i,index),get_map(j,index))

                    def mult_bin(i,j,index):
                        return self.utils.multiply(F_bin_maps[index,i],F_bin_maps[index,j])

                    def transform(prod_map, k):
                        if k==-2:
                            filt = self.flXs_m2
                        elif k==-1:
                            filt = self.flXs_m1
                        elif k==0:
                            filt = self.flXs_p0
                        elif k==1:
                            filt = self.flXs_p1
                        elif k==2:
                            filt = self.flXs_p2
                        elif k==3:
                            filt = self.flXs_p3
                        elif k==4:
                            filt = self.flXs_p4
                        else:
                            raise Exception()
                        return 6./5.*self._transform_maps(prod_map, filt[:,:,[radial_index]], self.r_weights[t][[radial_index]])
    
                    def transform_bin(prod_map, k):
                        filt = np.asarray(self.flXs_bin[:,:,[radial_index],k], order='C')
                        return 6./5.*self._transform_maps(None, filt, self.r_weights[t][[radial_index]], lm_map=prod_map)
                    
                    if t=='fNL-loc':
                        if (verb and radial_index==0): print("Computing Q-derivative for fNL-loc")
    
                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            Qs[index,q_index] += 2.*transform(mult(-1,+2,index),-1)
                            Qs[index,q_index] += transform(mult(-1,-1,index),+2)
                        q_index += 1

                    elif t=='fNL-eq':
                        if (verb and radial_index==0): print("Computing Q-derivative for fNL-eq")

                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            Qs[index,q_index] += 6.*transform(mult(+1,0,index)-mult(-1,+2,index),-1)
                            Qs[index,q_index] += 6.*transform(mult(-1,+1,index)-mult(0,0,index),0)
                            Qs[index,q_index] += 6.*transform(mult(-1,0,index),+1)
                            Qs[index,q_index] -= 3.*transform(mult(-1,-1,index),+2)
                        q_index += 1

                    elif t=='fNL-orth':
                        if (verb and radial_index==0): print("Computing Q-derivative for fNL-orth")
                        
                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            Qs[index,q_index] += 18*transform(mult(+1,0,index)-mult(-1,2,index),-1)
                            Qs[index,q_index] += transform(18*mult(-1,+1,index)-24*mult(0,0,index),0)
                            Qs[index,q_index] += 18*transform(mult(-1,0,index),+1)
                            Qs[index,q_index] -= 9*transform(mult(-1,-1,index),+2)
                        q_index += 1
                    
                    elif t=='fNL-orth2':
                        if (verb and radial_index==0): print("Computing Q-derivative for fNL-orth2")
                        p = 27./(743./(7.*(20*np.pi**2.-193.))-21.)
                        
                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            Qs[index,q_index] += 2./9.*p*transform(-10*mult(1,1,index)+15.*mult(0,2,index)-6*mult(-1,3,index)+mult(-2,4,index),-2)
                            Qs[index,q_index] -= 2./3.*transform(-(9+5*p)*mult(0,1,index)+3*(3+p)*mult(-1,2,index)+2*p*mult(-2,3,index),-1)
                            Qs[index,q_index] += 2./3.*transform(-(9+10.*p)*mult(0,0,index)+(9+5.*p)*mult(-1,1,index)+5*p*mult(-2,2,index),0)
                            Qs[index,q_index] += transform(2./3.*(9+5.*p)*mult(-1,0,index)-40./9.*p*mult(-2,1,index),1)
                            Qs[index,q_index] += transform(-(3+p)*mult(-1,-1,index)+10./3.*p*mult(-2,0,index),2)
                            Qs[index,q_index] -= 4./3.*p*transform(mult(-2,-1,index),3)
                            Qs[index,q_index] += 1./9.*p*transform(mult(-2,-2,index),4)
                        q_index += 1

                    elif t=='fNL-feat-res':
                        # Q-derivative for the genuinely-convergent representation (see the cubic-term
                        # comment above for the Mellin-shift derivation). By the product rule, d/da_lm of a
                        # merged cubic term fnl_sum(A,B,C) (A,B,C legs from the 4-multiset decomposition)
                        # gives, for each DISTINCT leg power v present in {A,B,C}, a term
                        # (count of v in the multiset) * transform(inner=product of the OTHER 2 legs, outer=v)
                        # -- e.g. for {-3,-3,1} (coeff K1=3*c1): outer=-3 has multiplicity 2 (inner=-3,1 leg
                        # product), outer=1 has multiplicity 1 (inner=-3,-3 leg product); matches the existing
                        # fNL-loc pattern (2*transform(mult(-1,2),-1)+transform(mult(-1,-1),2)) for a repeated-
                        # twice multiset. Collecting all 4 multisets' contributions by outer power gives 9
                        # distinct (inner-product, outer-filter) terms, each with its own omega-dependent
                        # coefficient (a linear combination of the c1,c2,c3 group coefficients). Verified
                        # against the fully explicit, unmerged (36-raw-term x 3-outer-choices) form on an
                        # actual SHT-based run to machine precision (that check only validates internal
                        # combinatorial consistency, not the overall scale -- see below).
                        #
                        # Overall normalization: the general Q_lm^X[x,y] formula (eqns.tex) has an explicit
                        # 1/6 prefactor together with "+5 perms", i.e. it sums over all 6 distinguishable
                        # (outer, B-slot-with-x, C-slot-with-y) bijections. This code's "3 outer choices"
                        # construction only distinguishes which leg is outer, not the (x,y) order of the 2
                        # inner legs (mult() is symmetric, so swapping which inner leg gets 'x' vs 'y' is not
                        # separately counted) -- i.e. it implicitly sums only 3 of the 6 bijections, with each
                        # of those 3 representing 2 of the true 6 (the x<->y swap). So an extra 1/3 (= (1/6)
                        # of 6 true bijections, divided by the 2 already-collapsed via mult()'s symmetry, per
                        # 3 counted terms) is needed on top of the "count of occurrences" multiplicities used
                        # below. Confirmed by direct numerical differentiation of the (independently-verified)
                        # cubic numerator w.r.t. a single a_lm mode: this code's Q_lm was exactly 3x the true
                        # derivative before this 1/3 was added (ratio 3.006, both real and imaginary parts).
                        # Compressed exp-sum Q-derivative (sm3/sm1). Convention (derived analytically & verified
                        # vs fNL-loc/eq): Q_lm = 2 * d/dfield[ b3 / A_pix ]  (product rule; NO A_pix; factor 2).
                        # b3/A_pix = (1/3) Re[ C * bracket ], bracket = sum_key gamma_key fnl_sum_c(M_k0,M_k1,M_k2),
                        # C = pref0 kappa^{iw}, fnl_sum_c = sum_{r,node} w_r W_node sum_pix M M M (COMPLEX maps).
                        # Since the maps are complex, Re[cc * M_a M_b M_c] (cc = C gamma_key W_node, complex per-node)
                        # is expanded into REAL triple products over the {re,im} leg maps:
                        #   Re[cc P] = Re(cc) Re(P) - Im(cc) Im(P),
                        #   Re(P) = RRR - RII - IRI - IIR,   Im(P) = RRI + RIR + IRR - III   (R=Mre, I=Mim per slot).
                        # The product rule then uses the REAL _transform_maps on real inner products with the real
                        # re/im legs (leg of a removed Mre_p map = flXs_exp_re[p]; of Mim_p = flXs_exp_im[p]) -- no
                        # complex SHT, no Wirtinger conjugate. Overall factor 2*(1/3) = 2/3; per-node coeff folded
                        # into the inner map so contributions sharing an outer leg merge to one transform.
                        if (verb and radial_index==0): print("Computing Q-derivative for fNL-feat-res (%s)"%self.feat_scheme)
                        # Assembly extracted to _feat_q_deriv (shared by the batched MC-Fisher driver).
                        self._feat_q_deriv(Qs[:,q_index], Fexp_maps, radial_index)
                        q_index += 1

                    elif 'neural' in t:
                        n = int(t.split('-')[1])
                        if verb: print("Computing Q-derivative for neural-%d fNL"%n)
                        
                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            for iterm in range(self.neural_terms[n]):
                                if not self.neural_cyclic[n]:
                                    Qs[index,q_index] += 6./5.*self.neural_weights[n][iterm]*self._transform_maps(self.utils.multiply(neural_alphas[n][iterm][index],neural_betas[n][iterm][index]),self.gamma_lXs[n][iterm][:,:,[radial_index]],self.r_weights[t][[radial_index]])
                                    Qs[index,q_index] += 6./5.*self.neural_weights[n][iterm]*self._transform_maps(self.utils.multiply(neural_betas[n][iterm][index],neural_gammas[n][iterm][index]),self.alpha_lXs[n][iterm][:,:,[radial_index]],self.r_weights[t][[radial_index]])
                                    Qs[index,q_index] += 6./5.*self.neural_weights[n][iterm]*self._transform_maps(self.utils.multiply(neural_gammas[n][iterm][index],neural_alphas[n][iterm][index]),self.beta_lXs[n][iterm][:,:,[radial_index]],self.r_weights[t][[radial_index]])
                                else:
                                    Qs[index,q_index] += 12./5.*self.neural_weights[n][iterm]*self._transform_maps(self.utils.multiply(neural_alphas[n][iterm][index],neural_betas[n][iterm][index]),self.beta_lXs[n][iterm][:,:,[radial_index]],self.r_weights[t][[radial_index]])
                                    Qs[index,q_index] += 6./5.*self.neural_weights[n][iterm]*self._transform_maps(self.utils.multiply(neural_betas[n][iterm][index],neural_betas[n][iterm][index]),self.alpha_lXs[n][iterm][:,:,[radial_index]],self.r_weights[t][[radial_index]])
                        q_index += 1

                    elif t=='binned':
                        if verb and (radial_index==0): print("Computing Q-derivative for the binned bispectrum")
                            
                        # Define all possible pairs of bins
                        unique_pairs12 = np.unique(self.k_triplet_ids[:,[0,1]], axis=0)
                        unique_pairs31 = np.unique(self.k_triplet_ids[:,[2,0]], axis=0)
                        unique_pairs23 = np.unique(self.k_triplet_ids[:,[1,2]], axis=0)
                        unique_pairs = np.unique(np.concatenate([unique_pairs12, unique_pairs23, unique_pairs31]), axis=0)
                        
                        # Iterate over pairs (to avoid recomputing harmonic transforms)
                        for pair in unique_pairs:
                            binA, binB = pair
    
                            # Compute possible third k-bin (symmetrizing over permutations)
                            last_bins = []
                            if pair in unique_pairs12:
                                bin_indices = np.where((self.k_triplet_ids[:,[0,1]]==[binA,binB]).all(1))[0]
                                for bin_index in bin_indices:
                                    binC,bin_id = self.k_triplet_ids[bin_index,[2,3]]
                                    last_bins.append([binC,bin_id,self.bin_degeneracy[bin_index]])
                            if pair in unique_pairs31:
                                bin_indices = np.where((self.k_triplet_ids[:,[2,0]]==[binA,binB]).all(1))[0]
                                for bin_index in bin_indices:
                                    binC,bin_id = self.k_triplet_ids[bin_index,[1,3]]
                                    last_bins.append([binC,bin_id,self.bin_degeneracy[bin_index]])
                            if pair in unique_pairs23:
                                bin_indices = np.where((self.k_triplet_ids[:,[1,2]]==[binA,binB]).all(1))[0]
                                for bin_index in bin_indices:
                                    binC,bin_id = self.k_triplet_ids[bin_index,[0,3]]
                                    last_bins.append([binC,bin_id,self.bin_degeneracy[bin_index]])
                            
                            # Count up how many times each pair occurred
                            last_bins, counts = np.unique(last_bins,axis=0,return_counts=True)
                            
                            # Iterate over both the 11 and 22 pieces and compute output
                            for index in [0,1]:
                                # Compute harmonic transforms of the pairwise products
                                prod_map_lm = np.asarray(self.base.to_lm_vec(mult_bin(binA, binB, index), lmax=self.lmax)[0,self.lminfilt], order='C')
                                
                                # Assemble output
                                self.utils.assemble_binned_bispectrum_Q(last_bins.astype(np.int32), counts.astype(np.int32), prod_map_lm, self.flXs_bin, radial_index, self.r_weights[t], Qs[index], q_index)
                            
                        # Update output index
                        q_index += self.n_binned

                    elif t=='isw-lensing':
                        if radial_index >0:
                            q_index += 1
                            continue
                        
                        if (verb and radial_index==0): print("Computing Q-derivative for isw-lensing")
                        
                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            
                            ## First term
                            # X = T
                            input_map = 2.*np.real(V_maps[index][0]*V_isw_maps[index].conjugate())
                            Qs[index,q_index,0] = self.base.to_lm(input_map[None],lmax=self.lmax)[0,self.lminfilt]
                            if self.pol:
                                # X = E, B
                                input_map = V_maps[index][1]*V_isw_maps[index].conjugate() - V_maps[index][2]*V_isw_maps[index]
                                output_lm = 0.5*self.base.to_lm_spin(input_map[None],input_map[None].conjugate(),spin=2,lmax=self.lmax)[:,self.lminfilt]
                                Qs[index,q_index,1] = output_lm[0] + output_lm[1]
                                Qs[index,q_index,2] = -1.0j*(output_lm[0] - output_lm[1])
                                    
                            ## Second term
                            # Y = T
                            pref = np.sqrt(self.ls*(self.ls+1.))
                            input_map = U_maps[index][0]*V_isw_maps[index] 
                            out_lm = self.base.to_lm_spin(input_map.conjugate(), input_map, spin=1, lmax=self.lmax)[:,self.lminfilt].copy()
                            Qs[index,q_index,0] += -self.C_lens_weight['TT'][self.ls]*pref*(out_lm[0]-out_lm[1])
                            if self.pol:
                                Qs[index,q_index,1] += -self.C_lens_weight['TE'][self.ls]*pref*(out_lm[0]-out_lm[1])
    
                                # Y = E, B
                                prefP = np.sqrt((self.ls+2.)*(self.ls-1.))
                                prefM = np.sqrt((self.ls-2.)*(self.ls+3.)) 
    
                                # Compute spin-1 transforms
                                input_map = (U_maps[index][1] + 1.0j*U_maps[index][2])*V_isw_maps[index]
                                out_lm1 = self.base.to_lm_spin(input_map, input_map.conjugate(), spin=1, lmax=self.lmax)[:,self.lminfilt]
                                
                                # Compute spin-3 transforms
                                input_map = -(U_maps[index][1] + 1.0j*U_maps[index][2])*V_isw_maps[index].conjugate()                        
                                out_lm3 = self.base.to_lm_spin(input_map, input_map.conjugate(), spin=3, lmax=self.lmax)[:,self.lminfilt]
    
                                # Assemble output
                                diff_lm = prefP*(out_lm1[0]-out_lm1[1]) + prefM*(out_lm3[0]-out_lm3[1])
                                sum_lm = prefP*(out_lm1[0]+out_lm1[1]) + prefM*(out_lm3[0]+out_lm3[1])
                                Qs[index,q_index,0] += 0.5*self.C_lens_weight['TE'][self.ls]*diff_lm
                                Qs[index,q_index,1] += 0.5*self.C_lens_weight['EE'][self.ls]*diff_lm
                                Qs[index,q_index,2] += -1.0j*0.5*self.C_lens_weight['BB'][self.ls]*sum_lm
                                del out_lm1, out_lm3, diff_lm, sum_lm, prefP, prefM
                            
                            ## Third term
                            input_map = U_maps[index][0]*V_maps[index][0].conjugate()
                            if not self.pol:
                                input_map = U_maps[index][0]*V_maps[index][0].conjugate()
                                Qs[index,q_index,0] += -self.C_Tphi[self.ls]*np.sqrt(self.ls*(self.ls+1.))*np.sum(np.array([1,-1])[:,None]*self.base.to_lm_spin(input_map, input_map.conjugate(), spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                            else: 
                                input_map = 2.*U_maps[index][0]*V_maps[index][0].conjugate()
                                input_map += (U_maps[index][1] + 1.0j*U_maps[index][2])*V_maps[index][1].conjugate() + (-U_maps[index][1].conjugate() + 1.0j*U_maps[index][2].conjugate())*V_maps[index][2]
                                out_lm = np.sqrt(self.ls*(self.ls+1.))*np.sum(np.array([1,-1])[:,None]*self.base.to_lm_spin(input_map, input_map.conjugate(), spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                                Qs[index,q_index,0] += -0.5*self.C_Tphi[self.ls]*out_lm
                                Qs[index,q_index,1] += -0.5*self.C_Ephi[self.ls]*out_lm
                                del out_lm
                        q_index += 1
                    
            if weighting=='Ainv' and verb: print("Applying S^-1 weighting to output")
            for findex in range(2):
                self._weight_Q_maps(Qs[findex], weighting)
            return Qs.reshape(2,self.total_size,-1)

        # Compute Q3 maps
        if verb: print("\n# Computing Q3 map for S^-1 weighting")
        Q3_Sinv = compute_Q3('Sinv')
        if verb: print("\n# Computing Q3 map for A^-1 weighting")
        Q3_Ainv = compute_Q3('Ainv')

        # Assemble Fisher matrix
        if verb: print("\n# Assembling Fisher matrix\n")

        # Compute Fisher matrix as an outer product
        fish = self._assemble_fish(Q3_Sinv, Q3_Ainv, sym=False)
        if verb: print("\n# Fisher matrix contribution %d computed successfully!"%seed)

        return fish

    @_timer_func('fisher')
    def compute_fisher_contribution_feat_batch(self, seed, feat_params_list, verb=False):
        """BATCHED MC Fisher contribution (single GRF-pair `seed`) for a SET of (omega,kappa) sharing the
        current node + radial grid. The (omega,kappa)-INDEPENDENT work -- the two GRFs, their S^-1/A^-1
        weighting, and the per-radial-index filtered feature maps (the dominant SHT cost) -- is done ONCE and
        reused; only the cheap per-param Q-derivative assembly (_feat_q_deriv) is looped over the batch.

        Returns fish (nparam, nparam): the [i,i] diagonal is param i's own Fisher contribution (equal to the
        single-param compute_fisher_contribution(seed) to machine precision); off-diagonals are the cross-
        Fisher between the different feature templates (computed for free from the shared maps).

        Assumes 'fNL-feat-res' is the (only) template. feat_params_list: list of (omega,kappa) or dicts."""
        assert self.templates == ['fNL-feat-res'], "compute_fisher_contribution_feat_batch supports a lone fNL-feat-res template"
        if self.ints_1d and (not hasattr(self, 'r_arr')):
            raise Exception("Need radial integration points (prepare a common r-grid) first!")
        params = [p if isinstance(p, (tuple, list)) else (p['omega'], p['kres_cs']) for p in feat_params_list]
        nparam = len(params)
        print("Computing BATCHED feat MC Fisher (seed %d, %d params)" % (seed, nparam))

        # two GRFs (shared across all params)
        t_init = time.time()
        a_maps = []
        for ii in range(2):
            if self.ones_mask:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic', lmax=self.lmax, deconvolve_beam=True))
            else:
                a_maps.append(self.base.generate_data(seed=seed+int((1+ii)*1e9), output_type='harmonic', deconvolve_beam=True))
        self.timers['fish_grfs'] += time.time()-t_init

        def compute_Q3(weighting):
            t_init = time.time()
            if weighting == 'Sinv':
                if self.ones_mask:
                    Uinv_a_lms = [np.asarray(self.applySinv(self.base.beam_lm[:,self.base.l_arr<=self.lmax]*a_lm, input_type='harmonic', lmax=self.lmax)[:,self.lminfilt], order='C') for a_lm in a_maps]
                else:
                    Uinv_a_lms = [np.asarray(self.applySinv(self.mask*self.base.to_map(self.base.beam_lm*a_lm), lmax=self.lmax)[:,self.lminfilt], order='C') for a_lm in a_maps]
                self.timers['Sinv'] += time.time()-t_init
            elif weighting == 'Ainv':
                if self.ones_mask:
                    Uinv_a_lms = [np.asarray(self.base.applyAinv(a_lm, input_type='harmonic', lmax=self.lmax)[:,self.lminfilt], order='C') for a_lm in a_maps]
                else:
                    Uinv_a_lms = [np.asarray(self.base.applyAinv(a_lm, input_type='harmonic')[:,self.lfilt], order='C') for a_lm in a_maps]
                self.timers['Ainv'] += time.time()-t_init

            # nparam observables (one per feat param); Q11/Q22 (2) x nparam x npol x nlm
            Qs = np.zeros((2, nparam, 1+2*self.pol, np.sum(self.lfilt)), dtype=np.complex128, order='C')
            for radial_index in range(self.N_r):
                if verb and radial_index == 0: print("Creating F[exp] maps (shared across params)")
                Fexp_maps = self._filter_pair(Uinv_a_lms, 'F_exp', radial_index)   # SHARED filtering SHTs
                # SHARED outer SHTs: each distinct 2-leg product is transformed once, recombined per param
                self._feat_q_batch(Qs, Fexp_maps, radial_index, params)
            # S^-1 / m-weight each observable (explicit loop over nparam; _weight_Q_maps only spans total_size)
            if weighting == 'Ainv' and verb: print("Applying S^-1 weighting to output")
            for findex in range(2):
                for ip in range(nparam):
                    if weighting == 'Ainv':
                        full_Q = np.zeros((1+2*self.pol, len(self.lminfilt)), dtype=np.complex128)
                        full_Q[:, self.lminfilt] = self.beam_lm*Qs[findex, ip]
                        t0 = time.time()
                        if self.ones_mask:
                            Qs[findex, ip] = self.applySinv(full_Q, input_type='harmonic', lmax=self.lmax)[:, self.lminfilt]
                        else:
                            Qs[findex, ip] = self.applySinv(self.mask*self.base.to_map(full_Q, lmax=self.lmax), lmax=self.lmax)[:, self.lminfilt]
                        self.timers['Sinv'] += time.time()-t0
                    elif weighting == 'Sinv':
                        Qs[findex, ip] = self.m_weight*Qs[findex, ip]
            return Qs.reshape(2, nparam, -1)

        if verb: print("\n# Computing Q3 map for S^-1 weighting")
        Q3_Sinv = compute_Q3('Sinv')
        if verb: print("\n# Computing Q3 map for A^-1 weighting")
        Q3_Ainv = compute_Q3('Ainv')
        fish = self._assemble_fish(Q3_Sinv, Q3_Ainv, sym=False)
        if verb: print("\n# Batched feat Fisher contribution %d computed successfully!" % seed)
        return fish

    def compute_fisher(self, N_it, verb=False):
        """
        Compute the Fisher matrix using N_it realizations. These are run in serial (since the code is already parallelized).
        
        For high-dimensional problems, it is usually preferred to split the computation across a cluster with MPI, calling compute_fisher_contribution for each instead of this function.
        """
        # Initialize output
        fish = np.zeros((self.total_size,self.total_size))
        
        # Iterate over N_it seeds
        for seed in range(N_it):
            print("Computing Fisher contribution %d of %d"%(seed+1,N_it))
            fish += self.compute_fisher_contribution(seed, verb=verb*(seed==0))/N_it
        
        # Store matrix in class attributes
        self.fish = fish
        self.inv_fish = np.linalg.inv(fish)

        return fish
    
    ### WRAPPER
    def Bl_full(self, data, fish=[], include_linear_term=True, verb=False, input_type='map'):
        """
        Compute the quasi-optimal bispectrum estimator. This is a wrapper of Bl_numerator, including the Fisher matrix multiplication.
        
        The code either uses pre-computed Fisher matrices or reads them in on input. 
        
        We can also optionally switch off the linear terms.
        """
        if verb: print("")

        if len(fish)!=0:
            self.fish = fish
            self.inv_fish = np.linalg.inv(fish)

        if not hasattr(self,'inv_fish'):
            raise Exception("Need to compute Fisher matrix first!")
        
        # Compute numerator
        Bl_num = self.Bl_numerator(data, verb=verb, include_linear_term=include_linear_term, input_type=input_type)

        # Apply normalization
        Bl_out = np.matmul(self.inv_fish,Bl_num)

        # Create output dictionary
        Bl_dict = {}
        index = 0
        # Iterate over fields
        for t in self.templates:
            Bl_dict[t] = Bl_out[index]
            index += 1
            
        return Bl_dict