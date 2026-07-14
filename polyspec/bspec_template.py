### Code for binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2026)
## This module contains the bispectrum template estimation code

import numpy as np
import time
from scipy.special import gamma, p_roots, lpmn
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
            assert 'u_arr' in self.feat_params, "Must specify u_arr to use resonant templates!"
            assert 'omega' in self.feat_params, "Must specify omega to use resonant templates!"
            self.u_arr = self.feat_params['u_arr']
            self.N_u = len(self.u_arr)
            self.omega = self.feat_params['omega']

            # k-powers (x_j(k) ~ k^{kpow}) needed for the exact separable form of B_res: p_2(k)=k^{-2}e^{-ku}, p_3(k)=k^{-3}e^{-ku}
            self.feat_kpows = [-2,-3]

            # Common Mellin weight, W(u) = u^{i*omega-1}/Gamma(i*omega); the exact B_res form collapses onto this single weight,
            # with the extra 1/u, 1/u^3 factors of the u^{i*omega-2}, u^{i*omega-4} pieces applied explicitly at the summation stage.
            log_u = np.log(self.u_arr)
            dlnu = np.zeros(self.N_u, dtype=np.float64)
            dlnu[:-1] += 0.5*np.diff(log_u)
            dlnu[1:] += 0.5*np.diff(log_u)
            du = dlnu*self.u_arr
            self.u_weights = np.asarray(du*self.u_arr**(1j*self.omega-1.)/gamma(1j*self.omega), dtype=np.complex128, order='C')
        
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

            print("Computing exponential f_l^X(r,u) integrals")
            #TODO: check amplitude dependence here!
            self.flXs_exp = {}
            for kpow in self.feat_kpows:
                self.flXs_exp[kpow] = np.zeros((self.lmax+1,1+2*self.pol,self.N_r,self.N_u),dtype=np.float64,order='C')
                q_integral_exp(self.k_arr, Pzeta_arr, self.u_arr, -kpow/3., self.feat_params['kres_cs'], self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.flXs_exp[kpow])
        
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
            return {kpow: np.asarray([[self._compute_weighted_map_single(imap, np.asarray(self.flXs_exp[kpow][:,:,:,iu],order='C'), radial_index) for iu in range(self.N_u)] for imap in input_maps],order='C') for kpow in self.feat_kpows}

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
            output['f_exp'] = {}
            for kpow in self.feat_kpows:
                flXs_exp_flat = np.asarray(self.flXs_exp[kpow].reshape(self.flXs_exp[kpow].shape[0], self.flXs_exp[kpow].shape[1], -1), order='C')
                output['f_exp'][kpow] = self._compute_weighted_maps(input_map, flXs_exp_flat).reshape(self.N_r, self.N_u, self.base.Npix)
            
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
                #TODO: update to the exact 5-term B_res form (currently uses the old single-term k^{-2} ansatz)
                deriv_matrix = np.asarray(fisher_deriv_fNL_feat_res(self.flXs_exp[-2], self.quad_weights_1d, self.u_weights, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'),
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
            
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
                # fNL-feat-res template: exact separable form of B_res, collapsed onto a single u-integral
                # with common weight W(u) = u^{i*omega-1}/Gamma(i*omega); see eqns.tex for the derivation.
                # B_res/(As^2 Ares) = -(omega+3i)*pi^4*omega^2*cosh(pi*omega/2)*kappa^{i*omega}
                #                     * int du W(u) [ p3(k1)p3(k2)p3(k3)*(i*omega-2)/u^3
                #                                     + (p2(k1)p2(k2)p3(k3) + 2 perm)/u
                #                                     + p2(k1)p2(k2)p2(k3) ] + c.c.
                # with p_n(k) = k^{-n} e^{-ku}.
                print("Computing fNL-feat-res template")

                t_init = time.time()
                omega = self.omega
                kappa = self.feat_params['kres_cs']
                prefactor = -(omega+3j)*np.pi**4*omega**2*np.cosh(np.pi*omega/2)*kappa**(1j*omega)
                u_weights_m1 = self.u_weights/self.u_arr
                u_weights_m3 = self.u_weights/self.u_arr**3*(1j*omega-2)

                f2, f3 = proc_maps['f_exp'][-2], proc_maps['f_exp'][-3]
                # All 3 legs use the *same* data map (just filtered differently), so fnl_sum_2d_complex's
                # pointwise product is commutative under argument-slot permutation: the 3 "2 perm." terms
                # below are numerically identical, so we use a single call with multiplicity 3 (verified
                # against the explicit unsimplified 3-call form to machine precision).
                # p3(k1)p3(k2)p3(k3)
                bracket  = self.utils.fnl_sum_2d_complex(self.r_weights[t], u_weights_m3, f3, f3, f3)
                # p2(k1)p2(k2)p3(k3) + 2 perm.
                bracket += 3*self.utils.fnl_sum_2d_complex(self.r_weights[t], u_weights_m1, f2, f2, f3)
                # p2(k1)p2(k2)p2(k3)
                bracket += self.utils.fnl_sum_2d_complex(self.r_weights[t], self.u_weights, f2, f2, f2)

                summ = prefactor*bracket
                b3_num[index] = 1./3.*summ.real*self.base.A_pix
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
                        # Linear term: replace 2 of the 3 legs with simulations, summed over the 3 cyclic
                        # rotations of each of the 5 cubic-term pieces. fnl_sum_2d_complex is commutative in
                        # its 3 map arguments (verified numerically), so many of the resulting 15 raw
                        # (piece, rotation) terms coincide -- not only within a piece (2 legs sharing a
                        # k-power filter merge), but *across* the 3 mixed p2p2p3-type pieces too, since they
                        # all share the same u-weight and reduce to just 2 distinct (data,sim) multisets.
                        # This collapses to 4 distinct calls; verified to match the fully explicit,
                        # unmerged 15-call sum to machine precision.
                        t_init = time.time()
                        omega = self.omega
                        kappa = self.feat_params['kres_cs']
                        prefactor = -(omega+3j)*np.pi**4*omega**2*np.cosh(np.pi*omega/2)*kappa**(1j*omega)
                        u_weights_m1 = self.u_weights/self.u_arr
                        u_weights_m3 = self.u_weights/self.u_arr**3*(1j*omega-2)

                        f2, f3 = proc_maps['f_exp'][-2], proc_maps['f_exp'][-3]
                        sf2, sf3 = this_proc_maps['f_exp'][-2], this_proc_maps['f_exp'][-3]

                        # p3(k1)p3(k2)p3(k3): all 3 legs identical -> single call, x3
                        bracket  = 3*self.utils.fnl_sum_2d_complex(self.r_weights[t], u_weights_m3, f3, sf3, sf3)
                        # p2p2p3 + 2 perm.: data in a p2 slot (x6, from all 3 pieces' 2 p2-slots each) or the p3 slot (x3)
                        bracket += 6*self.utils.fnl_sum_2d_complex(self.r_weights[t], u_weights_m1, f2, sf2, sf3)
                        bracket += 3*self.utils.fnl_sum_2d_complex(self.r_weights[t], u_weights_m1, sf2, sf2, f3)
                        # p2(k1)p2(k2)p2(k3): all 3 legs identical -> single call, x3
                        bracket += 3*self.utils.fnl_sum_2d_complex(self.r_weights[t], self.u_weights, f2, sf2, sf2)

                        summ = prefactor*bracket
                        b1_num[index] += -1./3.*summ.real*self.base.A_pix/self.N_it
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

    ### OPTIMIZATION
    @_timer_func('optimization')
    def optimize_radial_sampling_1d(self, reduce_r=1, tolerance=1e-3, N_split=None, split_index=None, initial_r_points=None, verb=False, ideal_only=False, input_derivatives={}):
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
        
        # Create radial array
        r_raw = np.asarray(list(np.arange(1,self.r_star*0.95,50*reduce_r))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,2.5*reduce_r))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r)))

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
                        if (verb and radial_index==0): print("Computing Q-derivative for fNL-feat-res")
                        #TODO: update to the exact 5-term B_res form (currently uses the old single-term k^{-2} ansatz)

                        # Iterate over both the 11 and 22 pieces
                        for index in [0,1]:
                            for iu in range(self.N_u):
                                prod_map = self.utils.multiply(Fexp_maps[-2][index,iu], Fexp_maps[-2][index,iu])
                                filt = np.asarray(self.flXs_exp[-2][:,:,[radial_index],iu], order='C')
                                Qs[index,q_index] += 2.*self.u_weights[iu].real*self._transform_maps(prod_map, filt, self.r_weights[t][[radial_index]])
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