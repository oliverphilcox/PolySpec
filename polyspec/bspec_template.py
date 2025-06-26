### Code for binned/template polyspectrum estimation on the full-sky. Author: Oliver Philcox (2022-2025)
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
    - base: PolyBin class
    - mask: HEALPix mask applied to data. We can optionally specify a vector of three masks for [T, Q, U].
    - applySinv: function which returns S^-1 ~ P^dag Cov^{-1} in harmonic space, when applied to a given input map, where P = Mask * Beam.
    - templates: types of templates to compute e.g. [fNL-loc, fNL-eq, isw-lensing, neural]
    - k_arr, Tl_arr: k-array, plus T- and (optionally) E-mode transfer functions for all ell. Required for all primordial templates.
    - lmin, lmax: minimum/maximum ell (inclusive)
    - ns, As, k_pivot: primordial power spectrum parameters
    - r_values, r_weights: radial sampling points and weights for 1-dimensional integrals
    - C_Tphi, C_Ephi: cross spectrum of temperature/polarization and lensing  [C^Tphi_0, C^Tphi_1, etc.]. Required if 'isw-lensing' is in templates.
    - C_lens_weight: dictionary of lensed power spectra (TT, TE, etc.). Required if 'isw-lensing' is in templates.
    - r_star, r_hor: Comoving distance to last-scattering and the horizon (default: Planck 2018 values).
    - neural_input: Input neural-network bispectrum templates. These must take the form (weights, alpha, beta, [gamma]), where alpha/beta/gamma are functions of k and i, for i = 1 ... length(weights).
    """
    def __init__(self, base, mask, applySinv, templates, lmin, lmax,  k_arr=[], Tl_arr=[], r_arr=[], ns=0.96, As=2.1e-9, k_pivot=0.05, r_values = [], r_weights = {}, C_Tphi=[], C_Ephi=[], C_lens_weight = {}, r_star=None, r_hor=None, neural_inputs=None):
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
        if not type(mask)==float or type(mask)==int:
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
                if (t in self.all_templates_1d):
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
        self.all_templates_1d = ['fNL-loc','fNL-eq','neural']
        self.all_templates = self.all_templates_1d+['isw-lensing']
        ii = 0
        for t in templates:
            ii += 1
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
            self.to_compute.append(['p','q'])
            self.ints_1d = True
        if 'fNL-eq' in templates:
            self.to_compute.append(['p','q','r1','r2'])
            self.ints_1d = True
        if 'neural' in templates:
            # Check inputs
            assert self.neural_inputs is not None, "Must supply neural network inputs!"
            assert len(self.neural_inputs) in [3, 4], "Neural network inputs must be of the form (weights, alpha, beta) or (weights, alpha, beta, gamma)"
            self.neural_weights = np.asarray(self.neural_inputs[0],dtype=np.float64,order='C')
            self.neural_terms = len(self.neural_weights)
            if len(self.neural_inputs) == 3:
                self.neural_cyclic = True
                print("Using a cyclic neural network input with %d terms"%self.neural_terms)
            else:
                self.neural_cyclic = False
                print("Using a neural network input with %d terms"%self.neural_terms)
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
        self.to_compute = np.unique(np.concatenate(self.to_compute))
        
        # Create filtering for minimum ls
        self.lminfilt = self.base.l_arr[self.base.l_arr<=self.lmax]>=self.lmin
        
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

        This fills arrays such as plXs and qlXs arrays. Note that values outside the desired ell & field range will be set to zero.
        """
        # Print dimensions of k and r
        print("N_k: %d"%len(self.k_arr))
        if ints_1d: print("N_r: %d"%self.N_r)
        
        # Clear saved quantities, if necessary
        if hasattr(self, 't0_num'): delattr(self, 't0_num')
        if ints_1d:
            if hasattr(self, 'plXs'): delattr(self, 'plXs')
            if hasattr(self, 'qlXs'): delattr(self, 'qlXs')
            if hasattr(self, 'alpha_lXs'): delattr(self, 'alpha_lXs')
            if hasattr(self, 'beta_lXs'): delattr(self, 'beta_lXs')
            if hasattr(self, 'gamma_lXs'): delattr(self, 'gamma_lXs')
            
        # Precompute all spherical Bessel functions on a regular grid
        print("Precomputing Bessel functions")
        max_kr = max(self.k_arr)*max(self.r_arr)
        
        x_arr = list(np.arange(0,self.lmax*2,0.01))+list(np.arange(self.lmax*2,min(max_kr*1.01,self.lmax*100),0.1))
        if max_kr>100*self.lmax:
            x_arr += list(np.linspace(self.lmax*100,max_kr*1.01,1000))
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

        if 'q' in self.to_compute and ints_1d:
            
            # Compute q integrals in Cython
            print("Computing q_l^X(r) integrals")
            self.qlXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            q_integral(self.k_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.qlXs)
            
        if 'p' in self.to_compute and ints_1d:
            
            # Compute p integrals in Cython
            print("Computing p_l^X(r) integrals")
            self.plXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral(self.k_arr, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.plXs)
            
        if 'r1' in self.to_compute and ints_1d:
            
            # Compute r integrals in Cython
            print("Computing r1_l^X(r) integrals")
            self.r1lXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, 1./3., self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.r1lXs)
            
        if 'r2' in self.to_compute and ints_1d:
            
            # Compute r2 integrals in Cython
            print("Computing r2_l^X(r) integrals")
            self.r2lXs = np.zeros((self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            p_integral_general(self.k_arr, Pzeta_arr, 2./3., self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.r2lXs)
        
        if self.neural_inputs is not None:
            
            # Compute neural integrals in Cython
            print("Computing f_l^X[alpha](r) integrals")
            self.alpha_lXs = np.zeros((self.neural_terms,self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            self.beta_lXs  = np.zeros((self.neural_terms,self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            if not self.neural_cyclic:
                self.gamma_lXs = np.zeros((self.neural_terms,self.lmax+1,1+2*self.pol,self.N_r),dtype=np.float64,order='C')
            for i in range(self.neural_terms):
                alphas = np.asarray(np.ravel([self.neural_inputs[1](np.float32(kk),i) for kk in self.k_arr]), dtype=np.float64)
                betas = np.asarray(np.ravel([self.neural_inputs[2](np.float32(kk),i) for kk in self.k_arr]), dtype=np.float64)
                f_integral(self.k_arr, alphas, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.alpha_lXs[i])
                f_integral(self.k_arr, betas, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.beta_lXs[i])
                if not self.neural_cyclic:
                    gammas = np.asarray(np.ravel([self.neural_inputs[3](np.float32(kk),i) for kk in self.k_arr]), dtype=np.float64)
                    f_integral(self.k_arr, gammas, Pzeta_arr, self.Tl_arr, jlkr, self.lmin, self.lmax, self.base.nthreads, self.gamma_lXs[i])
            
        if ints_1d: del jlkr
        
        # Define Cython utility class
        self.utils = fNL_utils(self.base.nthreads, self.N_r, self.base.l_arr.astype(np.int32),self.base.m_arr.astype(np.int32),
                                self.ls.astype(np.int32), self.ms.astype(np.int32))            
        
        print("Precomputation complete")
        
    ### MAP TRANSFORMATIONS
    @_timer_func('map_transforms')
    def _compute_weighted_maps(self, h_lm_filt, flX_arr, spin=0):
        """
        Compute [Sum_lm {}_sY_lm(i) f_l^X(i) h_lm^X] maps for each sampling point i, given the relevant weightings. These are used in the bispectrum numerators and Fisher matrices.
        """
        if not (hasattr(self,'r_arr') or hasattr(self,'rtau_arr')):
            raise Exception("Radial arrays have not been computed!")
        
        # Sum over polarizations (only filling non-zero elements)
        summ = np.zeros((len(flX_arr[0,0]),len(self.lminfilt)),order='C',dtype=np.complex128)
        summ[:,self.lminfilt] = self.utils.apply_fl_weights(flX_arr, h_lm_filt, 1.)
        
        # Compute SHTs 
        if spin!=0:
            return self.base.to_map_vec(summ, output_spin=spin, lmax=self.lmax)[0]
        else:
            return self.base.to_map_vec(summ, output_spin=spin, lmax=self.lmax)

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

    def _filter_pair(self, input_maps, filtering = 'Q'):
        """Compute the processed field with a given filtering for a pair of input maps."""
        
        if filtering=='P':
            return np.asarray([self._compute_weighted_maps(imap, self.plXs) for imap in input_maps],order='C')     
        
        elif filtering=='Q':
            return np.asarray([self._compute_weighted_maps(imap, self.qlXs) for imap in input_maps],order='C')     
        
        elif filtering=='R1':
            return np.asarray([self._compute_weighted_maps(imap, self.r1lXs) for imap in input_maps],order='C')     
        
        elif filtering=='R2':
            return np.asarray([self._compute_weighted_maps(imap, self.r2lXs) for imap in input_maps],order='C')     
        
        elif filtering=='U':
            return np.asarray([self._compute_lensing_U_map(imap) for imap in input_maps], order='C')        
            
        elif filtering=='V':
            return np.asarray([self._compute_lensing_V_map(imap) for imap in input_maps], order='C')        
        
        elif filtering=='V-ISW':
            return np.asarray([self._compute_isw_V_map(imap) for imap in input_maps], order='C')        
        
        elif filtering=='neural-alpha':
            return np.asarray([[self._compute_weighted_maps(imap, self.alpha_lXs[i]) for imap in input_maps] for i in range(self.neural_terms)], order='C')
        
        elif filtering=='neural-beta':
            return np.asarray([[self._compute_weighted_maps(imap, self.beta_lXs[i]) for imap in input_maps] for i in range(self.neural_terms)], order='C')

        elif filtering=='neural-gamma':
            return np.asarray([[self._compute_weighted_maps(imap, self.gamma_lXs[i]) for imap in input_maps] for i in range(self.neural_terms)], order='C')

        else:
            raise Exception("Filtering %s is not implemented!"%filtering)

    def _apply_all_filters(self, input_map):
        """Compute the processed fields with all relevant filterings for a single input map."""
        
        # Output array
        output = {}
        
        # Compute local maps
        if 'p' in self.to_compute:
            output['p'] = self._compute_weighted_maps(input_map, self.plXs)
              
        if 'q' in self.to_compute:
            output['q'] = self._compute_weighted_maps(input_map, self.qlXs)
        
        # Compute equilateral maps
        if 'r1' in self.to_compute:
            output['r1'] = self._compute_weighted_maps(input_map, self.r1lXs)
            
        if 'r2' in self.to_compute:
            output['r2'] = self._compute_weighted_maps(input_map, self.r2lXs) 
              
        # Compute lensing maps
        if 'u' in self.to_compute:
            output['u'] = self._compute_lensing_U_map(input_map)        
            
        if 'v' in self.to_compute:
            output['v'] = self._compute_lensing_V_map(input_map)
            
        if 'v-isw' in self.to_compute:
            output['v-isw'] = self._compute_isw_V_map(input_map)
            
        if self.neural_inputs is not None:
            output['neural-alpha'] = np.zeros((self.neural_terms, len(self.alpha_lXs[0,0,0]),self.base.Npix),order='C',dtype=np.float64)
            output['neural-beta'] = np.zeros((self.neural_terms, len(self.beta_lXs[0,0,0]),self.base.Npix),order='C',dtype=np.float64)
            for i in range(self.neural_terms):
                output['neural-alpha'][i] = self._compute_weighted_maps(input_map, self.alpha_lXs[i])
                output['neural-beta'][i] = self._compute_weighted_maps(input_map, self.beta_lXs[i])
            if not self.neural_cyclic:
                output['neural-gamma'] = np.zeros((self.neural_terms, len(self.gamma_lXs[0,0,0]),self.base.Npix),order='C',dtype=np.float64)
                for i in range(self.neural_terms):
                    output['neural-gamma'][i] = self._compute_weighted_maps(input_map, self.gamma_lXs[i])
            
        return output
    
    ### SIMULATION FUNCTIONS
    def _process_sim(self, sim, input_type='map'):
        """
        Process a single input simulation. This is used for the linear term of the bispectrum estimator.
        
        We return a set of weighted maps for this simulation (filtered by e.g. p_l^X).
        """
        # Transform to Fourier space and normalize appropriately
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
        
        for index in range(len(self.templates)):
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
    def _transform_maps(self, map12, flXs, weights, spin=0):
        """Compute Sum_i w_i M_LM f^X_L(i) for real-space map M(n). We optionally average over spins."""
        output = np.zeros((1+2*self.pol,np.sum(self.lfilt)),dtype='complex')
        if spin==0:
            lm_map = np.asarray(self.base.to_lm_vec(map12,lmax=self.lmax)[:,self.lminfilt],order='C')
            return self.utils.radial_sum(lm_map, weights, flXs)
            # return np.sum(self.base.to_lm_vec(map12,lmax=self.lmax).T[self.lminfilt,None,:]*flXs*weights,axis=2).T
        elif spin==1:
            lm_map = np.asarray(self.base.to_lm_vec([map12,map12.conjugate()],spin=1,lmax=self.lmax)[:,:,self.lminfilt],order='C')
            return self.utils.radial_sum_spin1(lm_map, weights, flXs)
            # return 0.5*np.sum((np.array([1,-1])[:,None,None]*self.base.to_lm_vec([map123,map123.conjugate()],spin=1,lmax=self.lmax)).sum(axis=0).T[self.lminfilt,None,:]*flXs*weights,axis=2).T
        else:
            raise Exception(f"Wrong spin s = {spin}!")

    def _compute_fisher_derivatives(self, templates, N_fish_optim=None, verb=False):
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
                
            if template=='fNL-loc':
                if verb: print("\tComputing fNL-loc Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_loc(self.plXs, self.qlXs, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
               
            elif template=='fNL-eq':
                if verb: print("\tComputing fNL-eq Fisher matrix derivative exactly")
                deriv_matrix = np.asarray(fisher_deriv_fNL_eq(self.plXs, self.qlXs, self.r1lXs, self.r2lXs, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                    legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                
            elif template=='neural':
                if verb: print("\tComputing neural Fisher matrix derivative exactly")
                if not self.neural_cyclic:
                    deriv_matrix = np.asarray(fisher_deriv_neural(self.alpha_lXs, self.beta_lXs, self.gamma_lXs, self.neural_weights, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                        legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                else:
                    deriv_matrix = np.asarray(fisher_deriv_neural_cyclic(self.alpha_lXs, self.beta_lXs, self.neural_weights, self.quad_weights_1d, np.asarray(self.base.beam[:,None]*self.base.beam[None,:]*self.base.inv_Cl_tot_mat,order='C'), 
                                        legs, w_mus, self.lmin, self.lmax, self.base.nthreads))
                
            else:
                raise Exception("Template %s not implemented!"%template)
                
            output[template] = np.sum(deriv_matrix), deriv_matrix
            
        self.timers['analytic_fisher'] += time.time()-t_init

        return output

    ### NUMERATOR
    @_timer_func('numerator')
    def Bl_numerator(self, data, include_linear_term=True, verb=False, input_type='map'):
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
        t_init = time.time()
        h_data_lm = np.asarray(self.applySinv(data, input_type=input_type, lmax=self.lmax)[:,self.lminfilt], order='C')
        self.timers['Sinv'] += time.time()-t_init
           
        # Compute all relevant weighted maps
        proc_maps = self._apply_all_filters(h_data_lm)
        
        # Define 3- and 1-field arrays
        b3_num = np.zeros(len(self.templates))
        if include_linear_term:
            b1_num = np.zeros(len(self.templates))
            
        if verb: print("# Assembling bispectrum numerator (3-field term)")
        for ii,t in enumerate(self.templates):
            
            if t=='fNL-loc':
                # fNL-local template
                print("Computing fNL-local template")
                
                t_init = time.time()
                b3_num[ii] = 3./5.*self.utils.fnl_sum(self.r_weights[t], proc_maps['p'], proc_maps['p'], proc_maps['q'])*self.base.A_pix
                self.timers['fNL_summation'] += time.time()-t_init

            elif t=='fNL-eq':
                # fNL-eq template
                print("Computing fNL-eq template")
                
                t_init = time.time()
                summ  = 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['r1'], proc_maps['r2'], proc_maps['p'])
                summ -= 3*self.utils.fnl_sum(self.r_weights[t], proc_maps['p'], proc_maps['p'], proc_maps['q'])
                summ -= 2*self.utils.fnl_sum(self.r_weights[t], proc_maps['r2'], proc_maps['r2'], proc_maps['r2'])
                b3_num[ii] = 3./5.*summ*self.base.A_pix
                self.timers['fNL_summation'] += time.time()-t_init

            elif t=='neural':
                # Neural-network input template
                print("Computing neural template")
                
                t_init = time.time()
                if not self.neural_cyclic:
                    b3_num[ii] = 3./5.*self.utils.neural_sum(self.r_weights[t], self.neural_weights, proc_maps['neural-alpha'], proc_maps['neural-beta'], proc_maps['neural-gamma'])*self.base.A_pix 
                else: 
                    b3_num[ii] = 3./5.*self.utils.neural_sum(self.r_weights[t], self.neural_weights, proc_maps['neural-alpha'], proc_maps['neural-beta'], proc_maps['neural-beta'])*self.base.A_pix
                self.timers['fNL_summation'] += time.time()-t_init
                
            elif t=='isw-lensing':
                # ISW-Lensing template
                print("Computing ISW-lensing template")
                
                t_init = time.time()
                b3_num[ii] = 0.5*isw_bispectrum_sum(proc_maps['u'], proc_maps['v'], proc_maps['v-isw'], self.base.nthreads)*self.base.A_pix
                self.timers['lensing_summation'] += time.time()-t_init
                
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
                for ii,t in enumerate(self.templates):
                    if t=='fNL-loc':
                        t_init = time.time()
                        
                        # Sum over permutations
                        summ  = 2.*self.utils.fnl_sum(self.r_weights[t], proc_maps['p'], this_proc_maps['p'], this_proc_maps['q'])
                        summ += self.utils.fnl_sum(self.r_weights[t], this_proc_maps['p'], this_proc_maps['p'], proc_maps['q'])
                        b1_num[ii] += -3./5.*summ*self.base.A_pix/self.N_it
                        self.timers['fNL_summation'] += time.time()-t_init
                        
                    if t=='fNL-eq':
                        t_init = time.time()
                        
                        # Sum over permutations
                        summ  = 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['r1'], this_proc_maps['r2'], this_proc_maps['p'])
                        summ += 6*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['r1'], proc_maps['r2'], this_proc_maps['p'])
                        summ += 6*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['r1'], this_proc_maps['r2'], proc_maps['p'])
                        summ -= 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['p'], this_proc_maps['p'], this_proc_maps['q'])
                        summ -= 3*self.utils.fnl_sum(self.r_weights[t], this_proc_maps['p'], this_proc_maps['p'], proc_maps['q'])
                        summ -= 6*self.utils.fnl_sum(self.r_weights[t], proc_maps['r2'], this_proc_maps['r2'], this_proc_maps['r2'])
                        b1_num[ii] += -3./5.*summ*self.base.A_pix/self.N_it
                        self.timers['fNL_summation'] += time.time()-t_init
                        
                    if t=='neural':
                        t_init = time.time()
                        
                        # Sum over permutations
                        if not self.neural_cyclic:
                            summ  = self.utils.neural_sum(self.r_weights[t], self.neural_weights, this_proc_maps['neural-alpha'], this_proc_maps['neural-beta'], proc_maps['neural-gamma'])
                            summ += self.utils.neural_sum(self.r_weights[t], self.neural_weights, this_proc_maps['neural-alpha'], this_proc_maps['neural-gamma'], proc_maps['neural-beta'])
                            summ += self.utils.neural_sum(self.r_weights[t], self.neural_weights, this_proc_maps['neural-beta'], this_proc_maps['neural-gamma'], proc_maps['neural-alpha'])
                            b1_num[ii] += -3./5.*summ*self.base.A_pix/self.N_it
                        else:
                            summ  = 2.*self.utils.neural_sum(self.r_weights[t], self.neural_weights, this_proc_maps['neural-alpha'], this_proc_maps['neural-beta'], proc_maps['neural-beta'])
                            summ += self.utils.neural_sum(self.r_weights[t], self.neural_weights, this_proc_maps['neural-beta'], this_proc_maps['neural-beta'], proc_maps['neural-alpha'])
                            b1_num[ii] += -3./5.*summ*self.base.A_pix/self.N_it
                        self.timers['fNL_summation'] += time.time()-t_init
                    
                    if t=='isw-lensing':
                        t_init = time.time()
               
                        # Sum over 3 permutations
                        summ =  isw_bispectrum_sum(proc_maps['u'], this_proc_maps['v'], this_proc_maps['v-isw'], self.base.nthreads)
                        summ += isw_bispectrum_sum(this_proc_maps['u'], proc_maps['v'], this_proc_maps['v-isw'], self.base.nthreads)
                        summ += isw_bispectrum_sum(this_proc_maps['u'], this_proc_maps['v'], proc_maps['v-isw'], self.base.nthreads)
                        b1_num[ii] += -0.5*summ*self.base.A_pix/self.N_it
                        self.timers['lensing_summation'] += time.time()-t_init
                                            
        if include_linear_term:
            b_num = b3_num+b1_num
        else:
            b_num = b3_num

        return b_num

    ### OPTIMIZATION
    @_timer_func('optimization')
    def optimize_radial_sampling_1d(self, reduce_r=1, tolerance=1e-3, N_split=None, split_index=None, initial_r_points=None, verb=False):
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
        r_raw = np.asarray(list(np.arange(1,self.r_star*0.95,50*reduce_r))+list(np.arange(self.r_star*0.95,self.r_hor*1.05,5*reduce_r))+list(np.arange(self.r_hor*1.05,self.r_hor+5000,50*reduce_r)))
        r_init = 0.5*(r_raw[1:]+r_raw[:-1])
        self.quad_weights_1d = r_init**2*np.diff(r_raw)
        r_weights = {}
        
        # Partition the radial indices if required or read in precomputed points
        if initial_r_points is not None:
            assert split_index is None, "Cannot specify both initial_r_points and index_split"
            assert len(initial_r_points)==len(np.unique(initial_r_points)), "initial_r_points cannot contain repeated points"
            inds = np.asarray([np.where(r==r_init)[0][0] for r in initial_r_points])
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
        ordered_templates = [tem for tem in self.templates if tem in self.all_templates_1d]
        
        # Create list of radial indices in the optimized representation
        inds = []
        inds_init = np.arange(self.N_r)
        
        # Compute all Fisher matrix derivatives of interest
        if verb: print("Computing all Fisher matrix derivatives")
        derivs = self._compute_fisher_derivatives(ordered_templates, verb=verb)
        
        # Save ideal Fisher matrices
        if not hasattr(self, 'ideal_fisher'):
            self.ideal_fisher = {}
        for t in ordered_templates:
            self.ideal_fisher[t] = derivs[t][0]
        
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
                
            # Set up iteration
            if score/init_score >= tolerance:
                
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
                    inv_deriv = np.linalg.inv(deriv_matrix[inds][:,inds])
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
                    
                    # Update memory when score is accepted
                    w_vals_old = w_vals
                    score_old = score
                    
                    # Compute indices for next iteration
                    next_ind = inds_init[notinds][np.argsort(np.sum(G_mat,axis=1)**2/np.diag(G_mat))[-1]]
                    inds.append(next_ind)
                    
            if len(G_mat)==0:
                raise Exception("Failed to converge after %d iterations; this indicates a bug!"%N_fish_optim)
                
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
        fish = np.zeros((len(self.templates),len(self.templates)),dtype='complex')

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
            
            # Filter maps
            if verb: print("Computing filtered maps")
            if 'q' in self.to_compute:
                if verb: print("Creating Q maps")
                Q_maps = self._filter_pair(Uinv_a_lms, 'Q')   
            if 'p' in self.to_compute:
                if verb: print("Creating P maps")
                P_maps = self._filter_pair(Uinv_a_lms, 'P')   
            if 'r1' in self.to_compute:
                if verb: print("Creating R1 maps")
                R1_maps = self._filter_pair(Uinv_a_lms, 'R1')   
            if 'r2' in self.to_compute:
                if verb: print("Creating R2 maps")
                R2_maps = self._filter_pair(Uinv_a_lms, 'R2')   
            if 'u' in self.to_compute:
                if verb: print("Creating U maps")
                U_maps = self._filter_pair(Uinv_a_lms, 'U')
            if 'v' in self.to_compute:
                if verb: print("Creating V maps")
                V_maps = self._filter_pair(Uinv_a_lms, 'V')
            if 'v-isw' in self.to_compute:
                if verb: print("Creating ISW V maps")
                V_isw_maps = self._filter_pair(Uinv_a_lms, 'V-ISW')
            if self.neural_inputs is not None:
                if verb: print("Creating neural maps")
                neural_alphas = self._filter_pair(Uinv_a_lms, 'neural-alpha')
                neural_betas = self._filter_pair(Uinv_a_lms, 'neural-beta')
                if not self.neural_cyclic:
                    neural_gammas = self._filter_pair(Uinv_a_lms, 'neural-gamma')
            
            # Define output arrays (Q11, Q22)
            Qs = np.zeros((2,len(self.templates),1+2*self.pol,np.sum(self.lfilt)),dtype=np.complex128,order='C')
            
            # Compute products (with symmetries)
            for ii,t in enumerate(self.templates):
                if t=='fNL-loc':
                    
                    if verb: print("Computing Q-derivative for fNL-loc")

                    # Iterate over both the 11 and 22 pieces
                    for index in [0,1]:
                        Qs[index,ii]  = 12./5.*self._transform_maps(self.utils.multiply(P_maps[index],Q_maps[index]),self.plXs,self.r_weights[t])
                        Qs[index,ii] += 6./5.*self._transform_maps(self.utils.multiply(P_maps[index],P_maps[index]),self.qlXs,self.r_weights[t])
                
                if t=='fNL-eq':
                    
                    if verb: print("Computing Q-derivative for fNL-eq")

                    # Iterate over both the 11 and 22 pieces
                    for index in [0,1]:
                        Qs[index,ii]  = 36./5.*self._transform_maps(self.utils.multiply(R1_maps[index],R2_maps[index]),self.plXs,self.r_weights[t])
                        Qs[index,ii] += 36./5.*self._transform_maps(self.utils.multiply(P_maps[index],R2_maps[index]),self.r1lXs,self.r_weights[t])
                        Qs[index,ii] += 36./5.*self._transform_maps(self.utils.multiply(P_maps[index],R1_maps[index]),self.r2lXs,self.r_weights[t])
                        Qs[index,ii] -= 36./5.*self._transform_maps(self.utils.multiply(P_maps[index],Q_maps[index]),self.plXs,self.r_weights[t])
                        Qs[index,ii] -= 18./5.*self._transform_maps(self.utils.multiply(P_maps[index],P_maps[index]),self.qlXs,self.r_weights[t])
                        Qs[index,ii] -= 36./5.*self._transform_maps(self.utils.multiply(R2_maps[index],R2_maps[index]),self.r2lXs,self.r_weights[t])
                        
                if t=='neural':
                    
                    if verb: print("Computing Q-derivative for neural fNL")

                    # Iterate over both the 11 and 22 pieces
                    for index in [0,1]:
                        for iterm in range(self.neural_terms):
                            if not self.neural_cyclic:
                                Qs[index,ii] += 6./5.*self.neural_weights[iterm]*self._transform_maps(self.utils.multiply(neural_alphas[iterm][index],neural_betas[iterm][index]),self.gamma_lXs[iterm],self.r_weights[t])
                                Qs[index,ii] += 6./5.*self.neural_weights[iterm]*self._transform_maps(self.utils.multiply(neural_betas[iterm][index],neural_gammas[iterm][index]),self.alpha_lXs[iterm],self.r_weights[t])
                                Qs[index,ii] += 6./5.*self.neural_weights[iterm]*self._transform_maps(self.utils.multiply(neural_gammas[iterm][index],neural_alphas[iterm][index]),self.beta_lXs[iterm],self.r_weights[t])
                            else:
                                Qs[index,ii] += 12./5.*self.neural_weights[iterm]*self._transform_maps(self.utils.multiply(neural_alphas[iterm][index],neural_betas[iterm][index]),self.beta_lXs[iterm],self.r_weights[t])
                                Qs[index,ii] += 6./5.*self.neural_weights[iterm]*self._transform_maps(self.utils.multiply(neural_betas[iterm][index],neural_betas[iterm][index]),self.alpha_lXs[iterm],self.r_weights[t])
                
                if t=='isw-lensing':
                    if verb: print("Computing Q-derivative for isw-lensing")
                    
                    # Iterate over both the 11 and 22 pieces
                    for index in [0,1]:
                        
                        ## First term
                        # X = T
                        input_map = 2.*np.real(V_maps[index][0]*V_isw_maps[index].conjugate())
                        Qs[index,ii,0] = self.base.to_lm(input_map[None],lmax=self.lmax)[0,self.lminfilt]
                        if self.pol:
                            # X = E, B
                            input_map = V_maps[index][1]*V_isw_maps[index].conjugate() - V_maps[index][2]*V_isw_maps[index]
                            output_lm = 0.5*self.base.to_lm_spin(input_map[None],input_map[None].conjugate(),spin=2,lmax=self.lmax)[:,self.lminfilt]
                            Qs[index,ii,1] = output_lm[0] + output_lm[1]
                            Qs[index,ii,2] = -1.0j*(output_lm[0] - output_lm[1])
                                
                        ## Second term
                        # Y = T
                        pref = np.sqrt(self.ls*(self.ls+1.))
                        input_map = U_maps[index][0]*V_isw_maps[index] 
                        out_lm = self.base.to_lm_spin(input_map.conjugate(), input_map, spin=1, lmax=self.lmax)[:,self.lminfilt].copy()
                        Qs[index,ii,0] += -self.C_lens_weight['TT'][self.ls]*pref*(out_lm[0]-out_lm[1])
                        if self.pol:
                            Qs[index,ii,1] += -self.C_lens_weight['TE'][self.ls]*pref*(out_lm[0]-out_lm[1])

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
                            Qs[index,ii,0] += 0.5*self.C_lens_weight['TE'][self.ls]*diff_lm
                            Qs[index,ii,1] += 0.5*self.C_lens_weight['EE'][self.ls]*diff_lm
                            Qs[index,ii,2] += -1.0j*0.5*self.C_lens_weight['BB'][self.ls]*sum_lm
                            del out_lm1, out_lm3, diff_lm, sum_lm, prefP, prefM
                        
                        ## Third term
                        input_map = U_maps[index][0]*V_maps[index][0].conjugate()
                        if not self.pol:
                            input_map = U_maps[index][0]*V_maps[index][0].conjugate()
                            Qs[index,ii,0] += -self.C_Tphi[self.ls]*np.sqrt(self.ls*(self.ls+1.))*np.sum(np.array([1,-1])[:,None]*self.base.to_lm_spin(input_map, input_map.conjugate(), spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                        else: 
                            input_map = 2.*U_maps[index][0]*V_maps[index][0].conjugate()
                            input_map += (U_maps[index][1] + 1.0j*U_maps[index][2])*V_maps[index][1].conjugate() + (-U_maps[index][1].conjugate() + 1.0j*U_maps[index][2].conjugate())*V_maps[index][2]
                            out_lm = np.sqrt(self.ls*(self.ls+1.))*np.sum(np.array([1,-1])[:,None]*self.base.to_lm_spin(input_map, input_map.conjugate(), spin=1,lmax=self.lmax)[:,self.lminfilt],axis=0)
                            Qs[index,ii,0] += -0.5*self.C_Tphi[self.ls]*out_lm
                            Qs[index,ii,1] += -0.5*self.C_Ephi[self.ls]*out_lm
                            del out_lm
                    
            if weighting=='Ainv' and verb: print("Applying S^-1 weighting to output")
            for qindex in range(2):
                self._weight_Q_maps(Qs[qindex], weighting)
            return Qs.reshape(2,len(self.templates),-1)

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
        fish = np.zeros((len(self.templates),len(self.templates)))
        
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