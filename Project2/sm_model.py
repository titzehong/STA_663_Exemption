from socket import AF_X25
import pandas as pd 
import numpy as np 
import pymc3 as pm 
import theano
import theano.tensor as tt
import arviz as az

# Matplotlib hack
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os 
import pickle 

from sm_data import generate_prediction_data, generate_NPI_prediction_data
from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
from sm_utils import *
from sm_utils import _get_convolution_ready_gt,\
     one_step_sus_adjustment_local, one_step_sus_adjustment_dorm,\
          dorm_calc_infected, community_calc_infected, merge_frames,\
              create_mixture_si, _get_convolution_ready_gt_multi_variant, \
                  _get_convolution_ready_gt_mixture_variants

QUARANTINE_DELAY_FP = 'data/onset-delay/quarantined_onset_breakdown.pkl'
UNLIKED_DELAY_FP = 'data/onset-delay/unlinked_onset_breakdown.pkl'

class SemiMechanisticModels:

    def __init__ (self):

        self.model = None
        self._trace = None
        self.ppc = None

        self.local_cases = None
        self.dorm_cases = None
        self.total_cases = None
        self.imported_cases = None
        self.NPIS_array = None



    def build_model(self, local_cases,
                          dorm_cases,
                          imported_cases,
                          NPIS_array,
                          len_observed,
                          total_cases=None,
                          separate_dorms=True,
                          separate_quarantine=False,
                          test_counts=None,
                          quarantine_counts=None,
                          prediction_t=0,
                          likelihood_fun='PO',
                          coef_prior='beta', 
                          vax_adjustment=False,
                          effective_vax_rate=None,
                          community_pop=None,
                          dorm_pop=None,
                          a_rate_regime=None,
                          a_rate_regime2=None,
                          use_latent=False,
                          variant_pct_series=None,
                          variant_si_dists=None):
        
        self.local_cases = local_cases
        self.dorm_cases = dorm_cases
        self.imported_cases = imported_cases
        self.total_cases = total_cases
        self.separate_dorms = separate_dorms
        self.separate_quarantine = separate_quarantine
        self.NPIS_array = NPIS_array
        self.len_observed = len_observed
        self.test_counts = test_counts
        self.quarantine_counts = quarantine_counts 
        self.coef_prior = coef_prior
        self.effective_vax_rate = effective_vax_rate
        self.community_pop = community_pop
        self.dorm_pop = dorm_pop
        self.vax_adjustment = vax_adjustment
        self.likelihood_fun = likelihood_fun
        self.a_rate = a_rate_regime
        self.a_rate2 = a_rate_regime2
        self.use_latent = use_latent

        self.variant_pct_series = variant_pct_series
        self.variant_si_dists = variant_si_dists

        # Get prediction steps -> bad pls fix
        self.prediction_t = prediction_t
        self.end_date = len_observed - self.prediction_t

        num_NPIS = NPIS_array.shape[1]
        
        # Prepare p delay and conv
        
        if variant_pct_series is not None and variant_pct_series is not None:
            
            print("Using Mixture Variant")

            # Get si matrix
            len_gt = len(variant_si_dists[0])
            si_matrix = create_mixture_si_matrix(variant_pct_series, len_observed, variant_si_dists)

            convolution_ready_gt = _get_convolution_ready_gt_mixture_variants(len_observed,
                                                                            len_gt,
                                                                            si_matrix)
        
        else:
            convolution_ready_gt = _get_convolution_ready_gt(len_observed)

        try:
            p_delay = get_p_delay()
        except:
            #get p_delay from latestdata.csv
            delay = pd.read_csv("data/p_delay.csv")
            p_delay = np.array(delay)
            p_delay[0:6] = [0.0001]

        if separate_dorms and not separate_quarantine:
            raise ValueError('no')
            p_delay_dorm = np.array([0.0001, 0.0001, 0.0001, 0.1, 0.2, 0.4, 0.2, 0.1])
            p_delay_local = p_delay

            with pm.Model() as self.model:
                
                # R_t Foreign
                log_r_t_foreign = pm.GaussianRandomWalk(
                    'log_r_t_foreign',
                    #sigma=0.035,
                    sigma=0.02,
                    shape=len_observed)
                
                # R_t Local
                log_r_t_local = pm.GaussianRandomWalk(
                    'log_r_t_local',
                    #sigma=0.035,
                    sigmma=0.02,
                    shape=len_observed)
                
                if num_NPIS>1:  # Means not using the index thingy
                    # NPI Covariates
                    beta_list = []
                    for i in range(num_NPIS):
                        beta_list.append(pm.Normal(f"b_{i}", 0, sigma=0.03))

                    ### if using hierarchical 
                    #beta_prior_mean = pm.Normal("b_hyper_mean", 0, sigma=0.1)
                    #beta_prior_var = pm.Normal("b_hyper_var", 0.03, sigma=0.1)
                    #for i in range(num_NPIS):
                    #    beta_list.append(pm.Normal(f"b_{i}", beta_prior_mean, sigma=beta_prior_var))

                    betas = pm.math.stack(beta_list, 0)
                    #beta_intercept = pm.Normal('b_inter', 0, sigma=0.1)
                    rt_covariates = pm.math.dot(np.array(NPIS_array), betas)
                    
                else: # Using index
                    beta = pm.Normal(f"b_index", 0, sigma=0.03)
                    rt_covariates = beta*np.array(NPIS_array.reshape(-1))
                
                # Form r_t as GRW + covariates
                rt_covariates = pm.Deterministic('rt covariate ts', rt_covariates)
                r_t_foreign = pm.Deterministic('r_t_foreign', pm.math.exp(log_r_t_foreign + rt_covariates))
                r_t_local = pm.Deterministic('r_t_local', pm.math.exp(log_r_t_local + rt_covariates))
                    
                # Imported cases leak percent
                """
                log_eps_t = pm.GaussianRandomWalk(
                    'log_eps_t',
                    sigma=0.035,
                    shape=len_observed)
                """
                #eps_t = pm.Deterministic('eps_t', pm.math.exp(log_eps_t))
                eps_t = pm.Beta('eps_t', alpha=1, beta=1000)
                
                
                # Infection Latent Dormitory
                seed_dorm = pm.Exponential('Seed_dorm', 150)  # Scale of infection will be small (Dorm cases started at 0)
                y0_dorm = tt.zeros(len_observed)
                y0_dorm = tt.set_subtensor(y0_dorm[0], seed_dorm)
                # Apply recursively to populate tensor
                outputs_dorm, _ = theano.scan(
                        fn=lambda t, gt , y, r_t_foreign: tt.set_subtensor(y[t], tt.sum(r_t_foreign*y*gt)),
                        sequences=[tt.arange(1, len_observed), convolution_ready_gt],
                        outputs_info=y0_dorm,
                        non_sequences=r_t_foreign,
                        n_steps=len_observed-1,
                )
                infections_dorm = pm.Deterministic('infections_dorm', outputs_dorm[-1])
                
                # Infection Latent Local only
                seed = pm.Exponential('Seed_local', 1)  # Scale of infection will be small (Local cases were quite high already)
                y0_local = tt.zeros(len_observed)
                y0_local = tt.set_subtensor(y0_local[0], seed)
                # Apply recursively to populate tensor
                outputs_local, _ = theano.scan(
                        fn=lambda t, gt , i_cases, y, r_t_local: tt.set_subtensor(y[t], tt.sum(r_t_local*y*gt) + eps_t * i_cases),
                        sequences=[tt.arange(1, len_observed), convolution_ready_gt, theano.shared(imported_cases)],
                        outputs_info=y0_local,
                        non_sequences=r_t_local,
                        n_steps=len_observed-1,
                )
                
                infections_local = pm.Deterministic('infections_local', outputs_local[-1])
                
                
                
                # Onset - Delay Dist - Local
                t_p_delay_local = pm.Data("p_delay_local", p_delay_local)
                
                # Onset - Delay Dist - Dorm
                t_p_delay_dorm = pm.Data("p_delay_dorm", pd.Series(p_delay_dorm))
                
                
                # Test adjusted positive - Local
                test_adjusted_positive_local = pm.Deterministic(
                    "test adjusted positive local",
                    conv(infections_local, t_p_delay_local, len(p_delay), len_observed)
                )
                test_adjusted_positive_jittered_local = pm.Deterministic('test_adjusted_positive_jit_local',
                                                            test_adjusted_positive_local + 0)
                
                # Test adjusted Positive - Dorm
                test_adjusted_positive_dorm = pm.Deterministic(
                    "test adjusted positive dorm",
                    conv(infections_dorm, t_p_delay_dorm, len(pd.Series(p_delay_dorm)), len_observed)
                )   

                test_adjusted_positive_jittered_dorm = pm.Deterministic('test_adjusted_positive_jit_dorm',
                                                            test_adjusted_positive_dorm + 0)
                
                # Likelihood
                if likelihood_fun == 'ZINB':
                    pm.ZeroInflatedNegativeBinomial('Obs', 
                            mu=test_adjusted_positive_jittered, 
                            alpha=pm.Gamma("alpha", mu=1, sigma=0.5),
                            psi = pm.Beta('psi', 2,2),                    
                            observed=local_cases)    
                    
                elif likelihood_fun == 'NB':
                    pm.NegativeBinomial('Obs', 
                            mu=test_adjusted_positive_jittered, 
                            alpha=pm.Gamma("alpha", mu=1, sigma=0.5),
                            observed=local_cases)
                
                elif likelihood_fun == 'PO':
                    print('hi')
                    pm.Poisson('Obs_local', 
                            mu=test_adjusted_positive_jittered_local,
                            observed=local_cases)
                    
                    pm.Poisson('Obs_dorm', 
                            mu=test_adjusted_positive_jittered_dorm,
                            observed=dorm_cases)
            return self.model

        elif separate_quarantine and separate_dorms:
            
            print("MODELING COMMUNITY AND DORM CASES")

            if (test_counts is None) or (quarantine_counts is None):
                raise ValueError('test_counts and quarantine_counts must be input')

            if (len(self.local_cases) != 2) and (not isinstance(local_cases, list)):
                raise ValueError('local_cases should be an list with two elements, Un-quarantined cases series as the first and quarantined cases series as the second')

            # Prepare observed data
            model_input_local_unquarantined = local_cases[0]
            model_input_local_quarantined = local_cases[1]

            """
            p_delay_dorm = np.array([0.0001, 0.0001, 0.0001, 0.1, 0.2, 0.4, 0.2, 0.1])
            p_delay_q = np.array([0.0001, 0.0001, 0.0001, 0.1, 0.2, 0.4, 0.2, 0.1])
            p_delay_local = p_delay
            """

            # Dorm and quanratined
            p_delay_quarantined = get_custom_p_delay(QUARANTINE_DELAY_FP, incubation_days=4)
            p_delay_dorm = p_delay_quarantined
            p_delay_q = p_delay_quarantined

            # Un-Quarantined
            p_delay_local = get_custom_p_delay(UNLIKED_DELAY_FP, incubation_days=4)

            print("Dorm Pop: ",dorm_pop)
            print("community pop: ",community_pop)

            with pm.Model() as self.model:
                
                # R_t Foreign
                log_r_t_foreign = pm.GaussianRandomWalk(
                    'log_r_t_foreign',
                    sigma=0.02,
                    shape=len_observed)
                
                # R_t Local
                log_r_t_local = pm.GaussianRandomWalk(
                    'log_r_t_local',
                    #mu=0.0006,
                    sigma=0.02,
                    shape=len_observed)
                    
                print("Adjusted priors")

                if num_NPIS>1:  # Means not using the index thingy
                    # NPI Covariates
                    beta_list = []

                    if (self.coef_prior == 'gaussian') or (self.use_latent): 
                        print("Gaussian Covariate Prior")
                        for i in range(num_NPIS):
                            beta_list.append(pm.Normal(f"b_{i}", 0, sigma=0.05))
                        betas = pm.math.stack(beta_list, 0)
                        #beta_intercept = pm.Normal('b_inter', 0, sigma=0.1)
                    
                    elif self.coef_prior == 'beta':
                        print("Beta Covariate Prior")
                        print("Number NPIS Being Formed: ", num_NPIS)
                        for i in range(num_NPIS):
                            beta_list.append(pm.Beta(f"b_{i}", mu=0.1, sigma=0.03))   

                        betas = pm.math.stack(beta_list, 0) * -1
                    
                    rt_covariates = pm.Deterministic('rt_covariate_series', pm.math.dot(np.array(NPIS_array), betas))

                    
                else: # Using index
                    beta = pm.Normal(f"b_index", 0, sigma=0.03)
                    rt_covariates = beta*np.array(NPIS_array.reshape(-1))
                

                # Form r_t as GRW + covariates
                r_t_foreign = pm.Deterministic('r_t_foreign', pm.math.exp(log_r_t_foreign + rt_covariates))
                r_t_local = pm.Deterministic('r_t_local', pm.math.exp(0.2 + log_r_t_local + rt_covariates))
                
                # Vax Rate
                effective_vax_rate = pm.Data('effective vaccination rates', effective_vax_rate)

                # beta_coef sus dorm and sus community
                sus_pop_comm_beta = pm.Beta("b_comm_susc", mu=0.6, sigma=0.1)
                sus_pop_dorm_beta = pm.Beta("b_dorm_susc", mu=0.6, sigma=0.1)
                
                
                # Imported leakage
                eps_t = pm.Beta('eps_t', alpha=1, beta=10)

                ###################################
                #### Dorm Infection Generation ####
                ###################################
                seed_dorm = pm.Exponential('Seed_dorm', 150)  # Scale of infection will be small (Dorm cases started at 0)
                y0_dorm = tt.zeros(len_observed)
                y0_dorm = tt.set_subtensor(y0_dorm[0], seed_dorm)
                
                sus_0_dorm = tt.zeros(len_observed)
                sus_0_dorm = tt.set_subtensor(sus_0_dorm[0], dorm_pop)
                
                # Apply recursively to populate tensor
                outputs_dorm, _ = theano.scan(
                        fn=dorm_calc_infected,
                        sequences=[tt.arange(1, len_observed),
                                convolution_ready_gt, effective_vax_rate],
                        outputs_info=[y0_dorm, sus_0_dorm],
                        non_sequences=[r_t_foreign, sus_pop_dorm_beta, dorm_pop],
                        n_steps=len_observed-1,
                )
                
                # Retrieve the dorm infections and dorm susceptibles
                infections_dorm = pm.Deterministic('infections_dorm', outputs_dorm[0][-1])
                susceptible_dorm = pm.Deterministic('susceptible_dorm', outputs_dorm[1][-1])
                
                ################################
                #### Quanratined Parameters ####
                ################################            
                # Qurantine randomwalk
                log_lambda_t = pm.GaussianRandomWalk(
                    'log_lambda_t',
                    sigma=0.035,
                    shape=len_observed)

                # Beta for no. in quarantine covariate
                beta_quarantine = pm.Normal("beta_quarantine", 0, sigma=0.03)
                quarantine_covariates = beta_quarantine * quarantine_counts

                lambda_t = pm.Deterministic('Q Rate', pm.math.sigmoid(log_lambda_t + quarantine_covariates))

                ##################################################
                #### Community Q and UnQ Infection Generation ####
                ##################################################
                seed = pm.Exponential('Seed_local', 1)  # Scale of infection will be small (Local cases were quite high already)
                y0_local = tt.zeros(len_observed)
                y0_local = tt.set_subtensor(y0_local[0], seed)
                
                sus_0_comm = tt.zeros(len_observed)
                sus_0_comm = tt.set_subtensor(sus_0_comm[0], community_pop)
                
                # Apply recursively to populate tensor
                outputs_local, _ = theano.scan(community_calc_infected,
                        sequences=[tt.arange(1, len_observed), convolution_ready_gt,
                                theano.shared(imported_cases), lambda_t, effective_vax_rate],
                        outputs_info=[y0_local,sus_0_comm],
                        non_sequences=[r_t_local,sus_pop_comm_beta, community_pop, eps_t],
                        n_steps=len_observed-1,
                )


                infections_local = pm.Deterministic('infections_local_uncontained', outputs_local[0][-1])
                susceptible_community = pm.Deterministic('susceptible_community', outputs_local[1][-1])
                
                q_lambda_t = pm.Deterministic('infections_rate', (1-lambda_t)/lambda_t)

                infections_local_contained = pm.Deterministic('infections_local_contained', q_lambda_t*infections_local)
                

                ##############################################
                #### Calculate Implied Re(t) for analysis ####
                ##############################################
                # Calc implied R(t) for dorm and community
                exp_sus_rate_dorm = tt.exp(sus_pop_dorm_beta*susceptible_dorm/dorm_pop)
                exp_sus_rate_comm = tt.exp(sus_pop_comm_beta*susceptible_community/community_pop)
                
                # Calculate the implied number of infected
                implied_rt_inf_t_dorm = pm.Deterministic('R_t_dorm_implied', exp_sus_rate_dorm*r_t_foreign)
                implied_rt_inf_t_comm = pm.Deterministic('R_t_comm_implied' ,exp_sus_rate_comm*r_t_local)

                if False:
                    
                    print("Epidemia Vax Adjustment")
                    #########################################################
                    #### Adjustment for Reduction Susceptibles Community ####
                    #########################################################

                    effective_vax_rate = pm.Data('effective vaccination rates', effective_vax_rate)
                    
                    
                    adjusted_infections_summary,_ = theano.scan(
                            fn=one_step_sus_adjustment_local,
                        sequences=[infections_local_contained, infections_local, effective_vax_rate],
                            outputs_info=[community_pop,
                                        infections_local_contained[0],
                                        infections_local[0],
                                        ],
                            non_sequences=community_pop,
                            n_steps=len_observed,
                    )
                    
                    infections_local_contained_adjusted = adjusted_infections_summary[1]
                    infections_local_adjusted = adjusted_infections_summary[2]
                    susceptible_community = adjusted_infections_summary[0]
                    
                    
                    susceptible_community = pm.Deterministic('susceptible_community', susceptible_community)
                    infections_local_contained_adjusted = pm.Deterministic('infections_local_contained_adjusted',
                                                                        infections_local_contained_adjusted)
                    infections_local_adjusted = pm.Deterministic('infections_local_adjusted',
                                                                infections_local_adjusted)
                    
                    
                    #############################################
                    #### Adjustment for Reduction Dormitory  ####
                    #############################################
                    
                    
                    adjusted_infections_dorm_summary,_ = theano.scan(
                            fn=one_step_sus_adjustment_dorm,
                        sequences=[infections_dorm, effective_vax_rate],
                            outputs_info=[dorm_pop,
                                        infections_dorm[0]
                                        ],
                            non_sequences=dorm_pop,
                            n_steps=len_observed,
                    )
                    
                    infections_dorm_adjusted = adjusted_infections_dorm_summary[1]
                    susceptible_dorm = adjusted_infections_dorm_summary[0]
                    
                    
                    susceptible_dorm = pm.Deterministic('susceptible_dorm', susceptible_dorm)

                    #sus_dorm_print = tt.printing.Print("Sus Dorm")(susceptible_dorm[-1])

                    infections_dorm_adjusted = pm.Deterministic('infections_dorm_adjusted',
                                                                infections_dorm_adjusted)
                
                else:
                    print("NO Vax Adjustment")
                    infections_local_adjusted = pm.Deterministic('infections_local_adjusted', infections_local)
                    infections_local_contained_adjusted = pm.Deterministic('infections_local_contained_adjusted',
                                                                        infections_local_contained)
                    infections_dorm_adjusted = pm.Deterministic('infections_dorm_adjusted', infections_dorm)

                #############################
                #### Ascertainment Rates ####
                #############################
                                
                #### alpha Parameter - determines test rates from 3 sources ####
                # Dorm
                log_alpha_dorm_t = pm.GaussianRandomWalk(
                    'log_alpha_dorm_t',
                     # sigmoid(2) approx 0.88 corresponding to high prior belief that the people will be tested
                    sigma=0.0055,
                    shape=len_observed)
                
                # Local - UnQ
                log_alpha_unQ_t = pm.GaussianRandomWalk(
                    'log_alpha_unQ_t',
                    sigma=0.0055,
                    shape=len_observed)
                
                # Local - Q
                log_alpha_Q_t = pm.GaussianRandomWalk(
                    'log_alpha_Q_t',
                      # sigmoid(2) approx 0.88 corresponding to high prior belief that the people will be tested
                    sigma=0.0055,
                    shape=len_observed)
                

                # Hierarchical usage of test_counts
                beta_test_hier_mean = pm.Normal("beta_quarantine_hier_mean", 0, sigma=0.03)
                
                # prior for dorm, local unQ, Q
                # Fixed prior sd of 0.03
                test_dorm_tilde = pm.Normal("test_dorm_tilde", mu=0, sigma=1)
                beta_test_dorm = pm.Deterministic("beta_test_dorm", beta_test_hier_mean + 0.03 * test_dorm_tilde)

                test_unQ_tilde = pm.Normal("test_unQ_tilde", mu=0, sigma=1)
                beta_test_unQ = pm.Deterministic("beta_test_unQ", beta_test_hier_mean + 0.03 * test_unQ_tilde)

                test_Q_tilde = pm.Normal("test_Q_tilde", mu=0, sigma=1)
                beta_test_Q = pm.Deterministic("beta_test_Q", beta_test_hier_mean + 0.03 * test_Q_tilde)

                # Ascertainment Regime
                beta_ascertainment_regime_Q = pm.HalfNormal("ascertainment regime beta Q", sigma=5)  # For 2021 when stopped recording quarantines
                beta_ascertainment_regime_unQ = pm.HalfNormal("ascertainment regime beta unQ", sigma=1.5) # For 2021 when stopped recording quarantines
                beta_ascertainment_regime_unQ_2 = pm.HalfNormal("ascertainment regime beta unQ 2", sigma=0.5) # For 2022 when started counting more tests

                # Get alphas
                alpha_dorm_t = pm.Deterministic('alpha_dorm_t', pm.math.sigmoid(6.5 + log_alpha_dorm_t))
                
                alpha_unQ_t = pm.Deterministic('alpha_local_unQ_t', pm.math.sigmoid(1.5 + log_alpha_unQ_t +\
                                                                                    beta_test_unQ*test_counts -\
                                                                                    beta_ascertainment_regime_unQ*self.a_rate + \
                                                                                    beta_ascertainment_regime_unQ_2*self.a_rate2))
                
                #alpha_Q_t = pm.Deterministic('alpha_local_Q_t', pm.math.sigmoid(1.5 + log_alpha_Q_t + beta_test_Q*self.test_counts))
                alpha_Q_t = pm.Deterministic('alpha_local_Q_t', pm.math.sigmoid(6.5 + log_alpha_Q_t - beta_ascertainment_regime_Q*self.a_rate))
                
                """
                # Temp for debugging
                alpha_dorm_t=1
                alpha_unQ_t=1
                alpha_Q_t=1
                """

                # Onset - Delay Dist - Local
                t_p_delay_local = pm.Data("p_delay_local", p_delay_local)
                
                # Onset - Delay Dist - Qurantine
                t_p_delay_local_q = pm.Data('p_delay_local_qurantine', p_delay_q)
                
                # Onset - Delay Dist - Dorm
                t_p_delay_dorm = pm.Data("p_delay_dorm", pd.Series(p_delay_dorm))

                
                
                # Test adjusted positive - Local
                test_adjusted_positive_local = pm.Deterministic(
                    "test adjusted positive local",
                    conv(infections_local_adjusted, t_p_delay_local, len(p_delay_local), len_observed)
                )
                test_adjusted_positive_jittered_local = pm.Deterministic('test_adjusted_positive_jit_local',
                                                            alpha_unQ_t*test_adjusted_positive_local + 0.01)
                
                
                # Test adjusted positive - Local Quarantined
                test_adjusted_positive_local_qurantined = pm.Deterministic(
                    "test adjusted positive local Qurantined",
                    conv(infections_local_contained_adjusted, t_p_delay_local_q, len(p_delay_q), len_observed)
                )
                test_adjusted_positive_jittered_local_qurantined = pm.Deterministic('test_adjusted_positive_jit_local_qurantined',
                                                            alpha_Q_t*test_adjusted_positive_local_qurantined + 0.01)
                
                # Test adjusted Positive - Dorm
                test_adjusted_positive_dorm = pm.Deterministic(
                    "test adjusted positive dorm",
                    conv(infections_dorm_adjusted, t_p_delay_dorm, len(pd.Series(p_delay_dorm)), len_observed)
                )   

                test_adjusted_positive_jittered_dorm = pm.Deterministic('test_adjusted_positive_jit_dorm',
                                                            alpha_dorm_t*test_adjusted_positive_dorm + 0.01)
                print('chaanged')
                # Likelihood
                    
                if likelihood_fun == 'NB':

                    print("NEGATATIVE BINOMIAL LL")
                    
                    # Dispersion param
                    dispersion_phi_local = pm.HalfNormal('LL_Dispersion_local', sigma=5)
                    #dispersion_phi_q = pm.HalfNormal('LL_Dispersion_Q', sigma=5)
                    dispersion_phi_dorm = pm.HalfNormal('LL_Dispersion_dorm', sigma=3)
                    
                    alpha_nb_unQ = test_adjusted_positive_jittered_local / dispersion_phi_local
                    alpha_nb_Q = test_adjusted_positive_jittered_local_qurantined / dispersion_phi_local
                    alpha_nb_dorm = test_adjusted_positive_jittered_dorm / dispersion_phi_dorm
                    
                    
                    mask_val_unq = 1  # default 1
                    mask_val_q = 1
                    mask_val_dorm = 1

                    mask = model_input_local_unquarantined.copy()
                    mask = (~np.isnan(mask)).astype('float')

                    model_input_local_unquarantined_mask = model_input_local_unquarantined.copy()
                    model_input_local_unquarantined_mask[np.isnan(model_input_local_unquarantined_mask)] = mask_val_unq

                    model_input_local_quarantined_mask = model_input_local_quarantined.copy()
                    model_input_local_quarantined_mask[np.isnan(model_input_local_quarantined_mask)] = mask_val_q

                    dorm_cases_mask = dorm_cases.copy()
                    dorm_cases_mask[np.isnan(dorm_cases)] = mask_val_dorm

                    local_unq_likelihood = pm.Potential('Obs_local_LL',pm.NegativeBinomial.dist( 
                                    mu=test_adjusted_positive_jittered_local,
                                    alpha=alpha_nb_unQ)\
                                    .logp(model_input_local_unquarantined_mask.values)*mask)

                    local_q_likelihood = pm.Potential('Obs_local_q_LL',pm.NegativeBinomial.dist( 
                                                        mu=test_adjusted_positive_jittered_local_qurantined,
                                                        alpha=alpha_nb_Q)\
                                                    .logp(model_input_local_quarantined_mask.values)*mask)

                    local_q_likelihood = pm.Potential('Obs_dorm_LL',pm.NegativeBinomial.dist( 
                                                        mu=test_adjusted_positive_jittered_dorm,
                                                        alpha=alpha_nb_dorm)\
                                                    .logp(dorm_cases_mask.values)*mask)
                
                elif likelihood_fun == 'PO':
                    
                    # Suspect numerical errors here are occuring for logp function
                    mask_val_unq = 1  # default 1
                    mask_val_q = 1
                    mask_val_dorm = 1

                    mask = model_input_local_unquarantined.copy()
                    mask = (~np.isnan(mask)).astype('float')

                    model_input_local_unquarantined_mask = model_input_local_unquarantined.copy()
                    model_input_local_unquarantined_mask[np.isnan(model_input_local_unquarantined_mask)] = mask_val_unq

                    model_input_local_quarantined_mask = model_input_local_quarantined.copy()
                    model_input_local_quarantined_mask[np.isnan(model_input_local_quarantined_mask)] = mask_val_q

                    dorm_cases_mask = dorm_cases.copy()
                    dorm_cases_mask[np.isnan(dorm_cases)] = mask_val_dorm

                    local_unq_likelihood = pm.Potential('Obs_local_LL',pm.Poisson.dist( 
                                    mu=test_adjusted_positive_jittered_local)\
                                    .logp(model_input_local_unquarantined_mask.values)*mask)
        
                    local_q_likelihood = pm.Potential('Obs_local_q_LL',pm.Poisson.dist( 
                                                        mu=test_adjusted_positive_jittered_local_qurantined)\
                                                    .logp(model_input_local_quarantined_mask.values)*mask)
                    
                    local_q_likelihood = pm.Potential('Obs_dorm_LL',pm.Poisson.dist( 
                                                        mu=test_adjusted_positive_jittered_dorm)\
                                                    .logp(dorm_cases_mask.values)*mask)



            return self.model


        elif separate_quarantine and (not separate_dorms):
            
            print("MODELLING COMMUNITY CASES ONLY")

            if (test_counts is None) or (quarantine_counts is None):
                raise ValueError('test_counts and quarantine_counts must be input')

            if (len(self.local_cases) != 2) and (not isinstance(local_cases, list)):
                raise ValueError('local_cases should be an list with two elements, Un-quarantined cases series as the first and quarantined cases series as the second')

            # Prepare observed data
            model_input_local_unquarantined = local_cases[0]
            model_input_local_quarantined = local_cases[1]

            """
            p_delay_dorm = np.array([0.0001, 0.0001, 0.0001, 0.1, 0.2, 0.4, 0.2, 0.1])
            p_delay_q = np.array([0.0001, 0.0001, 0.0001, 0.1, 0.2, 0.4, 0.2, 0.1])
            p_delay_local = p_delay
            """

            # Dorm and quanratined
            p_delay_quarantined = get_custom_p_delay(QUARANTINE_DELAY_FP, incubation_days=4)
            p_delay_q = p_delay_quarantined

            # Un-Quarantined
            p_delay_local = get_custom_p_delay(UNLIKED_DELAY_FP, incubation_days=4)

            print("community pop: ",community_pop)

            with pm.Model() as self.model:
                
                
                # R_t Local
                log_r_t_local = pm.GaussianRandomWalk(
                    'log_r_t_local',
                    #mu=0.0006,
                    sigma=0.02,
                    shape=len_observed)
                    
                print("Adjusted priors")

                
                if num_NPIS>1:  # Means not using the index thingy
                    # NPI Covariates
                    beta_list = []

                    if (self.coef_prior == 'gaussian') or (self.use_latent): 
                        print("Gaussian Covariate Prior")
                        for i in range(num_NPIS):
                            beta_list.append(pm.Normal(f"b_{i}", 0, sigma=0.05))
                        betas = pm.math.stack(beta_list, 0)
                        #beta_intercept = pm.Normal('b_inter', 0, sigma=0.1)
                    
                    elif self.coef_prior == 'beta':
                        print("Beta Covariate Prior")
                        print("Number NPIS Being Formed: ", num_NPIS)
                        for i in range(num_NPIS):
                            beta_list.append(pm.Beta(f"b_{i}", mu=0.1, sigma=0.03))   

                        betas = pm.math.stack(beta_list, 0) * -1
                    
                    rt_covariates = pm.Deterministic('rt_covariate_series', pm.math.dot(np.array(NPIS_array), betas))

                else: # Using index
                    beta = pm.Normal(f"b_index", 0, sigma=0.03)
                    rt_covariates = beta*np.array(NPIS_array.reshape(-1))
                
                

                rt_covariates = 0  # Take out NPIs to see if it works

                # Form r_t as GRW + covariates
                r_t_local = pm.Deterministic('r_t_local', pm.math.exp(0.2 + log_r_t_local + rt_covariates))
                
                # Vax Rate
                effective_vax_rate = pm.Data('effective vaccination rates', effective_vax_rate)

                # beta_coef sus dorm and sus community
                sus_pop_comm_beta = pm.Beta("b_comm_susc", mu=0.6, sigma=0.1)
                
                
                # Imported leakage
                eps_t = pm.Beta('eps_t', alpha=1, beta=10)

                
                ################################
                #### Quanratined Parameters ####
                ################################            
                # Qurantine randomwalk
                log_lambda_t = pm.GaussianRandomWalk(
                    'log_lambda_t',
                    sigma=0.035,
                    shape=len_observed)

                # Beta for no. in quarantine covariate
                beta_quarantine = pm.Normal("beta_quarantine", 0, sigma=0.03)
                quarantine_covariates = beta_quarantine * quarantine_counts

                lambda_t = pm.Deterministic('Q Rate', pm.math.sigmoid(log_lambda_t + quarantine_covariates))

                ##################################################
                #### Community Q and UnQ Infection Generation ####
                ##################################################
                seed = pm.Exponential('Seed_local', 1)  # Scale of infection will be small (Local cases were quite high already)
                y0_local = tt.zeros(len_observed)
                y0_local = tt.set_subtensor(y0_local[0], seed)
                
                sus_0_comm = tt.zeros(len_observed)
                sus_0_comm = tt.set_subtensor(sus_0_comm[0], community_pop)
                
                # Apply recursively to populate tensor
                outputs_local, _ = theano.scan(community_calc_infected,
                        sequences=[tt.arange(1, len_observed), convolution_ready_gt,
                                theano.shared(imported_cases), lambda_t, effective_vax_rate],
                        outputs_info=[y0_local,sus_0_comm],
                        non_sequences=[r_t_local,sus_pop_comm_beta, community_pop, eps_t],
                        n_steps=len_observed-1,
                )


                infections_local = pm.Deterministic('infections_local_uncontained', outputs_local[0][-1])
                susceptible_community = pm.Deterministic('susceptible_community', outputs_local[1][-1])
                
                q_lambda_t = pm.Deterministic('infections_rate', (1-lambda_t)/lambda_t)

                infections_local_contained = pm.Deterministic('infections_local_contained', q_lambda_t*infections_local)
                

                ##############################################
                #### Calculate Implied Re(t) for analysis ####
                ##############################################
                # Calc implied R(t) for dorm and community
                exp_sus_rate_comm = tt.exp(sus_pop_comm_beta*susceptible_community/community_pop)
                
                # Calculate the implied number of infected
                implied_rt_inf_t_comm = pm.Deterministic('R_t_comm_implied' ,exp_sus_rate_comm*r_t_local)

                
                print("NO Vax Adjustment")
                infections_local_adjusted = pm.Deterministic('infections_local_adjusted', infections_local)
                infections_local_contained_adjusted = pm.Deterministic('infections_local_contained_adjusted',
                                                                    infections_local_contained)

                #############################
                #### Ascertainment Rates ####
                #############################
                                
                #### alpha Parameter - determines test rates from 3 sources ####

                
                # Local - UnQ
                log_alpha_unQ_t = pm.GaussianRandomWalk(
                    'log_alpha_unQ_t',
                    sigma=0.0055,
                    shape=len_observed)
                
                # Local - Q
                log_alpha_Q_t = pm.GaussianRandomWalk(
                    'log_alpha_Q_t',
                      # sigmoid(2) approx 0.88 corresponding to high prior belief that the people will be tested
                    sigma=0.0055,
                    shape=len_observed)
                

                # Hierarchical usage of test_counts
                beta_test_hier_mean = pm.Normal("beta_quarantine_hier_mean", 0, sigma=0.03)
                
                # prior for dorm, local unQ, Q
                # Fixed prior sd of 0.03


                test_unQ_tilde = pm.Normal("test_unQ_tilde", mu=0, sigma=1)
                beta_test_unQ = pm.Deterministic("beta_test_unQ", beta_test_hier_mean + 0.03 * test_unQ_tilde)

                test_Q_tilde = pm.Normal("test_Q_tilde", mu=0, sigma=1)
                beta_test_Q = pm.Deterministic("beta_test_Q", beta_test_hier_mean + 0.03 * test_Q_tilde)

                # Ascertainment Regime
                beta_ascertainment_regime_Q = pm.HalfNormal("ascertainment regime beta Q", sigma=5)  # For 2021 when stopped recording quarantines
                beta_ascertainment_regime_unQ = pm.HalfNormal("ascertainment regime beta unQ", sigma=1.5) # For 2021 when stopped recording quarantines
                beta_ascertainment_regime_unQ_2 = pm.HalfNormal("ascertainment regime beta unQ 2", sigma=0.5) # For 2022 when started counting more tests

                # Get alphas                
                alpha_unQ_t = pm.Deterministic('alpha_local_unQ_t', pm.math.sigmoid(1.5 + log_alpha_unQ_t +\
                                                                                    beta_test_unQ*test_counts -\
                                                                                    beta_ascertainment_regime_unQ*self.a_rate + \
                                                                                    beta_ascertainment_regime_unQ_2*self.a_rate2))
                
                #alpha_Q_t = pm.Deterministic('alpha_local_Q_t', pm.math.sigmoid(1.5 + log_alpha_Q_t + beta_test_Q*self.test_counts))
                alpha_Q_t = pm.Deterministic('alpha_local_Q_t', pm.math.sigmoid(6.5 + log_alpha_Q_t - beta_ascertainment_regime_Q*self.a_rate))
                
                """
                # Temp for debugging
                alpha_dorm_t=1
                alpha_unQ_t=1
                alpha_Q_t=1
                """

                # Onset - Delay Dist - Local
                t_p_delay_local = pm.Data("p_delay_local", p_delay_local)
                
                # Onset - Delay Dist - Qurantine
                t_p_delay_local_q = pm.Data('p_delay_local_qurantine', p_delay_q)
            
                
                
                # Test adjusted positive - Local
                test_adjusted_positive_local = pm.Deterministic(
                    "test adjusted positive local",
                    conv(infections_local_adjusted, t_p_delay_local, len(p_delay_local), len_observed)
                )
                test_adjusted_positive_jittered_local = pm.Deterministic('test_adjusted_positive_jit_local',
                                                            alpha_unQ_t*test_adjusted_positive_local + 0.01)
                
                
                # Test adjusted positive - Local Quarantined
                test_adjusted_positive_local_qurantined = pm.Deterministic(
                    "test adjusted positive local Qurantined",
                    conv(infections_local_contained_adjusted, t_p_delay_local_q, len(p_delay_q), len_observed)
                )
                test_adjusted_positive_jittered_local_qurantined = pm.Deterministic('test_adjusted_positive_jit_local_qurantined',
                                                            alpha_Q_t*test_adjusted_positive_local_qurantined + 0.01)
                

                print('chaanged')
                # Likelihood
                    
                
                if likelihood_fun == 'PO':
                    
                    # Suspect numerical errors here are occuring for logp function
                    mask_val_unq = 1  # default 1
                    mask_val_q = 1

                    mask = model_input_local_unquarantined.copy()
                    mask = (~np.isnan(mask)).astype('float')

                    model_input_local_unquarantined_mask = model_input_local_unquarantined.copy()
                    model_input_local_unquarantined_mask[np.isnan(model_input_local_unquarantined_mask)] = mask_val_unq

                    model_input_local_quarantined_mask = model_input_local_quarantined.copy()
                    model_input_local_quarantined_mask[np.isnan(model_input_local_quarantined_mask)] = mask_val_q



                    local_unq_likelihood = pm.Potential('Obs_local_LL',pm.Poisson.dist( 
                                    mu=test_adjusted_positive_jittered_local)\
                                    .logp(model_input_local_unquarantined_mask.values)*mask)
        
                    local_q_likelihood = pm.Potential('Obs_local_q_LL',pm.Poisson.dist( 
                                                        mu=test_adjusted_positive_jittered_local_qurantined)\
                                                    .logp(model_input_local_quarantined_mask.values)*mask)
                    
                elif likelihood_fun == 'NB':

                    print("NEGATATIVE BINOMIAL LL")
                    
                    # Dispersion param
                    dispersion_phi_local = pm.HalfNormal('LL_Dispersion_local', sigma=6)  # implies mean 4.787, std 3

                    mask_val_unq = 1  # default 1
                    mask_val_q = 1

                    mask = model_input_local_unquarantined.copy()
                    mask = (~np.isnan(mask)).astype('float')

                    model_input_local_unquarantined_mask = model_input_local_unquarantined.copy()
                    model_input_local_unquarantined_mask[np.isnan(model_input_local_unquarantined_mask)] = mask_val_unq

                    model_input_local_quarantined_mask = model_input_local_quarantined.copy()
                    model_input_local_quarantined_mask[np.isnan(model_input_local_quarantined_mask)] = mask_val_q

                    local_unq_likelihood = pm.Potential('Obs_local_LL',pm.NegativeBinomial.dist( 
                                    mu=test_adjusted_positive_jittered_local,
                                    alpha=dispersion_phi_local)\
                                    .logp(model_input_local_unquarantined_mask.values)*mask)

                    local_q_likelihood = pm.Potential('Obs_local_q_LL',pm.NegativeBinomial.dist( 
                                                        mu=test_adjusted_positive_jittered_local_qurantined,
                                                        alpha=dispersion_phi_local)\
                                                    .logp(model_input_local_quarantined_mask.values)*mask)

 
                
            return self.model

        else:
            raise ValueError('no')
            with pm.Model() as self.model:
                print("Starting Training")
                # r0 positive
                log_r_t = pm.GaussianRandomWalk(
                    'log_r_t',
                    sigma=0.035,
                    shape=len_observed)
                
                beta_list = []
                for i in range(num_NPIS):
                    beta_list.append(pm.Normal(f"b_{i}", 0, sigma=0.03))
                betas = pm.math.stack(beta_list, 0)
                #beta_intercept = pm.Normal('b_inter', 0, sigma=0.1)
                rt_covariates = pm.math.dot(np.array(NPIS_array), betas)
                
                
                # Form r_t as GRW + covariates
                r_t = pm.Deterministic('r_t', pm.math.exp(log_r_t + rt_covariates))
                
                    
                # Imported cases leak percent
                log_eps_t = pm.GaussianRandomWalk(
                    'log_eps_t',
                    sigma=0.035,
                    shape=len_observed)
                #eps_t = pm.Deterministic('eps_t', pm.math.exp(log_eps_t))
                eps_t = pm.Beta('eps_t', alpha=1, beta=1000)
                
                # Seed pop
                seed = pm.Exponential('Seed', 150)  # Scale of infection will be small
                y0 = tt.zeros(len_observed)
                y0 = tt.set_subtensor(y0[0], seed)
                # Apply recursively to populate tensor
                outputs, _ = theano.scan(
                        fn=lambda t, gt , i_cases, y, r_t: tt.set_subtensor(y[t], tt.sum(r_t*y*gt) + eps_t * i_cases),
                        sequences=[tt.arange(1, len_observed), convolution_ready_gt, theano.shared(imported_input)],
                        outputs_info=y0,
                        non_sequences=r_t,
                        n_steps=len_observed-1,
                )
                
                infections = pm.Deterministic('infections', outputs[-1])
                
                
                
                # Test observation
                t_p_delay = pm.Data("p_delay", p_delay)
                
                test_adjusted_positive = pm.Deterministic(
                    "test adjusted positive",
                    conv(infections, t_p_delay,len(p_delay), len_observed)
                )
                
                # For stability
                test_adjusted_positive_jittered = pm.Deterministic('test_adjusted_positive_jit',
                                                            test_adjusted_positive + 0)
                """
                # Accounts for number of tests 
                # Get number of tests
                tests = pm.Data("tests", tests_performed)
                exposure = pm.Deterministic(
                    "exposure",
                    pm.math.clip(tests_performed, 1000, 1e9)  # Hard code to 300 test a day for unobs period
                )
                
                positive = pm.Deterministic(
                    "positive", exposure * test_adjusted_positive_jittered
                )
                """
                
                # Likelihood
                if likelihood_fun == 'ZINB':
                    pm.ZeroInflatedNegativeBinomial('Obs', 
                            mu=test_adjusted_positive_jittered, 
                            alpha=pm.Gamma("alpha", mu=1, sigma=0.5),
                            psi = pm.Beta('psi', 2,2),                    
                            observed=model_input)    
                    
                elif likelihood_fun == 'NB':
                    pm.NegativeBinomial('Obs', 
                            mu=test_adjusted_positive_jittered, 
                            alpha=pm.Gamma("alpha", mu=1, sigma=0.5),
                            observed=model_input)
                
                elif likelihood_fun == 'PO':
                    print('hi')
                    pm.Poisson('Obs', 
                            mu=test_adjusted_positive_jittered,
                            observed=model_input)

            return self.model

    def sample(self,
              chains=2,
              tune=1500,
              draws=4000,
              t_accept=0.99, 
              init_choice='auto'):
        
        
        with self.model:
            self._trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=t_accept,
                chains=chains,#nuts={'target_accept':t_accept},
                init=init_choice
            )

        # Save model

        #save_fp = f"../Results/SM_Trace_Start_{end_date}_prediction_ahead_{prediction_t}_{tune_steps}_{train_steps}"
    
    def get_ppc(self):
        with self.model:
            self.ppc =  pm.sample_posterior_predictive(self._trace)


    def save_sample(self, save_fp):
        print("Saving to: ", save_fp)
        if not os.path.exists(save_fp):
            os.mkdir(save_fp)
        with self.model:
            pm.save_trace(self._trace, directory=save_fp, overwrite=True) 

    def plot_energy(self, save=True, save_fp=None):
        fig, ax = plt.subplots(figsize=(10,10))
        az.plot_energy(self._trace, ax=ax)

        if save:
            fig.savefig(save_fp, format='svg',  dpi=1200)


    def plot_latents(self, save=True, save_fp=None,actual_local_only=None, actual_dorm_only=None, show=False):
        # Plots latent infections and predicted positive cases

        if not self.separate_quarantine:
            fig, ax = plt.subplots(1,2, figsize=(20,10))

            ax[0].plot(self._trace['infections_local'].T, c='lightblue', alpha=0.1)
            ax[0].plot(self._trace['test_adjusted_positive_jit_local'].T, c='peachpuff', alpha=0.1)
            ax[0].plot(self._trace['infections_local'].mean(0), label='Local Infection')
            ax[0].plot(self._trace['test_adjusted_positive_jit_local'].mean(0), label='Local Positive')

            ax[1].plot(self._trace['infections_dorm'].T, c='peachpuff', alpha=0.1)
            ax[1].plot(self._trace['test_adjusted_positive_jit_dorm'].T, c='lightblue', alpha=0.1)
            ax[1].plot(self._trace['infections_dorm'].mean(0), label='Dorm Infection')
            ax[1].plot(self._trace['test_adjusted_positive_jit_dorm'].mean(0), label='Dorm Positive')


            if (actual_local_only is not None) and (actual_dorm_only is not None):
                ax[0].plot(np.concatenate([self.local_cases[0:self.end_date], actual_local_only]), c='r', alpha=0.5)
                ax[1].plot(np.concatenate([self.dorm_cases[0:self.end_date], actual_dorm_only]), c='r', alpha=0.5)
            else:
                ax[0].plot(self.local_cases, c='r', alpha=0.5)
                ax[1].plot(self.dorm_cases, c='r', alpha=0.5)  

            #ax.plot(imported_input, c='g')

            ax[0].axvline(x=self.end_date)
            ax[1].axvline(x=self.end_date)

            ax[0].legend()
            ax[1].legend()

            ax[0].set_title('Local')
            ax[1].set_title('Dorm')

            if show:
                plt.show()        

            if save:
                fig.savefig(save_fp, format='svg',  dpi=1200)
        
        else:
            fig, ax = plt.subplots(1,3, figsize=(30,10))

            # Un quarantined
            ax[0].plot(self._trace['infections_local_uncontained'].T, c='lightblue', alpha=0.1)
            ax[0].plot(self._trace['test_adjusted_positive_jit_local'].T, c='peachpuff', alpha=0.1)
            ax[0].plot(self._trace['infections_local_uncontained'].mean(0), label='Local Infection - Un Quarantined')
            ax[0].plot(self._trace['test_adjusted_positive_jit_local'].mean(0), label='Local Positive - Un Quarantined')

            # Quarantined
            ax[1].plot(self._trace['infections_local_contained'].T, c='lightblue', alpha=0.1)
            ax[1].plot(self._trace['test_adjusted_positive_jit_local_qurantined'].T, c='peachpuff', alpha=0.1)
            ax[1].plot(self._trace['infections_local_contained'].mean(0), label='Local Infection - Quarantined')
            ax[1].plot(self._trace['test_adjusted_positive_jit_local_qurantined'].mean(0), label='Local Positive - Quarantined')

            # Dorm
            if 'infections_dorm' in self._trace.varnames:
                ax[2].plot(self._trace['infections_dorm'].T, c='lightblue', alpha=0.1)
                ax[2].plot(self._trace['test_adjusted_positive_jit_dorm'].T, c='peachpuff', alpha=0.1)
                ax[2].plot(self._trace['infections_dorm'].mean(0), label='Dorm Infection')
                ax[2].plot(self._trace['test_adjusted_positive_jit_dorm'].mean(0), label='Dorm Positive')


            if (actual_local_only is not None) and (actual_dorm_only is not None):
                ax[0].plot(np.concatenate([self.local_cases[0:self.end_date], actual_local_only]), c='r', alpha=0.5)
                ax[1].plot(np.concatenate([self.dorm_cases[0:self.end_date], actual_dorm_only]), c='r', alpha=0.5)
            else:
                ax[0].plot(self.local_cases[0], c='r', alpha=0.5, label='Actual Cases Un Quarantined')
                ax[1].plot(self.local_cases[1], c='r', alpha=0.5, label='Actual Cases Quarantined')
                ax[2].plot(self.dorm_cases, c='r', alpha=0.5, label='Actual Cases Dorm')  

            #ax.plot(imported_input, c='g')

            ax[0].axvline(x=self.end_date)
            ax[1].axvline(x=self.end_date)
            ax[2].axvline(x=self.end_date)

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()

            ax[0].set_title('Local - Un Qurantined')
            ax[1].set_title('Local - Qurantined')
            ax[2].set_title('Dorm')

            if show:
                plt.show()        

            if save:
                fig.savefig(save_fp, format='svg',  dpi=1200)


    def plot_predictions(self, save=True, save_fp=None, actual_local_only=None, actual_dorm_only=None, show=False):
        
        if not self.ppc:
            self.get_ppc()

        if not self.separate_quarantine:
            fig, ax = plt.subplots(1,2, figsize=(20,10))
            #ax.plot(train_local_cases, c='b', label='Local Cases')
            #ax.plot(temp_dates, trace_10['Obs_missing'].T, c='silver', alpha=0.2)
            ax[0].plot(self.ppc['Obs_local'].T, c='lightblue', alpha=0.2)
            ax[0].plot(self.ppc['Obs_local'].mean(0), alpha=1, label='Mean Predicted Local Cases')

            ax[1].plot(self.ppc['Obs_dorm'].T, c='peachpuff', alpha=0.2)
            ax[1].plot(self.ppc['Obs_dorm'].mean(0), alpha=1, label='Mean Predicted Dorm Cases')

            # Actual cases
            ax[0].axvline(x=self.end_date)
            ax[1].axvline(x=self.end_date)

            if (actual_local_only is not None) and (actual_dorm_only is not None):
                ax[0].plot(np.concatenate([self.local_cases[0: self.end_date], actual_local_only]), alpha=0.5, label='Actual Local Cases Test')
                ax[1].plot(np.concatenate([self.dorm_cases[0: self.end_date], actual_dorm_only]), alpha=0.5, label='Actual Dorm Cases Test')
            else:
                ax[0].plot(self.local_cases, alpha=0.5, label='Actual Local Cases Test')
                ax[1].plot(self.dorm_cases, alpha=0.5, label='Actual Dorm Cases Test')


            ax[0].legend()
            ax[1].legend()

            ax[0].set_title("Predicted Cases - Local")
            ax[1].set_title("Predicted Cases - Dorm")

            if show:
                plt.show()

            if save:
                fig.savefig(save_fp, format='svg',  dpi=1200)
        
        else: 
            fig, ax = plt.subplots(1,3, figsize=(30,10))
            #ax.plot(train_local_cases, c='b', label='Local Cases')
            #ax.plot(temp_dates, trace_10['Obs_missing'].T, c='silver', alpha=0.2)
            ax[0].plot(self.ppc['Obs_local'].T, c='lightblue', alpha=0.2)
            ax[0].plot(self.ppc['Obs_local'].mean(0), alpha=1, label='Mean Predicted Local Un-Quarantined Cases')

            ax[1].plot(self.ppc['Obs_local_q'].T, c='lightblue', alpha=0.2)
            ax[1].plot(self.ppc['Obs_local_q'].mean(0), alpha=1, label='Mean Predicted Local Quarantined Cases')

            if 'infections_dorm' in self._trace.varnames:
                ax[2].plot(self.ppc['Obs_dorm'].T, c='lightblue', alpha=0.2)
                ax[2].plot(self.ppc['Obs_dorm'].mean(0), alpha=1, label='Mean Predicted Dorm Cases')
        
            # Actual cases
            ax[0].axvline(x=self.end_date)
            ax[1].axvline(x=self.end_date)
            ax[2].axvline(x=self.end_date)

            if (actual_local_only is not None) and (actual_dorm_only is not None):
                ax[0].plot(np.concatenate([self.local_cases[0: self.end_date], actual_local_only]), alpha=0.5, label='Actual Local Cases Test')
                ax[1].plot(np.concatenate([self.dorm_cases[0: self.end_date], actual_dorm_only]), alpha=0.5, label='Actual Dorm Cases Test')
            else:
                ax[0].plot(self.local_cases[0],c='red', alpha=0.5, label='Actual Local Cases Un-Qurantined Train')
                ax[1].plot(self.local_cases[1],c='red', alpha=0.5, label='Actual Local Cases Qurantined Cases Train')
                ax[2].plot(self.dorm_cases, c='red',alpha=0.5, label='Actual Dorm Cases Train')


            ax[0].legend()
            ax[1].legend()
            ax[2].legend()

            ax[0].set_title("Predicted Cases - Local Un Quarantined")
            ax[1].set_title("Predicted Cases - Local Quarantined")
            ax[2].set_title("Predicted Cases - Dorm")

            if show:
                plt.show()

            if save:
                fig.savefig(save_fp, format='svg',  dpi=1200)

                
    def save_data(self, save_dir):
        if not self.ppc:
            self.get_ppc()

        save_fp_trace = os.path.join(save_dir, 'trace.pkl')
        with open(save_fp_trace, 'wb') as f:
            pickle.dump(self._trace, f)


    def plot_rt(self, save=True, save_fp=None, show=False):
        
        fig, ax = plt.subplots(1,2, figsize=(20,10))

        ax[0].plot(self._trace['r_t_local'].T, c='silver', alpha=0.1)
        ax[0].plot(self._trace['r_t_local'].mean(0), c='black', alpha=1)
        ax[0].axhline(y=1, alpha=0.5)
        ax[0].axvline(x=self.end_date, alpha=0.5)
        ax[0].set_title("Effective Re(t) - Local")

        if 'infections_dorm' in self._trace.varnames:
            ax[1].plot(self._trace['r_t_foreign'].T, c='silver', alpha=0.1)
            ax[1].plot(self._trace['r_t_foreign'].mean(0), c='black', alpha=1)
            ax[1].axhline(y=1, alpha=0.5)
            ax[1].axvline(x=self.end_date, alpha=0.5)
            ax[1].set_title("Effective Re(t) - Dorm")

        if save:
            fig.savefig(save_fp, format='svg',  dpi=1200)
        
        if show:
            plt.show()
    
    def plot_alpha_t(self, save=True, save_fp=None, show=False):
        """ Plots alpha param for dorms, local unQ, and local Q, alpha being the testing rate

        Args:
            save (bool, optional): [description]. Defaults to True.
            save_fp ([type], optional): [description]. Defaults to None.
        """
        fig, ax = plt.subplots(1,3, figsize=(30,10))

        if 'infections_dorm' in self._trace.varnames:
            ax[0].plot(self._trace['alpha_dorm_t'].T, c='silver', alpha=0.1)
            ax[0].plot(self._trace['alpha_dorm_t'].mean(0), c='black', alpha=1)
            ax[0].axhline(y=1, alpha=0.5)
            ax[0].axvline(x=self.end_date, alpha=0.5)
            ax[0].set_title('Alpha(t) - Dorm')
        
        ax[1].plot(self._trace['alpha_local_unQ_t'].T, c='silver', alpha=0.1)
        ax[1].plot(self._trace['alpha_local_unQ_t'].mean(0), c='black', alpha=1)
        ax[1].axhline(y=1, alpha=0.5)
        ax[1].axvline(x=self.end_date, alpha=0.5)
        ax[1].set_title('Alpha(t) - UnQ')

        ax[2].plot(self._trace['alpha_local_Q_t'].T, c='silver', alpha=0.1)
        ax[2].plot(self._trace['alpha_local_Q_t'].mean(0), c='black', alpha=1)
        ax[2].axhline(y=1, alpha=0.5)
        ax[2].axvline(x=self.end_date, alpha=0.5)
        ax[2].set_title('Alpha(t) - Q')

        if save:
            fig.savefig(save_fp, format='svg',  dpi=1200)
        
        if show:
            plt.show()
    
    def plot_quarantine_rate(self, save=True, save_fp=None, show=False):

        fig, ax = plt.subplots(figsize=(20,20))

        ax.plot(self._trace['Q Rate'].T, c='silver', alpha=0.1)
        ax.plot(self._trace['Q Rate'].mean(0), c='black', alpha=1)
        ax.axhline(y=1, alpha=0.5)
        ax.axvline(x=self.end_date, alpha=0.5)
        ax.set_title('Qurantine Rate')

        if save:
            fig.savefig(save_fp, format='svg', dpi=1200)

        if show:
            plt.show()


    def get_accuracy(self, local_gt=None, dorm_gt=None, save=True, save_fp=None):

        if not self.ppc:
            self.get_ppc()

        if not self.separate_quarantine:
            output = {}

            # Extract posterior
            dorms_pred_mean = self.ppc['Obs_dorm'].mean(0)
            local_pred_mean = self.ppc['Obs_local'].mean(0)

            # Train
            dorms_pred_train = dorms_pred_mean[:self.end_date]
            local_pred_train = local_pred_mean[:self.end_date]

            # Test
            dorms_pred_test = dorms_pred_mean[self.end_date::]
            local_pred_test = local_pred_mean[self.end_date::]

            # Extract ground truth train
            dorms_train = self.dorm_cases[:self.end_date]
            locals_train = self.local_cases[:self.end_date]

            # Metrics
            rmse_dorm_train = np.sqrt(np.mean((dorms_train-dorms_pred_train)**2))
            mae_dorm_train = np.mean(np.abs(dorms_train - dorms_pred_train))

            rmse_local_train = np.sqrt(np.mean((locals_train-local_pred_train)**2))
            mae_local_train = np.mean(np.abs(locals_train - local_pred_train))

            output['dorm_train'] = (rmse_dorm_train, mae_dorm_train)
            output['local_train'] = (rmse_local_train, mae_local_train)

            # If test
            if (local_gt is not None) and (dorm_gt is not None):
                rmse_dorm_test = np.sqrt(np.mean((dorm_gt-dorms_pred_test)**2))
                mae_dorm_test = np.mean(np.abs(dorm_gt - dorms_pred_test))

                rmse_local_test = np.sqrt(np.mean((local_gt-local_pred_test)**2))
                mae_local_test = np.mean(np.abs(local_gt - local_pred_test))

                output['dorm_test'] = (rmse_dorm_test, mae_dorm_test)
                output['local_test'] = (rmse_local_test, mae_local_test)

            if save:
                with open(save_fp, 'wb') as f:
                    pickle.dump(output, f)

            return output
        
        else:
            output = {}

            # Extract posterior
            dorms_pred_mean = self.ppc['Obs_dorm'].mean(0)
            local_unQ_pred_mean = self.ppc['Obs_local'].mean(0)
            local_Q_pred_mean = self.ppc['Obs_local_q'].mean(0)

            # Train
            dorms_pred_train = dorms_pred_mean[:self.end_date]
            local_unQ_pred_train = local_unQ_pred_mean[:self.end_date]
            local_Q_pred_train = local_Q_pred_mean[:self.end_date]

            # Test
            dorms_pred_test = dorms_pred_mean[self.end_date::]
            local_unQ_pred_test = local_unQ_pred_mean[self.end_date::]
            local_Q_pred_test= local_Q_pred_mean[self.end_date::]

            # Extract ground truth train
            dorms_train = self.dorm_cases[:self.end_date]
            locals_unQ_train = self.local_cases[0][:self.end_date]
            locals_Q_train = self.local_cases[1][:self.end_date]

            # Metrics
            rmse_dorm_train = np.sqrt(np.mean((dorms_train-dorms_pred_train)**2))
            mae_dorm_train = np.mean(np.abs(dorms_train - dorms_pred_train))

            rmse_local_unQ_train = np.sqrt(np.mean((locals_unQ_train-local_unQ_pred_train)**2))
            mae_local_unQ_train = np.mean(np.abs(locals_unQ_train - local_unQ_pred_train))

            rmse_local_Q_train = np.sqrt(np.mean((locals_Q_train-local_Q_pred_train)**2))
            mae_local_Q_train = np.mean(np.abs(locals_Q_train - local_Q_pred_train))

            output['dorm_train'] = (rmse_dorm_train, mae_dorm_train)
            output['local_unQ_train'] = (rmse_local_unQ_train, mae_local_unQ_train)
            output['local_Q_train'] = (rmse_local_Q_train, mae_local_Q_train)

            # If test
            if (local_gt is not None) and (dorm_gt is not None):
                rmse_dorm_test = np.sqrt(np.mean((dorm_gt-dorms_pred_test)**2))
                mae_dorm_test = np.mean(np.abs(dorm_gt - dorms_pred_test))

                rmse_local_test = np.sqrt(np.mean((local_gt-local_pred_test)**2))
                mae_local_test = np.mean(np.abs(local_gt - local_pred_test))

                output['dorm_test'] = (rmse_dorm_test, mae_dorm_test)
                output['local_test'] = (rmse_local_test, mae_local_test)

            if save:
                with open(save_fp, 'wb') as f:
                    pickle.dump(output, f)

            return output


    def get_predictions(self):
        if not self.ppc:
            self.get_ppc()



    def save_infected(self, date_range, folder):
        """ Gets infected counts for all and saves """

        community_infections_samples = self._trace['infections_local_adjusted'] + \
                                self._trace['infections_local_contained_adjusted']

        if 'infections_dorm' in self._trace.varnames:
            dorm_infections_samples = self._trace['infections_dorm_adjusted']
        else:
            dorm_infections_samples = np.zeros([1000, self.len_observed])

        com_infections_agg = np.median(community_infections_samples, 0)
        bounds_com_inf_95 = az.hdi(community_infections_samples, hdi_prob=0.95)
        bounds_com_inf_67 = az.hdi(community_infections_samples, hdi_prob=0.67)

        
        dorm_infections_agg = np.median(dorm_infections_samples, 0)
        bounds_dorm_inf_95 = az.hdi(dorm_infections_samples, hdi_prob=0.95)
        bounds_dorm_inf_67 = az.hdi(dorm_infections_samples, hdi_prob=0.67)

        comm_inf_data = pd.DataFrame({
                        'Date': date_range.values,
                        'comm_infections_median': com_infections_agg,
                        'comm_infections_lb_95': bounds_com_inf_95[:,0],
                        'comm_infections_ub_95': bounds_com_inf_95[:,1],
                        'comm_infections_lb_67': bounds_com_inf_67[:,0],
                        'comm_infections_ub_67': bounds_com_inf_67[:,1]
                        })
        
        dorm_inf_data = pd.DataFrame({
                        'Date': date_range.values,
                        'dorm_infections_median': dorm_infections_agg,
                        'dorm_infections_lb_95': bounds_dorm_inf_95[:,0],
                        'dorm_infections_ub_95': bounds_dorm_inf_95[:,1],
                        'dorm_infections_lb_67': bounds_dorm_inf_67[:,0],
                        'dorm_infections_ub_67': bounds_dorm_inf_67[:,1]
                        })
        
        com_fp = os.path.join(folder, 'comm_infections.csv')
        dorm_fp = os.path.join(folder, 'dorm_infections.csv')

        comm_inf_data.to_csv(com_fp, index=False)
        dorm_inf_data.to_csv(dorm_fp, index=False)

        self.community_infected_results = comm_inf_data
        self.dorm_infetected_results = dorm_inf_data
    

    def save_community_cases(self, date_range, folder):

        if self.likelihood_fun=='NB':
            print("Negative Binomial LL")
            unlinked_cases_samples = sample_nb_trace(self._trace['test_adjusted_positive_jit_local'],
                                                    self._trace['LL_Dispersion_local'])
            
            linked_cases_samples = sample_nb_trace(self._trace['test_adjusted_positive_jit_local_qurantined'],
                                                    self._trace['LL_Dispersion_local'])
            
            if 'infections_dorm' in self._trace.varnames:
                dorm_cases_samples = sample_nb_trace(self._trace['test_adjusted_positive_jit_dorm'],
                                                        self._trace['LL_Dispersion_dorm'])
            else:
                dorm_cases_samples = np.zeros([1000, self.len_observed])
            
            combined_community_samples = sample_nb_trace(self._trace['test_adjusted_positive_jit_local'] +\
                                                    self._trace['test_adjusted_positive_jit_local_qurantined'],
                                                        self._trace['LL_Dispersion_local'])
        
        else:
            print("Poisson LL")
            unlinked_cases_samples = sample_poisson_trace(self._trace['test_adjusted_positive_jit_local'])
            linked_cases_samples = sample_poisson_trace(self._trace['test_adjusted_positive_jit_local_qurantined'])
            combined_community_samples = sample_poisson_trace(self._trace['test_adjusted_positive_jit_local'] +\
                                                            self._trace['test_adjusted_positive_jit_local_qurantined'])
            
            if 'infections_dorm' in self._trace.varnames:
                dorm_cases_samples = sample_poisson_trace(self._trace['test_adjusted_positive_jit_dorm'])
            else:
                dorm_cases_samples = np.zeros([1000, self.len_observed])

        unlinked_cases = np.median(unlinked_cases_samples ,0)
        linked_cases = np.median(linked_cases_samples ,0)
        dorm_cases = np.median(dorm_cases_samples ,0)
        combined_community = np.median(combined_community_samples, 0)


        # CIs
        bounds_unlinked_95 = az.hdi(unlinked_cases_samples, hdi_prob=0.95)
        bounds_unlinked_67 = az.hdi(unlinked_cases_samples, hdi_prob=0.67)

        bounds_linked_95 = az.hdi(linked_cases_samples, hdi_prob=0.95)
        bounds_linked_67 = az.hdi(linked_cases_samples, hdi_prob=0.67)

        bounds_dorm_95 = az.hdi(dorm_cases_samples, hdi_prob=0.95)
        bounds_dorm_67 = az.hdi(dorm_cases_samples, hdi_prob=0.67)

        bounds_total_com_95 = az.hdi(combined_community_samples, hdi_prob=0.95)
        bounds_total_com_67 = az.hdi(combined_community_samples, hdi_prob=0.67)

        case_titles = ['Unlinked', 'Linked', 'Dormitory', 'Community Total']
        case_medians = [unlinked_cases, linked_cases, dorm_cases, combined_community]
        case_95 = [bounds_unlinked_95, bounds_linked_95, bounds_dorm_95, bounds_total_com_95]
        case_67 = [bounds_unlinked_67, bounds_linked_67, bounds_dorm_67, bounds_total_com_67]

        outputs_coll = []
        for ct, cm, c95, c67 in zip(case_titles, case_medians, case_95, case_67):

            fn = os.path.join(folder, 'Case_Counts_Samples_'+ct+'_.csv')
            
            output_info = pd.DataFrame({
                        'Date':date_range.values,
                            ct+'_median': cm,
                        ct+'_lb_95': c95[:,0],
                        ct+'_ub_95': c95[:,1],
                        ct+'_lb_67': c67[:,0],
                        ct+'_ub_67': c67[:,1],
                        })
            output_info.to_csv(fn, index=False)
            outputs_coll.append(output_info)

        outputs_coll = merge_frames(outputs_coll, colname='Date')
        self.case_counts_results = outputs_coll

    

    def save_rt_scores(self, date_range, folder, agg_type_walks='median'):



        bounds_q_95 = az.hdi(1-self._trace['Q Rate'], hdi_prob=0.95)
        bounds_q_67 = az.hdi(1-self._trace['Q Rate'], hdi_prob=0.67)
        if agg_type_walks=='mean':
            q_rate = (1-self._trace['Q Rate']).mean(0)  
        else:
            q_rate = np.median((1-self._trace['Q Rate']),0)


        bounds_rt_95_unadj = az.hdi(self._trace['r_t_local'], hdi_prob=0.95)
        bounds_rt_67_unadj = az.hdi(self._trace['r_t_local'], hdi_prob=0.67)
        if agg_type_walks=='mean':
            rt_com_unadj = self._trace['r_t_local'].mean(0)  
        else:
            rt_com_unadj = np.median(self._trace['r_t_local'],0)


        bounds_rt_95 = az.hdi(self._trace['R_t_comm_implied'], hdi_prob=0.95)
        bounds_rt_67 = az.hdi(self._trace['R_t_comm_implied'], hdi_prob=0.67)
        if agg_type_walks=='mean':
            rt_com = self._trace['R_t_comm_implied'].mean(0)  
        else:
            rt_com = np.median(self._trace['R_t_comm_implied'],0)


        if 'infections_dorm' in self._trace.varnames:
            bounds_rt_95_dorm_unadj = az.hdi(self._trace['r_t_foreign'], hdi_prob=0.95)
            bounds_rt_67_dorm_unadj = az.hdi(self._trace['r_t_foreign'], hdi_prob=0.67)
            if agg_type_walks=='mean':
                rt_dorm_unadj = self._trace['r_t_foreign'].mean(0)  
            else:
                rt_dorm_unadj = np.median(self._trace['r_t_foreign'],0)

            bounds_rt_95_dorm = az.hdi(self._trace['R_t_dorm_implied'], hdi_prob=0.95)
            bounds_rt_67_dorm = az.hdi(self._trace['R_t_dorm_implied'], hdi_prob=0.67)
            if agg_type_walks=='mean':
                rt_dorm = self._trace['R_t_dorm_implied'].mean(0)  
            else:
                rt_dorm = np.median(self._trace['R_t_dorm_implied'],0)
        else:
            rt_dorm = np.zeros(self.len_observed)
            rt_dorm_unadj = np.zeros(self.len_observed)
            bounds_rt_95_dorm_unadj = az.hdi(np.zeros([1000,self.len_observed]), hdi_prob=0.95) 
            bounds_rt_67_dorm_unadj = az.hdi(np.zeros([1000,self.len_observed]), hdi_prob=0.95)
            bounds_rt_95_dorm = az.hdi(np.zeros([1000,self.len_observed]), hdi_prob=0.95)
            bounds_rt_67_dorm = az.hdi(np.zeros([1000,self.len_observed]), hdi_prob=0.95)

        # Combine Q R(t)
        samples_qrt = (self._trace['Q Rate']) * (self._trace['R_t_comm_implied'])

        bounds_qrt_95 = az.hdi(samples_qrt, hdi_prob=0.95)
        bounds_qrt_67 = az.hdi(samples_qrt, hdi_prob=0.67)
        if agg_type_walks=='mean':
            qrt_com = samples_qrt.mean(0)  
        else:
            qrt_com = np.median(samples_qrt,0)

        # Ascertainment rate - Dorm 
        if 'infections_dorm' in self._trace.varnames:
            bounds_ascertainment_95_dorm = az.hdi(self._trace['alpha_dorm_t'], hdi_prob=0.95)
            bounds_ascertainment_67_dorm = az.hdi(self._trace['alpha_dorm_t'], hdi_prob=0.67)
            if agg_type_walks=='mean':
                ascertainment_dorm = self._trace['alpha_dorm_t'].mean(0)  
            else:
                ascertainment_dorm = np.median(self._trace['alpha_dorm_t'],0)
        else:
            bounds_ascertainment_95_dorm = az.hdi(np.zeros([1000,self.len_observed]), hdi_prob=0.95) 
            bounds_ascertainment_67_dorm = az.hdi(np.zeros([1000,self.len_observed]), hdi_prob=0.95) 
            ascertainment_dorm = np.zeros(self.len_observed) 

        # Ascertainment rate - Un Quaratined 
        bounds_ascertainment_95_unQ = az.hdi(self._trace['alpha_local_unQ_t'], hdi_prob=0.95)
        bounds_ascertainment_67_unQ = az.hdi(self._trace['alpha_local_unQ_t'], hdi_prob=0.67)
        if agg_type_walks=='mean':
            ascertainment_unQ = self._trace['alpha_local_unQ_t'].mean(0)  
        else:
            ascertainment_unQ = np.median(self._trace['alpha_local_unQ_t'],0)

        # Ascertainment rate - Quaratined 
        bounds_ascertainment_95_Q = az.hdi(self._trace['alpha_local_Q_t'], hdi_prob=0.95)
        bounds_ascertainment_67_Q = az.hdi(self._trace['alpha_local_Q_t'], hdi_prob=0.67)
        if agg_type_walks=='mean':
            ascertainment_Q = self._trace['alpha_local_Q_t'].mean(0)  
        else:
            ascertainment_Q = np.median(self._trace['alpha_local_Q_t'],0)


        # Save
        fn = os.path.join(folder, 'rates.csv')

        case_titles = ['Quarantine Rate',
                      'R_t_comm_unadj',
                      'R_t_comm',
                      'R_t_Q_comm_rate',
                      'R_t_dorm_unadj',
                      'R_t_dorm',
                      'Ascertainment_dorm',
                      'Ascertainment_unQuarantined',
                      'Ascertainment_Quarantined']

        case_medians = [q_rate, rt_com_unadj, rt_com, qrt_com, rt_dorm_unadj, rt_dorm, ascertainment_dorm, ascertainment_unQ, ascertainment_Q]
        case_95 = [bounds_q_95, bounds_rt_95_unadj, bounds_rt_95, bounds_qrt_95, bounds_rt_95_dorm_unadj, bounds_rt_95_dorm, bounds_ascertainment_95_dorm, bounds_ascertainment_95_unQ, bounds_ascertainment_95_Q] 
        case_67 = [bounds_q_67, bounds_rt_67_unadj, bounds_rt_67, bounds_qrt_67, bounds_rt_67_dorm_unadj, bounds_rt_67_dorm, bounds_ascertainment_67_dorm, bounds_ascertainment_67_unQ, bounds_ascertainment_67_Q]
        temp = []

        for ct, cm, c95, c67 in zip(case_titles, case_medians, case_95, case_67):
            
            output_info = pd.DataFrame({
                        #'Date':period_dates.values,
                            ct+'_median': cm,
                        ct+'_lb_95': c95[:,0],
                        ct+'_ub_95': c95[:,1],
                        ct+'_lb_67': c67[:,0],
                        ct+'_ub_67': c67[:,1],
                        })
            temp.append(output_info)

        rate_table = pd.concat(temp,1)
        rate_table['Date'] = date_range.values
        rate_table.to_csv(fn, index=False)

        self.rate_results = rate_table
    
    def output_predictions_sumamrised(self, folder):
        
        all_info = [self.community_infected_results,
                    self.dorm_infetected_results,
                    self.case_counts_results,
                    self.rate_results]

        all_info_df = merge_frames(all_info, colname='Date')
        community_cases = self.local_cases[0] + self.local_cases[1]
        
        all_info_df = all_info_df[['Date',
        
                                'Quarantine Rate_median',
                                'Quarantine Rate_lb_95',
                                'Quarantine Rate_ub_95',
                                'Quarantine Rate_lb_67',
                                'Quarantine Rate_ub_67',

                                'R_t_comm_unadj_median',
                                'R_t_comm_unadj_lb_95',
                                'R_t_comm_unadj_ub_95',
                                'R_t_comm_unadj_lb_67',
                                'R_t_comm_unadj_ub_67',

                                'R_t_comm_median',
                                'R_t_comm_lb_95',
                                'R_t_comm_ub_95',
                                'R_t_comm_lb_67',
                                'R_t_comm_ub_67',

                                'R_t_Q_comm_rate_median',
                                'R_t_Q_comm_rate_lb_95',
                                'R_t_Q_comm_rate_ub_95',
                                'R_t_Q_comm_rate_lb_67',
                                'R_t_Q_comm_rate_ub_67',

                                'R_t_dorm_unadj_median',
                                'R_t_dorm_unadj_lb_95',
                                'R_t_dorm_unadj_ub_95',
                                'R_t_dorm_unadj_lb_67',
                                'R_t_dorm_unadj_ub_67',

                                'R_t_dorm_median',
                                'R_t_dorm_lb_95',
                                'R_t_dorm_ub_95',
                                'R_t_dorm_lb_67',
                                'R_t_dorm_ub_67',

                                'Ascertainment_dorm_median',
                                'Ascertainment_dorm_lb_95',
                                'Ascertainment_dorm_ub_95',
                                'Ascertainment_dorm_lb_67',
                                'Ascertainment_dorm_ub_95',
                                
                                'Ascertainment_unQuarantined_median',
                                'Ascertainment_unQuarantined_lb_95',
                                'Ascertainment_unQuarantined_ub_95',
                                'Ascertainment_unQuarantined_lb_67',
                                'Ascertainment_unQuarantined_ub_95',

                                'Ascertainment_Quarantined_median',
                                'Ascertainment_Quarantined_lb_95',
                                'Ascertainment_Quarantined_ub_95',
                                'Ascertainment_Quarantined_lb_67',
                                'Ascertainment_Quarantined_ub_95',
                                
                                'comm_infections_median',
                                'comm_infections_lb_95',
                                'comm_infections_ub_95',
                                'comm_infections_lb_67',
                                'comm_infections_ub_67',

                                'Community Total_median',
                                'Community Total_lb_95',
                                'Community Total_ub_95',
                                'Community Total_lb_67',
                                'Community Total_ub_67']]
        # Add in case counts
        all_info_df['Community Total Cases'] = community_cases
        
        # Save
        save_fp = os.path.join(folder, 'projections.csv')
        all_info_df.to_csv(save_fp, index=False)

        

    def plot_latents_new(self, date_range, save_fp, save, show=False):

        gt_unlinked = self.local_cases[0]
        gt_linked = self.local_cases[1]
        gt_dorm = self.dorm_cases 

        gt_vals_len = len(gt_dorm)

        infected_names = ['infections_local_uncontained','infections_local_contained','infections_dorm']
        infected_adj = ['infections_local_adjusted','infections_local_contained_adjusted','infections_dorm_adjusted']
        #infected_adj = ['infections_local_uncontained','infections_local_contained','infections_dorm']


        infected_labels = ['Local Infection - Un Quarantined - Unadjusted',
                        'Local Infection - Linked - Unadjusted',
                        'Dorm Infection - Unadjusted']

        infected_labels_adj = ['Predicted Infection - Local Un Quarantined',
                            'Predicted Infection - Local Quarantined',
                            'Predicted Infection - Dorm']

        case_names = ['test_adjusted_positive_jit_local', 'test_adjusted_positive_jit_local_qurantined',
                    'test_adjusted_positive_jit_dorm']

        case_labels = ['Predicted Positive - Local Un Quarantined',
                    'Predicted Positive - Local Quarantined',
                    'Dorm Predicted Positive']

        actual_vectors = [gt_unlinked.values, gt_linked.values, gt_dorm.values]
        actual_labels = ['Actual Positive - Local Un Quarantined',
                        'Actual Positive - Local Quarantined',
                        'Actual Positive - Dorm']

        plot_names = ['Local - Un Quarantined', 'Local - Quarantined', 'Dorm']

        plt_param_zip = list(zip(range(3), infected_names, infected_labels,infected_adj,infected_labels_adj,
                            case_names, case_labels,
                            actual_vectors, actual_labels,
                            plot_names))

        # Plots
        plt.rcParams.update({'font.size': 24})
        fig, ax = plt.subplots(3,1,figsize=(30,30))
        agg_type='median'


        if 'infections_dorm' in self._trace.varnames:
            endpoint = 3
        else:
            endpoint = 2

        # Plot 
        for i, i_n, i_l, i_n_a, i_l_a, c_n, c_l, a_v, a_l, p_name in plt_param_zip[:endpoint]:

            ax[i].set_xlim(date_range.values[150], date_range.values[-1])
            #ax[i].set_ylim(0, 2000)


            bounds_infected_adjusted = az.hdi(self._trace[i_n_a], hdi_prob=0.95)
            bounds_cases = az.hdi(self._trace[c_n], hdi_prob=0.95)

            # Un quarantined
            #ax.plot(lolz['infections_local_uncontained'].T, c='lightblue', alpha=0.1)
            #ax.plot(lolz['test_adjusted_positive_jit_local'].T, c='peachpuff', alpha=0.1)

            if agg_type=='mean':
                agg_preds_case = self._trace[c_n].mean(0)
                agg_preds_inf_adj = self._trace[i_n_a].mean(0)
            else:
                agg_preds_case = np.median(self._trace[c_n],0)
                agg_preds_inf_adj = np.median(self._trace[i_n_a],0)
            
            
            # Get max val infections
            max_val_inf = np.max(agg_preds_inf_adj)
            max_val_bounds = np.max(bounds_infected_adjusted[:,1])
            
            if (max_val_bounds - max_val_inf) < 1500:
                
                if max_val_bounds < 1000:
                    upper_bound = max_val_bounds + 200
                else:
                    upper_bound = max_val_bounds + 1000
            
            else:
                upper_bound = (2/3)*(max_val_bounds - max_val_inf) + max_val_inf
            
            ax[i].plot(date_range.values, agg_preds_inf_adj,
                    label=i_l_a,
                    color='blue',
                    linewidth=3)


            ax[i].plot(date_range.values, agg_preds_case,
                    label=c_l,
                    color='red',
                    linewidth=3)

            ax[i].fill_between(date_range.values, bounds_infected_adjusted[:,0], bounds_infected_adjusted[:,1], color='lightblue', alpha=.4)
            ax[i].fill_between(date_range.values, bounds_cases[:,0], bounds_cases[:,1], color='salmon', alpha=.5)

            ax[i].plot(date_range.values[:gt_vals_len],a_v, c='k', alpha=1, label=a_l, linewidth=4,marker='*',
                    markersize=10)

            ax[i].axvline(x=date_range.values[-1*self.prediction_t-1], alpha=1, c='k')
            
            train_end_date = date_range.values[-1*self.prediction_t-1]
            
            ax[i].legend(loc='upper left')

            ax[i].set_title(p_name)
            
            
            myFmt = mdates.DateFormatter("%b-%d")
            ax[i].xaxis.set_major_formatter(myFmt)

            ax[i].grid(True)
            ax[i].xaxis.set_major_locator(mdates.DayLocator(pd.Timestamp(train_end_date).day))
            #ax[i, case_no].xaxis.set_major_locator(mdates.DayLocator(30))

            #ax[i, case_no].axes.get_yaxis().set_visible(False)
            #ax[i, case_no].set_frame_on(False)

            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(True)
            ax[i].spines['left'].set_visible(False)
            
            
            ax[i].set_ylim(0, upper_bound)


        if save:
            full_fp = os.path.join(save_fp, 'latent_plots_new.svg')
            fig.savefig(full_fp, format='svg',  dpi=1200)
        else:
            plt.show()

    
    def plot_rt_q_new(self, date_range, save_fp, save, show=False):

        fig, ax = plt.subplots(4,1,figsize=(30,35))
        agg_type_walks='median'

        bounds_1 = az.hdi(self._trace['R_t_comm_implied'], hdi_prob=0.95)
        if agg_type_walks=='mean':
            rtc_implied = (self._trace['R_t_comm_implied']).mean(0)  
        else:
            rtc_implied = np.median((self._trace['R_t_comm_implied']),0)
        ax[0].fill_between(date_range.values, bounds_1[:,0], bounds_1[:,1], color='silver', alpha=.6)
        ax[0].plot(date_range.values,rtc_implied, c='black', alpha=1, linewidth=3)
        ax[0].axhline(y=1, alpha=0.6)
        ax[0].axvline(x=date_range.values[-1*self.prediction_t-1], alpha=1, c='k')
        #ax[0].set_title("Quarantine Rate")
        ax[0].text(0.3, 0.95, 'Implied Re(t) Community', 
                transform=ax[0].transAxes, ha="left",fontsize=45)


        if 'rt_covariate_series' in self._trace.varnames:
            bounds_2 = az.hdi(self._trace['rt_covariate_series'], hdi_prob=0.95)
            if agg_type_walks=='mean':
                rt_com_covar = self._trace['rt_covariate_series'].mean(0)  
            else:
                rt_com_covar = np.median(self._trace['rt_covariate_series'],0)
                
            ax[2].fill_between(date_range.values, bounds_2[:,0], bounds_2[:,1], color='silver', alpha=.6)
            ax[2].plot(date_range.values, rt_com_covar, c='black', alpha=1, linewidth=3)
            ax[2].axhline(y=1, alpha=0)
            ax[2].axvline(x=date_range.values[-1*self.prediction_t-1], alpha=1, c='k')
            #ax[2].set_title("Effective Re(t) - NPIs Covariate Component")
            ax[2].text(0.3, 0.95, 'Contribution to Re(t) from NPIs', 
                    transform=ax[2].transAxes, ha="left",fontsize=45)


        bounds_3 = az.hdi(self._trace['r_t_local'], hdi_prob=0.95)
        if agg_type_walks=='mean':
            rt_com = self._trace['r_t_local'].mean(0)  
        else:
            rt_com = np.median(self._trace['r_t_local'],0)
            
        ax[1].fill_between(date_range.values, bounds_3[:,0], bounds_3[:,1], color='silver', alpha=.6)
        ax[1].plot(date_range.values, rt_com, c='black', alpha=1, linewidth=3)
        ax[1].axhline(y=1, alpha=0.1)
        ax[1].axvline(x=date_range.values[-1*self.prediction_t-1], alpha=1, c='k')
        #ax[1].set_title("Effective Re(t) - Local")
        ax[1].text(0.3, 0.95, 'Re(t) for Community Unlinked Cases', 
                transform=ax[1].transAxes, ha="left",fontsize=45)


        susceptibles_rt_addition = np.multiply(self._trace['b_comm_susc'].reshape(-1,1),
                                            (1/self.community_pop)*self._trace['susceptible_community'])

        bounds_beta_sus = az.hdi(susceptibles_rt_addition, hdi_prob=0.95)
        if agg_type_walks=='mean':
            beta_sus = susceptibles_rt_addition.mean(0)  
        else:
            beta_sus = np.median(susceptibles_rt_addition,0)
            
        ax[3].fill_between(date_range.values, bounds_beta_sus[:,0], bounds_beta_sus[:,1], color='silver', alpha=.6)
        ax[3].plot(date_range.values, beta_sus, c='black', alpha=1, linewidth=3)
        ax[3].axhline(y=1, alpha=0)
        ax[3].axvline(x=date_range.values[-1*self.prediction_t-1], alpha=1, c='k')
        #ax[2].set_title("Effective Re(t) - NPIs Covariate Component")
        ax[3].text(0.3, 0.95, 'Beta - Susceptible Community Size', 
                transform=ax[3].transAxes, ha="left",fontsize=45)

        train_end_date = date_range.values[-1*self.prediction_t-1]
        for i in range(4):
            myFmt = mdates.DateFormatter("%b-%d")
            ax[i].xaxis.set_major_formatter(myFmt)
            ax[i].grid(True)
            ax[i].xaxis.set_major_locator(mdates.DayLocator(pd.Timestamp(train_end_date).day))
   
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(True)
            ax[i].spines['left'].set_visible(False)

    
        if save:
            full_fp = os.path.join(save_fp, 'r_t_plots_new.svg')
            fig.savefig(full_fp, format='svg',  dpi=1200)
        else:
            plt.show()