import pandas as pd 
import numpy as np 
import pymc3 as pm 
import theano
import theano.tensor as tt
import arviz as az
import matplotlib.pyplot as plt
import os 
import argparse

from sm_data import generate_prediction_data, generate_NPI_prediction_data
from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
from sm_utils import *
from sm_utils import _get_convolution_ready_gt


parser = argparse.ArgumentParser(description='To add end_date and prediction_t')
parser.add_argument('end_date', type=int, action='store',
                     help='Date where training period ends')
parser.add_argument('-t', '--step_ahead', type=int, action='store',
                     help='Prediction steps ahead', default=7)

parser.add_argument('-c', '--num_chains', type=int, action='store',
                     help='Prediction steps ahead', default=2)
parser.add_argument('-l', '--tune_steps', type=int, action='store',
                     help='Prediction steps ahead', default=1500)
parser.add_argument('-p', '--train_steps', type=int, action='store',
                     help='Prediction steps ahead', default=3000)
args = parser.parse_args()

end_date = args.end_date
prediction_t = args.step_ahead

n_chains = args.num_chains
tune_steps = args.tune_steps
train_steps = args.train_steps
save_fp = f"../Results/SM_Trace_Start_{end_date}_prediction_ahead_{n_chains}_{tune_steps}_{train_steps}"

print("End Date: ", end_date)
print("Prediction_t: ", prediction_t)

print('\n')
print("TRAINING CONFIG: ")
print("No Chains: ", n_chains)
print("No Tuning: ", tune_steps)
print("No MCMC Steps: ", train_steps)

print('\n')
print("FP: ", save_fp)

DATA_PATH = '../data/Covid-19 SG Clean.csv'
NPI_PATH = 'data/NPIS_LC_processed.csv'
start_id = 50
#end_date = 115
#prediction_t = 7

model_input, imported_input, len_observed = generate_prediction_data(DATA_PATH, start_id,
                             end_date=end_date, prediction_t=prediction_t, imported_case_extra='ma')

NPIS_array = generate_NPI_prediction_data(NPI_PATH, start_id, end_date, prediction_t)
num_NPIS = NPIS_array.shape[1]

# Get serial interval and delay dist
convolution_ready_gt = _get_convolution_ready_gt(len_observed)
p_delay = get_p_delay()
likelihood_fun = 'PO' # PO or ZINB

with pm.Model() as model_11:
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
    
    trace_11 = pm.sample(chains=n_chains, tune=tune_steps, draws=train_steps,nuts={'target_accept':0.99})#, target_accept=0.99)

#save_fp = f"../Results/SM_Trace_Start_{end_date}_prediction_ahead_{prediction_t}_{tune_steps}_{train_steps}"
os.mkdir(save_fp)

pm.save_trace(trace_11, directory=save_fp, overwrite=True) 