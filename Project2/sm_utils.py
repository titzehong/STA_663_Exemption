from scipy import stats as sps
from theano.tensor.signal.conv import conv2d
from theano.ifelse import ifelse
from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
import theano
import theano.tensor as tt
import numpy as np 
import pandas as pd 
import pickle
import pymc3 as pm


def _get_generation_time_interval(variant='original'):
    """ Create a discrete P(Generation Interval)
        Source: https://www.ijidonline.com/article/S1201-9712(20)30119-3/pdf """
    
    if variant == 'original':
        # Original/Alpha
        mean_si = 4.7
        std_si = 2.9
    
    elif variant == 'delta':
        #https://wwwnc.cdc.gov/eid/article/28/2/21-1774_article
        # Delta Variant
        mean_si = 3.7
        std_si = 4.8
    
    elif variant == 'omicron':
        # Omicron Variant 
        # https://www.medrxiv.org/content/10.1101/2021.12.25.21268301v1.full.pdf
        mean_si = 2.22
        std_si = 1.62

    elif variant == 'omicron2':
        #https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2022.27.6.2200042
        mean_si = 3.5
        std_si = 2.9
    else:
        raise ValueError("variant must be original, delta or omicron")
    
    mu_si = np.log(mean_si ** 2 / np.sqrt(std_si ** 2 + mean_si ** 2))
    sigma_si = np.sqrt(np.log(std_si ** 2 / mean_si ** 2 + 1))
    dist = sps.lognorm(scale=np.exp(mu_si), s=sigma_si)

    # Discretize the Generation Interval up to 20 days max
    g_range = np.arange(0, 20)
    gt = pd.Series(dist.cdf(g_range), index=g_range)
    gt = gt.diff().fillna(0)
    gt /= gt.sum()
    gt = gt.values
    return gt


def _get_convolution_ready_gt_multi_variant(len_observed, switch_idx, v1, v2):
    ''' 
     Speeds up theano.scan by pre-computing the generation time interval
        vector. Th"ank you to Junpeng Lao for this optimization.
        Please see the outbreak simulation math here:
        https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html
        
    
        This is a modified version of _get_convolution_ready_gt() that blends both the serial interval from two variants
        together
        
    '''
    gt1 = _get_generation_time_interval(v1)
    gt2 = _get_generation_time_interval(v2)
    
    generation_days_considered = len(gt1)
    
    convolution_ready_gt = np.zeros((len_observed - 1, len_observed))
    for t in range(1, len_observed):
        
        begin = np.maximum(0, t - len(gt) + 1)  # curent time - generation_time_considered + 1 
                                                #(Basically no of days needed) no that can be above max!

        if t < switch_idx:
            gt = gt1

        elif t <= switch_idx + generation_days_considered: 

            ### Creates a composite SI vector 
            
            # no of days to replace from gt1
            switch_days = t - switch_idx

            # Take first part of gt1
            gt1_required = gt1[0 : t - begin + 1]  # Take as much v1 needed

            # Remaining new days take from gt2
            gt1_required[0: switch_days] = gt2[0: switch_days]


        else: # t>switch_idx + generation_days_considered (means variant 1 completely phased out)
            gt = gt2
        
    convolution_ready_gt = theano.shared(convolution_ready_gt)
    return convolution_ready_gt


def create_mixture_si(v_proportions, si_dists):
    """
    P(X=x) = v1_pct * v1_dist + (1-v1_pct)*v2_dist
    """
    mix_dist = np.sum([p*k for p,k in zip(v_proportions, si_dists)],0)
    return mix_dist



def _get_convolution_ready_gt_mixture_variants(len_observed, len_gt, si_matrix):
    
    # Initialize convolution matrix
    convolution_ready_gt = np.zeros((len_observed - 1, len_observed))

    # Iterate through columns
    for t in range(0, len_observed):     
        si_mix_t = si_matrix[t, :]

        if t+len_gt-1 <= len_observed - 1:  # Enough space to fit all elements of si_mix_t[1::]
            end_len = t+len_gt-1  
        else:
            remaining_space = len_observed - t - 1  # Need to squeeze
            end_len = t+remaining_space

        needed_elements = end_len-t

        convolution_ready_gt[t:end_len, t] = si_mix_t[1:needed_elements+1]

    convolution_ready_gt = theano.shared(convolution_ready_gt)
    return convolution_ready_gt


def create_mixture_si_matrix(vir_proportions, len_observed, si_dists):
    
    """
    vir_proportions - len_observed x no_variants matrix of proprtions, each row should sum to 1
    len_observed - number of days considered
    si_dists - List[np.array] where ith element is SI of variant i (must correspond to column in vir_proportions matrix)
    """
    
    
    assert(vir_proportions.shape[0] == len_observed)
    
    len_gt = len(si_dists[0]) # How many days to consider for serial interval (must be fixed)
    si_mix_matrix = np.zeros([len_observed, len_gt])

    for i,v in enumerate(vir_proportions):
        mix_dist = create_mixture_si(v, si_dists)
        si_mix_matrix[i, :] = mix_dist
    
    return si_mix_matrix


def _get_convolution_ready_gt(len_observed):
    """ Speeds up theano.scan by pre-computing the generation time interval
        vector. Thank you to Junpeng Lao for this optimization.
        Please see the outbreak simulation math here:
        https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html """
    gt = _get_generation_time_interval()
    convolution_ready_gt = np.zeros((len_observed - 1, len_observed))
    for t in range(1, len_observed):
        begin = np.maximum(0, t - len(gt) + 1)
        slice_update = gt[1 : t - begin + 1][::-1]
        convolution_ready_gt[
            t - 1, begin : begin + len(slice_update)
        ] = slice_update
    convolution_ready_gt = theano.shared(convolution_ready_gt)
    return convolution_ready_gt

def conv(infections, p_delay, len_p_delay, len_observed):
    "1D Convolution of a and b"
    return conv2d(
            tt.reshape(infections, (1, len_observed)),
            tt.reshape(p_delay, (1, len_p_delay)),
            border_mode='full',
            )[0, :len_observed]

def get_p_delay():
    delays = get_delays_from_patient_data(file_path='data/latestdata.csv')
    INCUBATION_DAYS = 3.5

    p_delay = delays.value_counts().sort_index()
    new_range = np.arange(0, p_delay.index.max() + 1)
    p_delay = p_delay.reindex(new_range, fill_value=0)
    p_delay /= p_delay.sum()
    p_delay = (
        pd.Series(0.0001*np.ones(INCUBATION_DAYS))
        .append(p_delay, ignore_index=True)
        .rename("p_delay")
    )
    return p_delay


def get_custom_p_delay(fp, incubation_days=5):
    
    # Loads dictionary
    delay_info_raw = pickle.load(open(fp, 'rb'))
    
    # Remove -1 
    delay_info_cleaned = dict([x for x in delay_info_raw.items() if x[0]!=-1])
    
    # Find sum and return normalized counts
    total_cases = sum(delay_info_cleaned.values())
    p_delay = pd.Series((1/total_cases)*np.array(list(delay_info_cleaned.values())))
    
    # Append incubation days to the front
    p_delay = (
        pd.Series(0.0001*np.ones(incubation_days))
        .append(p_delay, ignore_index=True)
        .rename("p_delay")
    )
    
    return p_delay

def one_step_sus_adjustment_local(ihat_quarantined_t,
                                  ihat_un_quarantined_t,
                                  e_vax_t,
                                  st_1,
                                  i_unq_t,
                                  i_q_t,
                                  total_pop):
    """
    Theano Scan Function

    ihat_quarantined_t -> Predicted no. linked infected by the model
    ihat_unquarantined_t -> Predicted no. unlinked infected by the model
    e_vax_t -> The effective vaccination rate on that day
    st_1 -> Previous value of the susceptibles
    i_unq_t -> Previous adjusted unquarantined counts (Not used but need to put it for theano)
    i_q_t -> Previous adjusted quarantined counts (Not used but need for model to work)
    total_pop -> Fix total population size
    """
    
    
    # Quarantined
    i_t_q = st_1 * (1-pm.math.exp(-1 * ihat_quarantined_t/total_pop))
    
    # Unquarantined
    i_t_uq = st_1 * (1-pm.math.exp(-1 * ihat_un_quarantined_t/total_pop))
    
    s_t = (st_1 - i_t_q - i_t_uq)*(1-e_vax_t)
    #theano.set_subtensor(zeros_subtensor, a_value)
    
    return [s_t, i_t_q, i_t_uq]


def one_step_sus_adjustment_dorm(ihat_dorm_t,
                                  e_vax_t,
                                  st_1,
                                  i_d_t,
                                  total_pop):
    """
    Theano Scan Function

    ihat_dorm_t -> Predicted no. dorm infected by the model
    e_vax_t -> The effective vaccination rate on that day
    st_1 -> Previous value of the susceptibles
    i_d_t -> Previous adjusted Dorm counts (Not used but need to put it for theano)
    total_pop -> Fix total population size
    """
    
    # Quarantined
    i_t_d = st_1 * (1-pm.math.exp(-1 * ihat_dorm_t/total_pop))
    

    s_t = (st_1 - i_t_d)*(1-e_vax_t)
    #theano.set_subtensor(zeros_subtensor, a_value)
    
    return [s_t, i_t_d]


def dorm_calc_infected(t, gt, e_vax_t, y, sus_series, r_t_foreign, beta, dorm_pop):
    """ To use in theano scan to generate dorm infected + susceptibles """


    # Calculate the addition to effective R(T) - unadjusted 
    exp_sus_rate = tt.exp(beta*sus_series/dorm_pop)
    
    # Calculate the implied number of infected
    inf_t_dorm = tt.sum(exp_sus_rate*r_t_foreign*y*gt)
    
    #mu_print = tt.printing.Print("inf_t_dorm")(inf_t_dorm)

    # Add people from 70 days ago
    """
    if tt.ge(t,70):
        re_susceptible = y[t-69] 
    else:
        re_susceptible = tt.cast(0, 'float64')
    """
    
    #re_susceptible_dorm = ifelse(tt.ge(t, 70),  tt.cast(y[t-69], 'float64') , tt.cast(0, 'float64'))
    #mu_print = tt.printing.Print("re_susceptible_dorm")(re_susceptible_dorm)
    remain_sus = (sus_series[t-1] - inf_t_dorm)*(1-e_vax_t)
    
    return [tt.set_subtensor(y[t], inf_t_dorm), tt.set_subtensor(sus_series[t], remain_sus)] 



def community_calc_infected(t, gt, i_cases, l_t, e_vax_t, y, sus_series, r_t_local, beta, community_pop,eps_t):
    """ To use in theano scan to generate community infected + susceptibles """

    exp_sus_rate = tt.exp(beta*sus_series/community_pop)  # Proportion of susceptible people
    inf_t_comm_unlinked = l_t*(tt.sum(exp_sus_rate*r_t_local*y*gt) + eps_t*i_cases)
    inf_t_comm_linked = ((1-l_t)/l_t)*inf_t_comm_unlinked
    
    # Add people from 70 days ago

    """
    if tt.ge(t,70):
        re_susceptible = y[t-69] 
    else:
        re_susceptible = tt.cast(0, 'float64')
    """
    re_susceptible = ifelse(tt.ge(t, 70), y[t-69] , tt.cast(0, 'float64'))

    remain_sus = (sus_series[t-1] - inf_t_comm_unlinked - inf_t_comm_linked + re_susceptible)*(1-e_vax_t)
    

    return tt.set_subtensor(y[t], inf_t_comm_unlinked), tt.set_subtensor(sus_series[t], remain_sus) 


def sample_multi_poisson(lambda_array):
    """ Function to sample poisson rvs from a 1d array of poisson means """
    #np.random.seed(991)
    return np.array([np.random.poisson(x) for x in lambda_array])


def sample_poisson_trace(lambda_samples):
    """ Function to sample poisson from a 2d array of poisson means """
    #np.random.seed(991)
    return np.array([sample_multi_poisson(x) for x in lambda_samples])


def sample_multi_nb(lambda_array, disp_phi):
    """ Function to sample Negative binomial from 1d array of neg bin mean and dispersion """
    implied_alpha = lambda_array/disp_phi

    implied_n = implied_alpha
    implied_p = implied_n / (implied_n + lambda_array)
    return [np.random.negative_binomial(n_t, p_t) for (n_t, p_t) in zip(implied_n, implied_p)]


def sample_nb_trace(lambda_samples, disp_phi_samples):
    """ Function to sample Negative binomial from 2d array of neg bin mean and dispersion """
    
    return np.array([sample_multi_nb(lamb_samp, phi_samp) for lamb_samp,
                     phi_samp in zip(lambda_samples,disp_phi_samples)])


def merge_frames(dataframe_list, colname):
    """ Merges a list of dataframes together on common attribute """
    output_frame = dataframe_list[0]

    for df in dataframe_list[1:]:
        output_frame = output_frame.merge(df, on=colname)
    
    return output_frame


