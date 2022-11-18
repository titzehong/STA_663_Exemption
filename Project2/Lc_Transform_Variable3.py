#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from sm_data import generate_prediction_data, generate_NPI_prediction_data
from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
from sm_utils import *
from sm_utils import _get_convolution_ready_gt
from sm_model import SemiMechanisticModels


# In[2]:


data_path = os.path.join('../covid19-modelling-sg/data/statistics', 'epidemic_curve.csv')
NPI_PATH1 = 'data/NPIS_LC_processed_V3.csv'
NPI_PATH2 = 'data/lockdown_index.csv'


# In[3]:


community_input, dorm_input, imported_input, total_input, len_obs = generate_prediction_data(data_path, start_id=50,
                             end_date=495, prediction_t=7, imported_case_extra='last',
                             separate_dorms=True)


# In[4]:


NPIS_array, date_ver = generate_NPI_prediction_data(NPI_PATH1, start_id=50, end_date=495, prediction_t=7)
print("NPIS Shape: ", NPIS_array.shape)


# # Model running 

# In[5]:


sm_forecast = SemiMechanisticModels()
# Instantiate model based on data
sm_forecast.build_model(local_cases=community_input,
                          dorm_cases=dorm_input,
                          imported_cases=imported_input,
                          NPIS_array=NPIS_array,
                          len_observed=len_obs,
                          total_cases=None,
                          separate_dorms=True,
                          likelihood_fun='PO')


# In[ ]:


sm_forecast.sample(init_choice='adapt_diag')
#sm_forecast.sample(chains=2, tune=1500, draws=3000)


# In[ ]:


sm_forecast._trace.varnames


# In[ ]:


az.plot_trace(sm_forecast.traces,var_names=['b_'+str(x) for x in range(10)])
plt.plot()


# In[ ]:




