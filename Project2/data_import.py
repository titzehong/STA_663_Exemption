import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 

from sm_data import generate_prediction_data, generate_NPI_prediction_data,\
     generate_ground_truth_forecast, get_isolation_data_model, \
              get_test_positivity_data_model,get_orders_issued_model
from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
from sm_utils import *
import time



end_date = 483  # Increments in +7 for cros validation, 490, 497, 504, 511
prediction_t = 14

separate_dorms = True
separate_quarantine = True


print("End Date: ", end_date)
print("Prediction_t: ", prediction_t)

DATA_PATH = os.path.join('../covid19-modelling-sg/data/statistics', 'epidemic_curve.csv')
NPI_PATH = 'data/NPIS_LC_processed_V8.csv'

#QUARANTINE_DATA_PATH =  '../covid19-modelling-sg/data/statistics/active_number_under_quarantine.csv'
#SHN_DATA_PATH = '../covid19-modelling-sg/data/statistics/individuals_under_shn.csv'

QUARANTINE_ISSUED_DATA_PATH =  '../covid19-modelling-sg/data/statistics/daily_quarantine_orders_issued.csv'
SHN_ISSUED_DATA_PATH = '../covid19-modelling-sg/data/statistics/shn_issued_by_press_release_date.csv'


TESTS_COUNTS_OWID_PATH = '../Data/owid-covid-data.csv'
TEST_COUNTS_REPO_PATH = '../covid19-modelling-sg/data/statistics/swab_figures.csv'
CASE_BREAKDOWN_PATH = os.path.join('../covid19-modelling-sg/data/statistics', 'epidemic_split_curve.csv')

start_id = 50

# IMPORT CASE DATA
community_input, dorm_input, imported_input, total_input, len_observed = generate_prediction_data(DATA_PATH, start_id=start_id,
                            end_date=end_date, prediction_t=prediction_t, imported_case_extra='ma',
                            separate_dorms=separate_dorms, separate_quarantine=separate_quarantine,
                            local_cases_breakdown_data_path=CASE_BREAKDOWN_PATH)

# IMPORT NPI DATA
NPIS_array, date_ver = generate_NPI_prediction_data(NPI_PATH, start_id=start_id, end_date=end_date, prediction_t=prediction_t)

num_NPIS = NPIS_array.shape[1]

# IMPORT TEST RATE + ISOLATION ORDERS
if separate_quarantine:
    """
    quarantine_stats = get_isolation_data_model(SHN_DATA_PATH, QUARANTINE_DATA_PATH,
                        DATA_PATH, start_id, end_date,
                        shn_start_val=0, prediction_t=prediction_t,
                        quarantine_start_val=0, scale=True,extrapolate_type='ma')
    """
    quarantine_stats = get_orders_issued_model(SHN_ISSUED_DATA_PATH, QUARANTINE_ISSUED_DATA_PATH,
                        DATA_PATH, start_id, end_date,
                        shn_issued_start_val=0, prediction_t=prediction_t,
                        qo_issued_start_val=0, scale=True,extrapolate_type='ma')

    test_stats = get_test_positivity_data_model(TESTS_COUNTS_OWID_PATH, TEST_COUNTS_REPO_PATH,
                            DATA_PATH, start_id,
                            end_date, prediction_t=prediction_t, start_val=100, scale=True,
                            extrapolate_type='ma')
else:
    quarantine_stats = None
    test_stats = None
