# Semi Mechanistic Models for COVID 19 Case Predictions in Singapore
Covid Modelling INP work.

# Model
Description of model here: https://github.com/titzehong/covid_modelling/blob/main/docs/COVID.pdf

# Get Data
The data comes from 2 sources, a hand processed csv of Non Pharmaceutical Interventions inside ... as well as a gitlab repo from Eugene.

To update data
```bash
git clone ssh://git@gitly.hopto.org:80/eugene/covid19-modelling-sg.git ./
```

## Data Sources
Files from gitlab contains the daily community cases, dormitory cases and imported cases, these serve as inputs to the model.
Files from NPIs give the current amount of NPIs the government has enacted 

## Scenario Based NPIs (Sept 2021)
Found [here](https://github.com/titzehong/covid_modelling/tree/main/data/Scenario%20NPI), there are 3 scenarios for the month of september:

A. Social gather at workplace: https://www.straitstimes.com/singapore/no-social-gatherings-at-spore-workplaces-from-sept-8-tougher-action-if-covid-19-cases

B. School lockdown: https://www.straitstimes.com/singapore/parenting-education/full-hbl-for-primary-special-education-schools-from-sept-27-for-10

1. No A and no B (i.e., no change in measures from 8th)
2. A but no B (i.e., change on 8th, but no change from 27th)
3. A and B (i.e., change on 8th and change on 27th)

# Set Up Environment
```bash
conda env create -f sm_env.yml
```

# Running Model 
To run model 
```bash
conda activate pymc3_env_mlflow
python sm_run.py END_DATE --separate_dorms --separate_quarantine --adjust_vax -t PREDICTION_STEPS -c NUM_CHAINS -l TUNE_STEPS -p TRAIN_STEPS -k P_ACCEPT -y 5 -f 'EXPERIMENT NAME' 
```
This automatically creates a folder Results in the parent directory and stores the training trace as well as generated plots

To run multiple runs in parallel:
```bash
python sm_run.py END_DATE --separate_dorms --separate_quarantine --adjust_vax -t PREDICTION_STEPS -c NUM_CHAINS -l TUNE_STEPS -p TRAIN_STEPS -k P_ACCEPT -y 5 -f 'EXPERIMENT NAME' &
sleep 3s
python sm_run.py END_DATE --separate_dorms --separate_quarantine --adjust_vax -t PREDICTION_STEPS -c NUM_CHAINS -l TUNE_STEPS -p TRAIN_STEPS -k P_ACCEPT -y 5 -f 'EXPERIMENT NAME' &
sleep 3s
python sm_run.py END_DATE --separate_dorms --separate_quarantine --adjust_vax -t PREDICTION_STEPS -c NUM_CHAINS -l TUNE_STEPS -p TRAIN_STEPS -k P_ACCEPT -y 5 -f 'EXPERIMENT NAME' &
```
Sleep is needed between invocations due to mlflow not handling things properly.

# View Tracked Results in MLflow
Running sm_run.py automatically logs training stuff to an MLflow directory, a folder mlflow will be created in the current directory storing training info.
```bash
mlflow ui
```
If running from a remote server (eg GCP). Forward it via ssh.
In remote terminal:
```bash
mlflow ui
```
On local terminal:
```bash
ssh -i ssh_key_file -L 5000:localhost:LOCAL_PORT <REMOTE_USER>@<REMOTE_HOST>
``` 
Then navigate to http://127.0.0.1:5000 on a web browser.



# To use model as a library

First load the data
```python
# Load data
from sm_data import generate_prediction_data, generate_NPI_prediction_data
from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
from sm_utils import *
from sm_utils import _get_convolution_ready_gt
from sm_model import SemiMechanisticModels

data_path = os.path.join('../covid19-modelling-sg/data/statistics', 'epidemic_curve.csv')
NPI_PATH1 = 'data/NPIS_LC_processed_V2.csv'
NPI_PATH2 = 'data/lockdown_index.csv'

# Load data
community_input, dorm_input, imported_input, total_input, len_obs = generate_prediction_data(data_path, start_id=50,
                             end_date=495, prediction_t=7, imported_case_extra='last',
                             separate_dorms=True)

# Load NPI Data
NPIS_array, date_ver = generate_NPI_prediction_data(NPI_PATH2, start_id=50, end_date=495, prediction_t=7)
print("NPIS Shape: ", NPIS_array.shape)
```

Instantiate the Model
```python 
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
              
```
Sample from the model. Note this takes up to 13hrs for 500 days of training data.
```python
sm_forecast.sample(init_choice='adapt_diag')
```
