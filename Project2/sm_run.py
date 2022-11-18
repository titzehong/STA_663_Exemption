from turtle import Turtle
import pandas as pd
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az

# Dumb matplotlib hack
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

from sm_data import generate_prediction_data, generate_NPI_prediction_data,\
     generate_ground_truth_forecast, get_isolation_data_model, \
              get_test_positivity_data_model,get_orders_issued_model, \
                  get_train_prediction_dates, get_ascertainment_regime, get_variant_pct_info,\
                      add_art_data

from covid.patients import get_delay_distribution, get_delays_from_patient_data, download_patient_data
from sm_utils import *
from sm_utils import _get_convolution_ready_gt, _get_generation_time_interval
from sm_model import SemiMechanisticModels
import mlflow
import time


# Example call
"""
python sm_run.py 276 --separate_quarantine --advi --adjust_vax --latent_mean -t 50 -c 2 -l 10 -p 10 -k 0.99 -n "./data/latents_mean_28feb22.csv" -v 8 -f 'Experimment Title'
"""

if __name__ == '__main__':

    start_timer = time.time()

    parser = argparse.ArgumentParser(description='To add end_date and prediction_t')
    parser.add_argument('end_date', type=int, action='store',
                        help='Date where training period ends')

    parser.add_argument('--val_run', dest='val_run', action='store_true')
    parser.add_argument('--separate_dorms', dest='separate_dorms', action='store_true')
    parser.add_argument('--separate_quarantine', dest='separate_quarantine', action='store_true')
    parser.add_argument('--adjust_vax', dest='adjust_vax', action='store_true')
    parser.add_argument('--advi', dest='use_advi', action='store_true')
    parser.add_argument('--PoLL', dest='use_poisson_ll', action='store_true')
    parser.add_argument('--latent_mean', dest='use_latent', action='store_true')

    parser.add_argument('-t', '--step_ahead', type=int, action='store',
                        help='Prediction steps ahead', default=7)

    parser.add_argument('-c', '--num_chains', type=int, action='store',
                        help='Prediction steps ahead', default=2)
    parser.add_argument('-l', '--tune_steps', type=int, action='store',
                        help='Prediction steps ahead', default=1500)
    parser.add_argument('-p', '--train_steps', type=int, action='store',
                        help='Prediction steps ahead', default=3000)
    parser.add_argument('-k', '--t_accept', type=float, action='store',
                        help='Num Cores', default=2)
    parser.add_argument('-y', '--scenario', type=int, action='store',
                        help='NPI Scenario', default=0)
    parser.add_argument('-v', '--vax_regime', type=int, action='store',
                        help='Effective Vax Rate Series to use', default=0)
    parser.add_argument('-f', '--exp_name', type=str, action='store',
                    help='mlflow experiment name', default='Default')
    parser.add_argument('-n', '--NPI_Path_input', type=str, action='store',
                    help='NPI Path', default='Default')


    args = parser.parse_args()


    end_date = args.end_date
    prediction_t = args.step_ahead

    val_run = args.val_run
    separate_dorms = args.separate_dorms
    separate_quarantine = args.separate_quarantine
    adjust_vax_flag = args.adjust_vax
    use_advi = args.use_advi
    use_po_ll = args.use_poisson_ll
    use_latent = args.use_latent

    n_chains = args.num_chains
    tune_steps = args.tune_steps
    train_steps = args.train_steps
    t_accept = args.t_accept
    exp_name = args.exp_name
    npi_scenario = args.scenario
    npi_path_input = args.NPI_Path_input

    vax_rate_choice = args.vax_regime
    # 0 for pessimistic
    # 1 for optimistic

    use_mixture_variants = True

    mlflow.set_experiment(experiment_name=exp_name)

    with mlflow.start_run():

        run_id = mlflow.active_run().info.run_id

        save_fp = f"../Results/SM_V2_Trace_Start_{end_date}_prediction_ahead_{n_chains}_{tune_steps}_{train_steps}_id_{run_id}"

        print("End Date: ", end_date)
        print("Prediction_t: ", prediction_t)
        if val_run:
            print("Valiation Run")

        print('\n')
        print("TRAINING CONFIG: ")
        print("No Chains: ", n_chains)
        print("No Tuning: ", tune_steps)
        print("No MCMC Steps: ", train_steps)

        print('\n')
        print("FP: ", save_fp)

        DATA_PATH = os.path.join('../covid19-modelling-sg/data/statistics', 'epidemic_curve.csv')


        if npi_path_input:
            NPI_PATH = npi_path_input
        else:           
            if npi_scenario == 0:
                NPI_PATH = 'data/Scenario NPI/scenarios_new/NPIS_phased_school_lifted_22Nov.csv'
                print("CASE 0")
            elif npi_scenario == 1:
                NPI_PATH = 'data/Scenario NPI/scenarios_new/NPIS_phased_school_lifted_25Oct.csv'
                print("CASE 1")
            elif npi_scenario == 2:
                NPI_PATH = 'data/Scenario NPI/scenarios_new/NPIS_phased_school_not_lifted_22Nov.csv'
                print("CASE 2")
            elif npi_scenario == 3:
                NPI_PATH = 'data/Scenario NPI/scenarios_new/NPIS_phased_school_not_lifted_25Oct.csv'
                print("CASE 3")
            elif npi_scenario == 4:
                NPI_PATH = 'data/Scenario NPI/NPIS_case1_25Oct.csv'
                print("CASE 1 - New (25 Oct)")
            elif npi_scenario == 5:
                NPI_PATH = 'data/Scenario NPI/NPIS_case2_25Oct.csv'
                print("CASE 2 - New (25 Oct)")
            elif npi_scenario == 6:
                NPI_PATH = 'data/Scenario NPI/NPIS_case3_11Oct_updated.csv'
                print("CASE 3 - New NPIs till 31st Dec")

            elif npi_scenario == 7:  # LC NPI Lifted
                NPI_PATH = 'data/Latent NPIs/sat_sun/lifted/125717_latents_mean.csv'
                use_latent=True
                print("Latent NPI Mean")
            elif npi_scenario == 8:  # LC NPI not lifted
                NPI_PATH = 'data/Latent NPIs/sat_sun/not_lifted/133257_latents_mean.csv'
                use_latent=True
                print("Latent NPI mean")

            else:
                raise ValueError('invalid value entered for scenario')

        #QUARANTINE_DATA_PATH =  '../covid19-modelling-sg/data/statistics/active_number_under_quarantine.csv'
        #SHN_DATA_PATH = '../covid19-modelling-sg/data/statistics/individuals_under_shn.csv'

        QUARANTINE_ISSUED_DATA_PATH =  '../covid19-modelling-sg/data/statistics/daily_quarantine_orders_issued.csv'
        SHN_ISSUED_DATA_PATH = '../covid19-modelling-sg/data/statistics/shn_issued_by_press_release_date.csv'


        TESTS_COUNTS_OWID_PATH = '../Data/owid-covid-data.csv'
        TEST_COUNTS_REPO_PATH = '../covid19-modelling-sg/data/statistics/swab_figures.csv'
        TEST_COUNTS_DATA_GOV_PATH = '../covid19-modelling-sg/data/statistics/average_daily_swabs_data_gov_sg.csv'
        CASE_BREAKDOWN_PATH = os.path.join('../covid19-modelling-sg/data/statistics', 'epidemic_split_curve_w_art.csv')
        ASCERTAINMENT_RATE_REGIME_PATH = 'data/Testing Regime.csv'
        ASCERTAINMENT_RATE_REGIME_2_PATH = 'data/Testing Regime ART.csv'
        VARIANT_PATH =  'data/variant_pct_v1_filled.csv'

        # Import fixed Vax effective rate:
        if adjust_vax_flag:

            if vax_rate_choice == 0: # Pessimistic
                print('Vax Path: ','VE_rate_model_30th_dec_pessimistic')
                vax_effective_rate = pd.read_csv('data/VE_rate_model_30th_dec_pessimistic.csv')
            elif vax_rate_choice == 1: # ==1 optimistic
                print('Vax Path: ','VE_rate_model_30th_dec_optimistic')
                vax_effective_rate = pd.read_csv('data/VE_rate_model_30th_dec_optimistic.csv')

            elif vax_rate_choice == 2: # no booster
                print('Vax Path: ','VE_rate_model_30th_dec_pessimistic_no_booster_updated')
                vax_effective_rate = pd.read_csv('data/VE_rate_model_30th_dec_pessimistic_no_booster_updated.csv')
        
            elif vax_rate_choice == 3: # waning booster
                print("Vax Path: ", 'VE_rate_model_30th_dec_pessimistic_updated')
                vax_effective_rate = pd.read_csv('data/VE_rate_model_30th_dec_pessimistic_updated.csv')
           
            elif vax_rate_choice == 4: # waning booster
                print("Vax Path: ", 'VE_rate_model_28th_Oct')
                vax_effective_rate = pd.read_csv('data/VE_rate_model_28th_Oct.csv')
                vax_dates = pd.to_datetime(vax_effective_rate['Date'])

            elif vax_rate_choice == 5: # Most updated
                print("Vax Path: ", 'VE_rate_model_11th_nov')
                vax_effective_rate = pd.read_csv('data/VE_rate_model_11th_nov.csv')
                vax_dates = pd.to_datetime(vax_effective_rate['Date'])

            elif vax_rate_choice == 6: # Updated 4th jan 2022
                print("Vax path: ", 'veffectivevax_4jan2022.csv')
                vax_effective_rate = pd.read_csv('data/veffectivevax_4jan2022.csv')
                vax_dates = pd.to_datetime(vax_effective_rate['Date'])

            elif vax_rate_choice == 7: # Updated 14 Feb 2022
                print("Vax path: ", 'vacc-owid-omicron-update.csv')
                vax_effective_rate = pd.read_csv('data/vacc-owid-omicron-update.csv')
                vax_dates = pd.to_datetime(vax_effective_rate['Date'])

            elif vax_rate_choice == 8: # Updated 3 march 2022
                print("Vax path: ", 'vacc-owid-28feb22.csv')
                vax_effective_rate = pd.read_csv('data/vacc-owid-28feb22.csv')
                vax_dates = pd.to_datetime(vax_effective_rate['Date'])

            if (vax_rate_choice not in [4,5,6,7,8]):
                vax_effective_rate = vax_effective_rate['Effective Vax t'].values
                vax_effective_rate = np.concatenate([0.00001*np.ones(10), vax_effective_rate])
            
            elif (vax_rate_choice == 7) or (vax_rate_choice == 8):
                vax_effective_rate = vax_effective_rate['dRate'].values
                vax_effective_rate = vax_effective_rate[17:]  # Should start from 1st Jan 2021

            else:  # 4,5 and 6 start from 4th
                vax_effective_rate = vax_effective_rate['Effective Vax t'].values
                vax_effective_rate = np.concatenate([0.00001*np.ones(3), vax_effective_rate])

            
        
            #community_popu =  (5400000-300000)*0.9 + 0.1 # 0.9 natural immune, add 0.1 to avoid numerical problem
            community_popu = 4096607
            dorm_popu = 323000*0.9 + 0.1 # https://www.straitstimes.com/singapore/47-per-cent-of-migrant-workers-in-dorms-have-had-a-covid-19-infection-say-manpower-and
        
        else:
            vax_effective_rate=None
            community_popu = None
            dorm_popu = None


        #start_id = 50  # Note: Dont commit this change (only temp)
        start_id = 494


        if use_po_ll:
            ll_form = 'PO'
        else:
            ll_form = 'NB'
        print("Likelihood Form: ", ll_form)


        # Log everything
        mlflow.log_param('end_date', end_date)
        mlflow.log_param('prediction_t', prediction_t)
        mlflow.log_param('n_chains', n_chains)
        mlflow.log_param('t_accept', t_accept)
        mlflow.log_param('tune_steps', tune_steps)
        mlflow.log_param('train_steps', train_steps)
        mlflow.log_param('DATA_PATH', DATA_PATH)
        mlflow.log_param('NPI_PATH', NPI_PATH)
        mlflow.log_param('save_path', save_fp)
        mlflow.log_param('NPI Scenario', npi_scenario)
        mlflow.log_param('Adjusting Vax', adjust_vax_flag)
        mlflow.log_param('Vax rate Choice', vax_rate_choice)
        mlflow.log_param('Initial_community_size', community_popu)
        mlflow.log_param('Initial_dorm_popu', dorm_popu)
        mlflow.log_param('ADVI', use_advi)
        mlflow.log_param("LL_Form", ll_form)

        mlflow.log_artifact('./sm_run.py')
        mlflow.log_artifact('./sm_model.py')
        mlflow.log_artifact('./sm_model.py')
        mlflow.log_artifact('./sm_utils.py')

        """
        use_data_gov = False
        if use_data_gov:  # Flag to accomodate new data format from data.gov
            print("Adding ART Test Numbers in")
            art_data_path = '../covid19-modelling-sg/data/statistics/covid19_case_numbers_data_gov_sg.csv'
            new_CASE_BREAKDOWN_PATH = './new_epi_split_curve_data.csv'

            add_art_data(CASE_BREAKDOWN_PATH, art_data_path, new_CASE_BREAKDOWN_PATH)
            CASE_BREAKDOWN_PATH = new_CASE_BREAKDOWN_PATH
            print("New case breakdown path: ", CASE_BREAKDOWN_PATH)
        """

        # IMPORT DATA
        community_input, dorm_input, imported_input, total_input, len_observed = generate_prediction_data(DATA_PATH, start_id=start_id,
                                    end_date=end_date, prediction_t=prediction_t, imported_case_extra='ma',
                                    separate_dorms=separate_dorms, separate_quarantine=separate_quarantine,
                                    local_cases_breakdown_data_path=CASE_BREAKDOWN_PATH)

        # Get dates
        train_dates, all_dates = get_train_prediction_dates(DATA_PATH,start_id,end_date,prediction_t)

        print("Train Start Date: ", train_dates[0])
        print("Train End Date: ", train_dates[-1])
        print("Forecast Start Date: ", all_dates[-1*prediction_t])
        print("Forecast End Date: ", all_dates[-1])



        if vax_effective_rate is not None:
            print('Making VE Length Correct')
            vax_effective_rate = vax_effective_rate[0:len_observed]  ## Only works till 31dec

        if val_run:
            com_gt, dorm_gt, import_gt, total_gt = generate_ground_truth_forecast(DATA_PATH, start_id=start_id,
                                end_date=end_date, prediction_t=prediction_t)


        if not use_latent:
            NPIS_array, date_ver = generate_NPI_prediction_data(NPI_PATH, start_id=start_id, end_date=end_date, prediction_t=prediction_t)

        else:
            print("Processing Latent NPIs")
            NPIS_array = pd.read_csv(NPI_PATH)
            NPIS_array = NPIS_array[start_id: start_id + len_observed].iloc[:,1:]

        num_NPIS = NPIS_array.shape[1]

        # Ascertainment regime
        a_rate_regime = get_ascertainment_regime(ASCERTAINMENT_RATE_REGIME_PATH, 
                                                start_id,
                                                end_date,
                                                prediction_t)

        a_rate_regime2 = get_ascertainment_regime(ASCERTAINMENT_RATE_REGIME_2_PATH, 
                                                start_id,
                                                end_date,
                                                prediction_t)
        
        #get_isolation_data_model, get_test_positivity_data_model

        if use_mixture_variants:
            variant_pct_series = get_variant_pct_info(VARIANT_PATH, start_id, end_date, prediction_t)
            variant_si_dists = [_get_generation_time_interval('original'),
                    _get_generation_time_interval('delta'),
                    _get_generation_time_interval('omicron2')]
        else:
            variant_pct_series = None
            variant_si_dists = None


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
                             qo_issued_start_val=0, scale=True,extrapolate_type='ma', shn_only=True)

            test_stats = get_test_positivity_data_model(TESTS_COUNTS_OWID_PATH, TEST_COUNTS_REPO_PATH,
                                  DATA_PATH,TEST_COUNTS_DATA_GOV_PATH, start_id,
                                  end_date, prediction_t=prediction_t, start_val=100, scale=True,
                                  extrapolate_type='ma')

        else:
            quarantine_stats = None
            test_stats = None

        # Print shapes
        print("NPI SHAPE: ", NPIS_array.shape)
        print("Quanratine SHAPE: ", len(quarantine_stats))
        print("Test Shape: ", test_stats.shape)
        print("Imported Shape: ", imported_input.shape)

        # TRAIN MODEL
        sm_forecast = SemiMechanisticModels()
        # Instatiate model based on data
        sm_forecast.build_model(local_cases=community_input,
                                dorm_cases=dorm_input,
                                imported_cases=imported_input,
                                NPIS_array=NPIS_array,
                                len_observed=len_observed,
                                total_cases=None,
                                separate_dorms=separate_dorms,
                                separate_quarantine=separate_quarantine,
                                test_counts=test_stats,
                                quarantine_counts=quarantine_stats,
                                prediction_t=prediction_t,
                                likelihood_fun=ll_form,
                                vax_adjustment=adjust_vax_flag,
                                effective_vax_rate=vax_effective_rate,
                                community_pop=community_popu,
                                dorm_pop=dorm_popu,
                                a_rate_regime=a_rate_regime,
                                a_rate_regime2=a_rate_regime2,
                                use_latent=use_latent,
                                variant_pct_series=variant_pct_series,
                                variant_si_dists=variant_si_dists)

        if use_advi:
            init_choice = 'advi+adapt_diag'
        else:
            init_choice = 'auto'
        print("Init Choice: ", init_choice)

        sm_forecast.sample(chains=n_chains,
                            tune=tune_steps,
                            draws=train_steps,
                            t_accept=t_accept,
                            init_choice=init_choice)

        # Log number of divergence
        divergent = sm_forecast._trace["diverging"]
        print("Number of Divergent %d" % divergent.nonzero()[0].size)
        mlflow.log_metric('Divergent Samples', divergent.nonzero()[0].size)

        # Log metric -> Train accuracy in sample, test accuracy if have
        os.mkdir(save_fp)
        os.mkdir(save_fp+'_Analysis')

        mlflow.log_artifacts(save_fp+'_Analysis')

        sm_forecast.save_sample(save_fp)
        

        # Generate and Save Model Predictions
        date_range = pd.DatetimeIndex(all_dates)

        print("Saving Counts")
        sm_forecast.save_infected(date_range, save_fp+'_Analysis')
        sm_forecast.save_community_cases(date_range, save_fp+'_Analysis')
        sm_forecast.save_rt_scores(date_range, save_fp+'_Analysis')
        # Only call this after calling the above 3
        sm_forecast.output_predictions_sumamrised(save_fp+'_Analysis')

        print("Plotting")
        sm_forecast.plot_latents_new(date_range, save_fp+'_Analysis', save=True)
        sm_forecast.plot_rt_q_new(date_range, save_fp+'_Analysis', save=True)
        mlflow.log_artifacts(save_fp+'_Analysis')

        #save_fp_preds_plot = os.path.join(save_fp+'_Analysis', 'predict_cases.svg')
        save_fp_latents = os.path.join(save_fp+'_Analysis', 'latents.svg')
        save_fp_rt = os.path.join(save_fp+'_Analysis', 'rt.svg')
        save_fp_results = os.path.join(save_fp+'_Analysis', 'preds.pkl')
        save_fp_energy = os.path.join(save_fp+'_Analysis', 'energy.svg')

        save_fp_alpha_t = os.path.join(save_fp+'_Analysis', 'alphas_t.svg')
        save_fp_Q_t = os.path.join(save_fp+'_Analysis', 'Q_rate.svg')

        # Get model stats + Plots
        if val_run:
            output_acc = sm_forecast.get_accuracy(com_gt, dorm_gt, save=True, save_fp=save_fp_results)
            sm_forecast.plot_predictions(save=True,save_fp=save_fp_preds_plot, actual_local_only=com_gt, actual_dorm_only=dorm_gt)
            sm_forecast.plot_latents(save=True,save_fp=save_fp_latents, actual_local_only=com_gt, actual_dorm_only=dorm_gt)
            sm_forecast.plot_rt(save=True, save_fp=save_fp_rt)
            sm_forecast.plot_energy(save=True, save_fp=save_fp_energy)

            mlflow.log_metric('Dorm Train RMSE', output_acc['dorm_train'][0])
            mlflow.log_metric('Dorm Train MAE', output_acc['dorm_train'][1])
            mlflow.log_metric('Local Train RMSE', output_acc['local_train'][0])
            mlflow.log_metric('Local Train MAE', output_acc['local_train'][1])

            mlflow.log_metric('Dorm Test RMSE', output_acc['dorm_test'][0])
            mlflow.log_metric('Dorm Test MAE', output_acc['dorm_test'][1])
            mlflow.log_metric('Local Test RMSE', output_acc['local_test'][0])
            mlflow.log_metric('Local Test MAE', output_acc['local_test'][1])

        else:
            #output_acc = sm_forecast.get_accuracy(save=True, save_fp=save_fp_results)
            #sm_forecast.plot_predictions(save=True,save_fp=save_fp_preds_plot)
            sm_forecast.plot_latents(save=True,save_fp=save_fp_latents)
            sm_forecast.plot_rt(save=True, save_fp=save_fp_rt)
            sm_forecast.plot_energy(save=True, save_fp=save_fp_energy)

            if separate_quarantine:
                sm_forecast.plot_alpha_t(save=True, save_fp=save_fp_alpha_t)
                sm_forecast.plot_quarantine_rate(save=True, save_fp=save_fp_Q_t)
                mlflow.log_artifact(save_fp_alpha_t)
                mlflow.log_artifact(save_fp_Q_t)

                """
                mlflow.log_metric('Dorm Train RMSE', output_acc['dorm_train'][0])
                mlflow.log_metric('Dorm Train MAE', output_acc['dorm_train'][1])
                mlflow.log_metric('Local UnQ Train RMSE', output_acc['local_unQ_train'][0])
                mlflow.log_metric('Local UnQ Train MAE', output_acc['local_unQ_train'][1])
                mlflow.log_metric('Local Q Train RMSE', output_acc['local_Q_train'][0])
                mlflow.log_metric('Local Q Train MAE', output_acc['local_Q_train'][1])
                """

            else:
                mlflow.log_metric('Dorm Train RMSE', output_acc['dorm_train'][0])
                mlflow.log_metric('Dorm Train MAE', output_acc['dorm_train'][1])
                mlflow.log_metric('Local Train RMSE', output_acc['local_train'][0])
                mlflow.log_metric('Local Train MAE', output_acc['local_train'][1])
        

        beta_vals = az.summary(sm_forecast._trace,['b_'+str(x) for x in range(20)])
        fig, ax = plt.subplots(20,1,figsize=(10,50))

        for i in range(20):
            
            beta_val = beta_vals['mean'][i]
            ci_low = beta_vals['hdi_3%'][i]
            ci_high = beta_vals['hdi_97%'][i]
            
            
            ax[i].plot(NPIS_array.iloc[:,i])
            ax[i].set_title(f'NPI - {i} | {beta_val} ({ci_low},{ci_high})', )

        fig.tight_layout()
        save_fp_betas = os.path.join(save_fp+'_Analysis', 'beta_npi.svg')
        save_fp_betas_vals = os.path.join(save_fp+'_Analysis', 'beta_npi_vals.csv')
        fig.savefig(save_fp_betas, format='svg',  dpi=1200)
        beta_vals.to_csv(save_fp_betas_vals)

        mlflow.log_artifact(save_fp_betas_vals)
        mlflow.log_artifact(save_fp_betas)
        #mlflow.log_artifact(save_fp_preds_plot)
        mlflow.log_artifact(save_fp_latents)
        mlflow.log_artifact(save_fp_rt)
        #mlflow.log_artifact(save_fp_results)
        mlflow.log_artifact(save_fp_energy)



        mlflow.log_artifact(DATA_PATH)
        mlflow.log_artifact(NPI_PATH)

        end_timer = time.time()
        mlflow.log_metric('Done', 1)
        mlflow.log_metric("Execution Time", end_timer-start_timer)




# Run index experiment
#python sm_run.py 115 --val_run -t 7 -c 1 -l 15 -p 20 -k 0.8 -f 'testing'
#python sm_run.py 115 --val_run -t 7 -c 2 -l 1500 -p 3500 -k 0.99 -f 'Validate lockdown index'
#python sm_run.py 122 --val_run -t 7 -c 2 -l 1500 -p 3500 -k 0.99 -f 'Validate lockdown index'
#python sm_run.py 129 --val_run -t 7 -c 2 -l 1500 -p 3500 -k 0.99 -f 'Validate lockdown index'
#python sm_run.py 136 --val_run -t 7 -c 2 -l 1500 -p 3500 -k 0.99 -f 'Validate lockdown index'
#python sm_run.py 143 --val_run -t 7 -c 2 -l 1500 -p 3500 -k 0.99 -f 'Validate lockdown index'
#python sm_run.py 150 --val_run -t 7 -c 2 -l 1500 -p 3500 -k 0.99 -f 'Validate lockdown index'


# Test new
#python sm_run.py 150 --separate_dorms --separate_quarantine -t 14 -c 2 -l 25 -p 50 -k 0.99 -f 'Unit Test Parallel'
