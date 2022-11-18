import pandas as pd 
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler


def import_case_data_v1(data_path, start_id):
    """ Import all case data

    Args:
        data_path ([str]): file path to the data
        start_id ([int]): start_id of the data

    Returns:
        [local_cases]: array of local cases
        [import_cases]: array of imported cases
        [len_observed]: length of local cases
    """

    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
    data = data.sort_values('Date')

    new_cases = data['Daily Confirmed '].values
    len_observed = len(new_cases)

    local_cases = data['Daily Local transmission'].values[start_id:]
    imported_cases = data['Daily Imported'].values[start_id:]
    len_observed = len(local_cases)
    
    return local_cases, imported_cases, len_observed
    #print(len_observed)


def import_case_data(data_path, start_id, separate_dorms=True):
    """ Import case data

    Args:
        data_path (str): file path to data from git repo
        start_id (int): first date -> normally chosen to be 50 cumulative cases
        separate_dorms (bool, optional): Whether to separate community and dorm. Defaults to True.

    Returns:
        [community_cases]: array of community cases
        [dorm_cases]: array of dorm cases
        [total_cases]: array of sum of community and dorm cases
        [import_cases]: array of imported cases
        [len_observed]: length of local cases
    """
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Unnamed: 0':'Date'})
    data['total_cases'] = data['Community Total Cases'] + data['Dormitory Residents Total Cases']
    data['Date'] = pd.to_datetime(data['Date'])

    community_cases = data['Community Total Cases'].values[start_id:]
    dorm_cases = data['Dormitory Residents Total Cases'].values[start_id:]
    total_cases = data['total_cases'].values[start_id:]
    imported_cases = data['Imported Cases'].values[start_id:]
    dates = data['Date'].values[start_id:]

    len_observed = len(community_cases)

    if separate_dorms:
        return community_cases, dorm_cases, imported_cases, total_cases, len_observed, dates
    
    else:
        return total_cases, imported_cases, len_observed



def import_NPIS_v1(data_path, start_id):
    """ Gets the NPIs for the total period

    Args:
        data_path (str): filepath to the NPIs data storage
        start_id (int): start date

    Returns:
        NPIS_wanted: dataframe of NPIs
    """
    NPIS_proc = pd.read_csv(data_path)
    NPIS_wanted = NPIS_proc.iloc[start_id:, 2::]
    dates = pd.to_datetime(NPIS_proc['Date'])

    return NPIS_wanted

def import_NPIS(data_path, start_id):
    """ Gets the NPIs for the total period

    Args:
        data_path (str): filepath to the NPIs data storage
        start_id (int): start date

    Returns:
        NPIS_wanted: dataframe of NPIs
    """

    NPIS_proc = pd.read_csv(data_path)
    if 'Unnamed: 0' in NPIS_proc.columns:
        NPIS_proc = NPIS_proc.drop('Unnamed: 0', axis=1)
    NPIS_wanted = NPIS_proc.iloc[start_id:, 1::]
    dates = pd.to_datetime(NPIS_proc['Date'])

    return NPIS_wanted, dates



def generate_prediction_data_v1(cases_data_path, start_id,
                             end_date, prediction_t, imported_case_extra='last'):
    local_cases, imported_cases, _ = import_case_data_v1(cases_data_path, start_id)
    
    train_local_cases = local_cases[0:end_date]
    train_imported_cases = imported_cases[0:end_date]

    #actual = local_cases[end_date:end_date+prediction_t]

    train_pad = [np.nan for x in range(prediction_t)]

    if imported_case_extra == 'last':
        imported_pad = [train_imported_cases[-1] for x in range(prediction_t)]

    elif imported_case_extra == 'ma':
        print('imported cases ma')
        ma_week = np.round(np.mean(train_imported_cases[-7::]))
        imported_pad = [ma_week for x in range(prediction_t)]
    else:
        imported_pad = imported_case_extra

    model_input = pd.Series(np.concatenate([train_local_cases, train_pad]))
    imported_input = np.concatenate([train_imported_cases, imported_pad])

    print("Length training cases: ", len(train_local_cases))
    print("Length into model: ", len(model_input)," ",len(imported_input))

    len_observed = len(model_input)

    return model_input, imported_input, len_observed


def generate_prediction_data(cases_data_path, start_id,
                             end_date, prediction_t,
                             imported_case_extra='last',
                             separate_dorms=True,
                             separate_quarantine=False,
                             local_cases_breakdown_data_path=None):
    if separate_dorms and (not separate_quarantine):
        local_cases, dorm_cases, imported_cases, total_cases, _, dates = import_case_data(cases_data_path,
                                                          start_id,
                                                          separate_dorms=True)
        
        train_local_cases = local_cases[0:end_date]
        train_dorm_cases = dorm_cases[0:end_date]
        train_imported_cases = imported_cases[0:end_date]
        train_total_cases = total_cases[0:end_date]
        train_dates = dates[0:end_date]

        print("Train Start Date: ", train_dates[0])
        print("Train End Date: ", train_dates[-1])

        #actual = local_cases[end_date:end_date+prediction_t]

        train_pad = [np.nan for x in range(prediction_t)]

        if imported_case_extra == 'last':
            imported_pad = [train_imported_cases[-1] for x in range(prediction_t)]

        elif imported_case_extra == 'ma':
            print('imported cases ma')
            ma_week = np.round(np.mean(train_imported_cases[-7::]))
            imported_pad = [ma_week for x in range(prediction_t)]
        else:
            imported_pad = imported_case_extra

        local_model_input = pd.Series(np.concatenate([train_local_cases, train_pad]))
        dorm_model_input = pd.Series(np.concatenate([train_dorm_cases, train_pad]))
        imported_input = np.concatenate([train_imported_cases, imported_pad])
        total_input = np.concatenate([train_total_cases, train_pad])

        print("Length Community training cases: ", len(train_local_cases))
        print("Length Dorm training cases: ", len(train_dorm_cases))
        print("Length into model: ",
              len(local_model_input)," ",
              len(dorm_model_input)," ",
              len(imported_input))

        len_observed = len(local_model_input)

        return local_model_input, dorm_model_input, imported_input, total_input, len_observed

    elif separate_quarantine:

        if not local_cases_breakdown_data_path:
            raise ValueError("Must have path for case quarantine breakdown data if separate quaratine is true")

        _, dorm_cases, imported_cases, total_cases, _, dates= import_case_data(cases_data_path,
                                                          start_id,
                                                          separate_dorms=True)
        

        model_input_local_unquarantined, model_input_local_quarantined = import_local_breakdown(local_cases_breakdown_data_path,
                                                                                                start_id, end_date,
                                                                                                prediction_t=prediction_t)


        train_dorm_cases = dorm_cases[0:end_date]
        train_imported_cases = imported_cases[0:end_date]
        train_total_cases = total_cases[0:end_date]
        train_dates = dates[0:end_date]

        print("Train Start Date: ", train_dates[0])
        print("Train End Date: ", train_dates[-1])

        train_pad = [np.nan for x in range(prediction_t)]
        
        if imported_case_extra == 'last':
            imported_pad = [train_imported_cases[-1] for _ in range(prediction_t)]

        elif imported_case_extra == 'ma':
            print('imported cases ma')
            ma_week = np.round(np.mean(train_imported_cases[-7::]))
            imported_pad = [ma_week for x in range(prediction_t)]
        else:
            imported_pad = imported_case_extra

        dorm_model_input = pd.Series(np.concatenate([train_dorm_cases, train_pad]))
        imported_input = np.concatenate([train_imported_cases, imported_pad])
        total_input = np.concatenate([train_total_cases, train_pad])

        print("Length Community training cases: ", len(model_input_local_unquarantined), " ", len(model_input_local_quarantined))
        print("Length Dorm training cases: ", len(train_dorm_cases))
        print("Length into model: ",
              len(model_input_local_unquarantined)," ",
              len(dorm_model_input)," ",
              len(imported_input))

        len_observed = len(model_input_local_unquarantined)

        return [model_input_local_unquarantined, model_input_local_quarantined], dorm_model_input, imported_input, total_input, len_observed

    else:
        local_cases, imported_cases, _ = import_case_data(cases_data_path,
                                                          start_id,
                                                          separate_dorms=False)
        
        train_local_cases = local_cases[0:end_date]
        train_imported_cases = imported_cases[0:end_date]

        #actual = local_cases[end_date:end_date+prediction_t]

        train_pad = [np.nan for x in range(prediction_t)]

        if imported_case_extra == 'last':
            imported_pad = [train_imported_cases[-1] for x in range(prediction_t)]

        elif imported_case_extra == 'ma':
            print('imported cases ma')
            ma_week = np.round(np.mean(train_imported_cases[-7::]))
            imported_pad = [ma_week for x in range(prediction_t)]
        else:
            imported_pad = imported_case_extra

        model_input = pd.Series(np.concatenate([train_local_cases, train_pad]))
        imported_input = np.concatenate([train_imported_cases, imported_pad])

        print("Length training cases: ", len(train_local_cases))
        print("Length into model: ", len(model_input)," ",len(imported_input))

        len_observed = len(model_input)

        return model_input, imported_input, len_observed


def get_train_prediction_dates(cases_data_path, start_id,
                             end_date, prediction_t):


    _, _, _, _, _, dates= import_case_data(cases_data_path,
                                                        start_id,
                                                        separate_dorms=True)

    train_dates = dates[0:end_date]
    
    all_dates = np.concatenate([train_dates,
                           [train_dates[-1] + np.timedelta64(x,'D') for x in range(1,prediction_t+1)]])
    
    train_dates = [pd.Timestamp(x) for x in list(train_dates)]
    all_dates = [pd.Timestamp(x) for x in list(all_dates)]
    # Get total dates

    return train_dates, all_dates


def import_local_breakdown(local_cases_breakdown_data_path, start_id, end_date, prediction_t=7):
    cases_breakdown = pd.read_csv(local_cases_breakdown_data_path)

    # Add for Eugene's modification
    if ('Community Cases (After 11th Oct)' in cases_breakdown.columns) and ('Community ART Cases' in cases_breakdown.columns):
        print("epidemic split curve is new format, adding 11th Oct AND ART cases to original community cases")
        # Add all cases after 11th oct as unlinked community cases
        cases_breakdown['Community Cases Detected through Surveillance'] = cases_breakdown['Community Cases Detected through Surveillance'] + cases_breakdown['Community Cases (After 11th Oct)'] + cases_breakdown['Community ART Cases']


    elif ('Community Cases (After 11th Oct)' in cases_breakdown.columns):
        print("epidemic split curve is new format, adding 11th Oct cases to original community cases")
        # Add all cases after 11th oct as unlinked community cases
        cases_breakdown['Community Cases Detected through Surveillance'] = cases_breakdown['Community Cases Detected through Surveillance'] + cases_breakdown['Community Cases (After 11th Oct)']


    model_input_local_unquarantined = cases_breakdown['Community Cases Detected through Surveillance'].values[start_id:start_id+end_date]
    model_input_local_quarantined = cases_breakdown['Community Cases Isolated before Detection'].values[start_id:start_id+end_date]


    if prediction_t > 0:
        train_pad = [np.nan for _ in range(prediction_t)]

        model_input_local_unquarantined = pd.Series(np.concatenate([model_input_local_unquarantined, train_pad]))
        model_input_local_quarantined = pd.Series(np.concatenate([model_input_local_quarantined, train_pad]))

    else:
        model_input_local_unquarantined = pd.Series(model_input_local_unquarantined)
        model_input_local_quarantined = pd.Series(model_input_local_quarantined)
    
    return model_input_local_unquarantined, model_input_local_quarantined


def generate_ground_truth_forecast(cases_data_path, start_id,
                             end_date, prediction_t):
    
        local_cases, dorm_cases, imported_cases, total_cases, _ = import_case_data(cases_data_path,
                                                          start_id,
                                                          separate_dorms=True)
        
        forecast_local_cases = local_cases[end_date:end_date+prediction_t]
        forecast_dorm_cases = dorm_cases[end_date:end_date+prediction_t]
        forecast_imported_cases = imported_cases[end_date:end_date+prediction_t]
        forecast_total_cases = total_cases[end_date:end_date+prediction_t]

        return forecast_local_cases, forecast_dorm_cases, forecast_imported_cases, forecast_total_cases


def generate_NPI_prediction_data_v1(NPI_data_path, start_id,
 end_date, prediction_t):
    
    NPIS_wanted = import_NPIS_v1(NPI_data_path, start_id)
    NPIS_array = NPIS_wanted.iloc[0:end_date+prediction_t]
    NPIS_array.loc[NPIS_array['Gathering_Max']==150,'Gathering_Max'] = 10
    NPIS_array = NPIS_array.values
    #num_NPIS = NPIS_array.shape[1]
    #print(NPIS_array.shape)

    return NPIS_array

def generate_NPI_prediction_data(NPI_data_path, start_id,
 end_date, prediction_t):
    
    NPIS_wanted, dates = import_NPIS(NPI_data_path, start_id)
    NPIS_array = NPIS_wanted.iloc[0:end_date+prediction_t]
    #NPIS_array.loc[NPIS_array['Gathering_Max']==150,'Gathering_Max'] = 10
    NPIS_array = NPIS_array.values
    #num_NPIS = NPIS_array.shape[1]
    #print(NPIS_array.shape)

    return NPIS_array, dates


def import_owid_test_counts(owid_data_path):
    """ Combines OWID data and repository data and outputs a dataframe with columns Data, test_positivity rate and daily number tests
    """

    #### Process OWID Data ####
    owid = pd.read_csv(owid_data_path)
    singapore_testing = owid[owid['location']=='Singapore']
    singapore_testing = singapore_testing[~singapore_testing['total_tests'].isna()][['date',
                                                                                 'total_tests',
                                                                                 'new_tests',
                                                                                 'positive_rate',
                                                                                 'tests_per_case']]
    singapore_testing['test_done_in_week'] = singapore_testing['total_tests'].diff()
    singapore_testing['average_test_daily'] = singapore_testing['test_done_in_week']/7.0
    daily_dt_range = pd.date_range(start=singapore_testing['date'].iloc[0], end=singapore_testing['date'].iloc[-1])

    singapore_testing_daily = singapore_testing.set_index('date')
    singapore_testing_daily.index = pd.to_datetime(singapore_testing_daily.index)
    singapore_testing_daily = singapore_testing_daily.reindex(daily_dt_range)

    singapore_testing_daily['average_test_daily'] = singapore_testing_daily['average_test_daily'].bfill()
    singapore_testing_daily = singapore_testing_daily.reset_index()
    singapore_testing_daily = singapore_testing_daily.rename({'index':'Date'},axis=1)

    return singapore_testing_daily

def import_repo_test_counts(repo_data_path):
    swab_fig_present = pd.read_csv(repo_data_path)
    swab_fig_present = swab_fig_present.rename(columns={'Unnamed: 0':'date'})
    swab_fig_present['date'] = pd.to_datetime(swab_fig_present['date'])
    swab_fig_present = swab_fig_present.groupby('date', as_index=False).first()

    daily_dt_range_2 = pd.date_range(start=swab_fig_present['date'].iloc[0],
                                end=swab_fig_present['date'].iloc[-1])

    swab_fig_present_daily = swab_fig_present.set_index('date')
    swab_fig_present_daily.index = pd.to_datetime(swab_fig_present_daily.index)
    swab_fig_present_daily = swab_fig_present_daily.reindex(daily_dt_range_2)

    swab_fig_present_daily['average_daily_number_of_swabs_tested_over_the_past_week_approx'] = \
        swab_fig_present_daily['average_daily_number_of_swabs_tested_over_the_past_week_approx'].bfill()

    swab_fig_present_daily = swab_fig_present_daily.reset_index()
    swab_fig_present_daily = swab_fig_present_daily.rename({'index':'Date'},axis=1)

    return swab_fig_present_daily

def import_data_gov_test_counts(data_path):
    swabs_new = pd.read_csv(data_path)
    swabs_new['date'] = pd.to_datetime(swabs_new['date'])
    daily_dt_range = pd.date_range(start=swabs_new['date'].iloc[0],
                                end=swabs_new['date'].iloc[-1])


    swabs_new_daily = swabs_new.set_index('date')
    swabs_new_daily.index = pd.to_datetime(swabs_new_daily.index)
    swabs_new_daily = swabs_new_daily.reindex(daily_dt_range)

    swabs_new_daily['total_daily_test_art_pcr'] = swabs_new_daily['average_daily_number_of_pcr_swabs_tested'] + \
                                                swabs_new_daily['average_daily_number_of_art_swabs_tested_over_the_past_week']

    swabs_new_daily['total_daily_test_art_pcr'] = \
        swabs_new_daily['total_daily_test_art_pcr'].bfill()

    swabs_new_daily = swabs_new_daily.reset_index()
    swabs_new_daily = swabs_new_daily.rename({'index':'Date'},axis=1)
    
    return swabs_new_daily


def get_comb_test_counts(owid_data_path, repo_data_path, data_gov_path):
    
    owid_test_counts = import_owid_test_counts(owid_data_path)
    repo_test_counts = import_repo_test_counts(repo_data_path)
    data_gov_test_counts = import_data_gov_test_counts(data_gov_path)

    combine_owid_repo = owid_test_counts.merge(repo_test_counts,
                                                 on='Date',
                                                 how='outer')

    # Combine OWID + REPO
    comb_daily_test_1 = agg_cols(combine_owid_repo['average_test_daily'],
                                combine_owid_repo['average_daily_number_of_swabs_tested_over_the_past_week_approx'])

    combine_owid_repo['Combined_Daily_Test'] = comb_daily_test_1
    combine_owid_repo = combine_owid_repo[['Date','Combined_Daily_Test']]

    # Combine (OWID+REPO) + DataGov
    combine_owid_repo  = combine_owid_repo.merge(data_gov_test_counts, on='Date', how='outer')
    # Get combined tests
    comb_daily_test_2 = agg_cols(combine_owid_repo['Combined_Daily_Test'],combine_owid_repo['total_daily_test_art_pcr'])
    combine_owid_repo['Combined_Daily_Test'] = comb_daily_test_2

    return combine_owid_repo

def get_test_positivity_rate_raw(owid_data_path, repo_data_path, case_counts_path,data_gov_path, start_val=100):

    test_counts = get_comb_test_counts(owid_data_path, repo_data_path,data_gov_path)

    # Get case counts
    case_counts = pd.read_csv(case_counts_path)
    case_counts = case_counts.rename(columns={'Unnamed: 0':'Date'})
    case_counts['Date'] = pd.to_datetime(case_counts['Date'])
    case_counts['total_cases'] = case_counts['Community Total Cases'] + case_counts['Dormitory Residents Total Cases']

    test_positivity_frame = case_counts[['Date','total_cases']].merge(test_counts[['Date','Combined_Daily_Test']],
                                         how='left',
                                         on='Date')

    # Forward fill any missing days
    test_positivity_frame['Combined_Daily_Test'] = test_positivity_frame['Combined_Daily_Test'].ffill()

    # Backfill
    test_positivity_frame.loc[0,'Combined_Daily_Test'] = start_val
    test_positivity_frame['Combined_Daily_Test'] = test_positivity_frame['Combined_Daily_Test'].interpolate()
    test_positivity_frame['Positivity_Rate'] = test_positivity_frame['total_cases']/test_positivity_frame['Combined_Daily_Test']

    return test_positivity_frame

def get_test_positivity_data_model(owid_data_path, repo_data_path,
                                  case_counts_path,data_gov_path, start_id,
                                  end_date, prediction_t=7, start_val=100, scale=True,
                                  extrapolate_type='ma'):
    # Only returns the number of tests conducted daily
    train_test_positivity = get_test_positivity_rate_raw(owid_data_path, repo_data_path,
                                                         case_counts_path,data_gov_path,
                                                         start_val=start_val)
                                                         
    test_counts = train_test_positivity['Combined_Daily_Test'].values
    test_counts = test_counts[start_id: start_id+end_date]

    if scale:
        mm_scaler = MinMaxScaler()
        test_counts = mm_scaler.fit_transform(test_counts.reshape(-1,1))
        test_counts = test_counts.reshape(-1)
    
    if prediction_t > 0:
        if extrapolate_type == 'last':
            test_counts_pad = [test_counts[-1] for x in range(prediction_t)]

        elif extrapolate_type == 'ma':
            print('imported cases ma')
            ma_week = np.mean(test_counts[-7::])
            test_counts_pad = [ma_week for x in range(prediction_t)]

        test_counts = np.concatenate([test_counts, test_counts_pad])

    return test_counts

def import_shn_data(shn_data_path):
    shn =  pd.read_csv(shn_data_path)
    shn = shn.rename(columns={'Unnamed: 0':'Date'})
    shn['Date'] = pd.to_datetime(shn['Date'])

    return shn

def import_quarantine_data(quarantine_data_path):
    """ Combines OWID data nad resposity data and outputs a dataframe with Data, number in quarantine, SHN and total isolated (quarantine+SHN)
    """
    active_quarantine = pd.read_csv(quarantine_data_path)
    active_quarantine = active_quarantine.rename(columns={'Unnamed: 0':'Date'})
    active_quarantine['Date'] = pd.to_datetime(active_quarantine['Date'])

    active_quarantine['Total_under_quarantine'] = active_quarantine['Persons under quarantine in Home Quarantine Order'] + \
                                                active_quarantine['Persons under quarantine in Home Quarantine Order'] +\
                                                active_quarantine['Persons under quarantine in situ'].fillna(0) 
                                                
    return active_quarantine


def get_isolation_data_raw(shn_data_path, quarantine_data_path, case_counts_path, shn_start_val=0, quarantine_start_val=0):

    shn = import_shn_data(shn_data_path)
    active_quarantine = import_quarantine_data(quarantine_data_path)

    case_counts = pd.read_csv(case_counts_path)
    case_counts = case_counts.rename(columns={'Unnamed: 0':'Date'})
    case_counts['Date'] = pd.to_datetime(case_counts['Date'])

    start_date = case_counts['Date'].iloc[0]
    end_date = case_counts['Date'].iloc[-1]

    combined_notices = shn.merge(active_quarantine[['Date',
                            'Total_under_quarantine']],
                             on='Date',
                             how='outer')
    
    combined_notices = combined_notices.ffill()

    # Expand date range to match the case count data frame
    combined_notices = combined_notices.set_index("Date")
    daily_dt_range = pd.date_range(start=start_date, end=end_date)
    combined_notices = combined_notices.reindex(daily_dt_range)
    combined_notices.reset_index(inplace=True)
    combined_notices = combined_notices.rename(columns={'index':'Date'})

    # Combine SHN numbers
    combined_notices['SHN'] = combined_notices['Individuals in Stay Home Notices (Hotel)'] + \
                            combined_notices['Individuals in Stay Home Notices (Home)'] 

    # Interpolate the missing back values
    combined_notices.loc[0,'SHN'] = shn_start_val
    combined_notices.loc[0,'Total_under_quarantine'] = quarantine_start_val

    combined_notices['SHN'] = combined_notices['SHN'].interpolate()
    combined_notices['Total_under_quarantine'] = combined_notices['Total_under_quarantine'].interpolate()

    # Since we are interested in total number of people in isolation
    combined_notices['Total_in_iso'] = combined_notices['Total_under_quarantine'] + \
                                    combined_notices['SHN']
    
    return combined_notices


def get_isolation_data_model(shn_data_path, quarantine_data_path,
                             case_counts_path, start_id, end_date,
                             shn_start_val=0, prediction_t=7,
                             quarantine_start_val=0, scale=True,extrapolate_type='ma'):

    isolation_data = get_isolation_data_raw(shn_data_path, quarantine_data_path,
                                            case_counts_path, shn_start_val=shn_start_val,
                                            quarantine_start_val=quarantine_start_val)

    isolation_counts = isolation_data['Total_in_iso'].values
    isolation_counts = isolation_counts[start_id: start_id+end_date]

    if scale:
        mm_scaler = MinMaxScaler()
        isolation_counts = mm_scaler.fit_transform(isolation_counts.reshape(-1,1))
        isolation_counts = isolation_counts.reshape(-1)
    
    if prediction_t > 0:
        if extrapolate_type == 'last':
            isolation_counts_pad = [isolation_counts[-1] for x in range(prediction_t)]

        elif extrapolate_type == 'ma':
            print('imported cases ma')
            ma_week = np.mean(isolation_counts[-7::])
            isolation_counts_pad = [ma_week for _ in range(prediction_t)]

        isolation_counts = np.concatenate([isolation_counts, isolation_counts_pad])

    return isolation_counts


def import_qo_issued_data(qo_issued_path):
    qo_issued = pd.read_csv(qo_issued_path)
    qo_issued = qo_issued.rename(columns={'Unnamed: 0':'Date'})
    qo_issued['Date'] = pd.to_datetime(qo_issued['Date'])
    return qo_issued

def import_shn_issued_data(shn_issued_path):
    shn_issued = pd.read_csv(shn_issued_path)
    shn_issued = shn_issued.rename(columns={'Unnamed: 0':'Date'})
    shn_issued['Date'] = pd.to_datetime(shn_issued['Date'])
    shn_issued['Total_SHN_issued'] = shn_issued['Daily Stay Home Notices issued (Hotel)'] + \
                            shn_issued['Daily Stay Home Notices issued (Home)']
    return shn_issued

def get_orders_issued_raw(shn_issued_path,
                          quarantine_issued_path,
                          case_counts_path,
                          shn_issued_start_val=0, qo_issued_start_val=0, shn_only=False):
    
    shn_issued = import_shn_issued_data(shn_issued_path)
    qo_issued = import_qo_issued_data(quarantine_issued_path)

    case_counts = pd.read_csv(case_counts_path)
    case_counts = case_counts.rename(columns={'Unnamed: 0':'Date'})
    case_counts['Date'] = pd.to_datetime(case_counts['Date'])

    start_date = case_counts['Date'].iloc[0]
    end_date = case_counts['Date'].iloc[-1]

    if not shn_only:
        combined_issues = qo_issued.merge(shn_issued[['Date',
                                'Total_SHN_issued']],
                                on='Date',
                                how='outer')
    else:
        combined_issues = shn_issued

    combined_issues = combined_issues.ffill()

    # Expand date range to match the case count data frame
    combined_issues = combined_issues.set_index("Date")
    daily_dt_range = pd.date_range(start=start_date, end=end_date)
    combined_issues = combined_issues.reindex(daily_dt_range)
    combined_issues.reset_index(inplace=True)
    combined_issues = combined_issues.rename(columns={'index':'Date'})


    # Interpolate the missing back values
    combined_issues.loc[0,'Total_SHN_issued'] = shn_issued_start_val

    if not shn_only:
        combined_issues.loc[0,'Quarantine Orders issued'] = qo_issued_start_val

    combined_issues['Total_SHN_issued'] = combined_issues['Total_SHN_issued'].interpolate()

    if not shn_only:
        combined_issues['Quarantine Orders issued'] = combined_issues['Quarantine Orders issued'].interpolate()
    
    if not shn_only:
        combined_issues['Total_isolation_issued'] = combined_issues['Total_SHN_issued'] +\
            combined_issues['Quarantine Orders issued']
    else:
        combined_issues['Total_isolation_issued'] = combined_issues['Total_SHN_issued']

    #print("First Date: ",combined_issues['Date'].iloc[0])
    #print("Latest Date: ",combined_issues['Date'].iloc[-1])
    
    return combined_issues


def get_orders_issued_model(shn_issued_path, quarantine_issued_path,
                             case_counts_path, start_id, end_date,
                             shn_issued_start_val=0, prediction_t=7,
                             qo_issued_start_val=0, scale=True,extrapolate_type='ma', shn_only=False):

    orders_issued_data = get_orders_issued_raw(shn_issued_path, quarantine_issued_path,
                                            case_counts_path, shn_issued_start_val=shn_issued_start_val,
                                            qo_issued_start_val=qo_issued_start_val, shn_only=shn_only)

    orders_issued_vals = orders_issued_data['Total_isolation_issued'].values
    orders_issued_vals = orders_issued_vals[start_id: start_id+end_date]

    if scale:
        mm_scaler = MinMaxScaler()
        orders_issued_vals = mm_scaler.fit_transform(orders_issued_vals.reshape(-1,1))
        orders_issued_vals = orders_issued_vals.reshape(-1)
    
    if prediction_t > 0:
        if extrapolate_type == 'last':
            orders_issued_pad = [orders_issued_vals[-1] for x in range(prediction_t)]

        elif extrapolate_type == 'ma':
            print('imported cases ma')
            ma_week = np.mean(orders_issued_vals[-7::])
            orders_issued_pad = [ma_week for _ in range(prediction_t)]

        orders_issued_vals = np.concatenate([orders_issued_vals, orders_issued_pad])

    return orders_issued_vals




def get_vax_rate(vax_rate_path,
                 prediction_t_vax,
                 preappend,
                 pfizer_peak_ve = 0.7207,
                 moderna_peak_ve = 0.7530,
                 pfizer_peak_ve_time = 8,
                 moderna_peak_ve_time = 8,
                 pfizer_daily_decrease = 0.0045,
                 moderna_daily_decrease = 0.0022,
                 natural_immunity_rate = 0.1):


    vaccine_data = pd.read_csv(vax_rate_path)
    vaccine_data['date'] = pd.to_datetime(vaccine_data['date'], dayfirst=True)

    if prediction_t_vax:
        print("Adding Vaccine Rows for prediction: ", prediction_t_vax)\
        
        pfizer_addition = np.mean(vaccine_data['Fully Vaccinated with Pfizer'].iloc[-7:].diff())
        moderna_addition = np.mean(vaccine_data['Fully Vaccinated with Moderna'].iloc[-7:].diff())
        
        
        # Create new dataframe to do extrapolation
        # Number of months to extend
        vaccine_data_temp = vaccine_data.copy()
        vaccine_data_temp.set_index('date', inplace=True)

        # Extrapolate the index first based on original index
        extrapolated_vaccine_data = pd.DataFrame(
                                data=vaccine_data_temp,
                                index=pd.date_range(
                                    start=vaccine_data_temp.index[0],
                                    periods=len(vaccine_data_temp.index) + prediction_t_vax,
                                    freq=vaccine_data_temp.index.freq
            )
        )
        
        last_value_pfizer = vaccine_data['Fully Vaccinated with Pfizer'].iloc[-1]
        last_value_moderna = vaccine_data['Fully Vaccinated with Moderna'].iloc[-1]
        
        moderna_extrapolated_values = last_value_moderna + moderna_addition*np.arange(1,prediction_t_vax+1)
        pfizer_extrapolated_values = last_value_pfizer + pfizer_addition*np.arange(1,prediction_t_vax+1)
        
        extrapolated_vaccine_data['Fully Vaccinated with Pfizer'].iloc[-prediction_t_vax::] = pfizer_extrapolated_values
        extrapolated_vaccine_data['Fully Vaccinated with Moderna'].iloc[-prediction_t_vax::] = moderna_extrapolated_values
        
        extrapolated_vaccine_data = extrapolated_vaccine_data.reset_index()
        extrapolated_vaccine_data = extrapolated_vaccine_data.rename(columns={'index':'date'})
        
        vaccine_data = extrapolated_vaccine_data
    
    vaccine_data['Pfizer Daily Addition'] = vaccine_data['Fully Vaccinated with Pfizer'].diff().fillna(0)
    vaccine_data['Moderna Daily Addition'] = vaccine_data['Fully Vaccinated with Moderna'].diff().fillna(0)



def get_ascertainment_regime(ascertainment_regime_path, start_id, end_date, prediction_t):
    ascertainment_regime = pd.read_csv(ascertainment_regime_path)
    ascertainment_regime['Date'] = pd.to_datetime(ascertainment_regime['Date'],format='%d/%m/%Y')

    ascertainment_regime_wanted = ascertainment_regime.iloc[start_id: start_id+end_date+prediction_t]

    print("Ascertainment Regime Start Date: ", ascertainment_regime_wanted['Date'].iloc[0])
    print("Ascertainment Regime End Date: ", ascertainment_regime_wanted['Date'].iloc[-1])

    return ascertainment_regime_wanted['Testing_Regime'].values


def get_variant_pct_info(variant_info_path, start_id, end_date, prediction_t):

    variants_df = pd.read_csv(variant_info_path)
    variants_df['Date'] = pd.to_datetime(variants_df['Date Cleaned'], format='%d/%m/%Y')

    variants_df_wanted = variants_df.iloc[start_id: start_id+end_date+prediction_t]

    print("Variant Regime Start Date: ", variants_df_wanted['Date'].iloc[0])
    print("Variant Regime End Date: ", variants_df_wanted['Date'].iloc[-1])

    sequence = ['pct_original','pct_delta','pct_omicron_int']
    print("Variant Sequence: ", sequence)

    variants_pct = variants_df_wanted[sequence].values

    return variants_pct

def add_art_data(epidemic_split_curve_path, data_gov_path, output_path):

    epidemic_split_curve = pd.read_csv(epidemic_split_curve_path)
    data_gov_curve = pd.read_csv(data_gov_path)

    epidemic_split_curve['Unnamed: 0'] = pd.to_datetime(epidemic_split_curve['Unnamed: 0'])
    data_gov_curve['press_release_date'] = pd.to_datetime(data_gov_curve['press_release_date'])

    epidemic_split_curve_mod = epidemic_split_curve.merge(data_gov_curve, how='left', left_on='Unnamed: 0', right_on='press_release_date')

    epidemic_split_curve_mod['data_gov_comm_cases'] = epidemic_split_curve_mod['total_count_of_case'] - \
                                                epidemic_split_curve_mod['DormResidents (After 11th Oct)']

    epidemic_split_curve_mod['comm_cases_art_pcr'] = \
        [x if not math.isnan(x) else y for y,x in zip(epidemic_split_curve_mod['Community Cases (After 11th Oct)'],
                                    epidemic_split_curve_mod['data_gov_comm_cases'])]

    # Change values
    epidemic_split_curve_mod['Community Cases (After 11th Oct)'] = epidemic_split_curve_mod['comm_cases_art_pcr']

    # Save
    epidemic_split_curve_mod.to_csv(output_path, index=False)


def agg_cols(s1,s2):
    output = []
    for v1,v2 in zip(s1,s2):
        if v1 and math.isnan(v2):
            val = v1
        elif v2 and math.isnan(v1):
            val = v2
        else:
            val = 0.5*(v1+v2)
        output.append(val)
    return output