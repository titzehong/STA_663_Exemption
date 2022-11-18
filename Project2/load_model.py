import csv
import pickle
import scipy.stats as ss
import numpy as np
from sm_data import import_case_data
from pathlib import Path
import argparse


def conf_int_scipy(x, ci=0.95):
    low_per = 100*(1-ci)/2.
    high_per = 100*ci + low_per
    cis = ss.scoreatpercentile(x, [low_per, high_per])
    return cis


def round_row(row):
    for i in range(1, len(row)):
        row[i] = round(row[i])
    return row


def export_csv(input_file, output_dir, col_name, current_datetime, round=False):
    output_file = Path(output_dir) / f'{col_name}.csv'
    with open(output_file, 'w', newline='', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['date', 'lower_95', 'upper_95', 'lower_67', 'upper_67', 'median', 'mean'])
        with open(input_file, 'rb') as f:
            trace = pickle.load(f)
            t_arr = trace[col_name].T
            mean_arr = trace[col_name].mean(0)
            for i in range(0, len(mean_arr)):
                lower_95, upper_95 = conf_int_scipy(t_arr[i], ci=0.95)
                lower_67, upper_67 = conf_int_scipy(t_arr[i], ci=0.67)
                median = np.median(t_arr[i])
                current_date = np.datetime_as_string(current_datetime, unit='D')
                output_row = [current_date, lower_95, upper_95, lower_67, upper_67, median, mean_arr[i]]
                if round:
                    output_row = round_row(output_row)
                csv_writer.writerow(output_row)
                current_datetime = current_datetime + np.timedelta64(1, 'D')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_id', type=int, required=True)
    parser.add_argument('--start_id', type=int, required=True)
    args = parser.parse_args()
    # end_id = 278
    # start_id = 344
    input_dir = f"../Results/{args.end_id}_prediction_ahead_30_2_1500_3000_Analysis/"
    load_fp = Path(input_dir) / 'trace.pkl'
    cases_data_path = f"../covid19-modelling-sg/data/statistics/epidemic_curve.csv"
    local_cases, dorm_cases, imported_cases, total_cases, _, dates = import_case_data(cases_data_path,
                                                                                      args.start_id,
                                                                                      separate_dorms=True)
    export_csv(load_fp, Path(input_dir), 'infections_local_uncontained', dates[0], round=True)
    export_csv(load_fp, Path(input_dir), 'infections_local_contained', dates[0], round=True)
    export_csv(load_fp, Path(input_dir), 'infections_dorm', dates[0], round=True)
    export_csv(load_fp, Path(input_dir), 'test_adjusted_positive_jit_local', dates[0], round=True)
    export_csv(load_fp, Path(input_dir), 'test_adjusted_positive_jit_dorm', dates[0], round=True)
    export_csv(load_fp, Path(input_dir), 'r_t_local', dates[0])
    export_csv(load_fp, Path(input_dir), 'r_t_foreign', dates[0])

