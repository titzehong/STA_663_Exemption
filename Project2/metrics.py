import numpy as np

def calc_wmape(ground_truth: np.array, predictions: np.array) -> float:
    """ Reference Formula
    https://en.wikipedia.org/wiki/WMAPE
    """
    assert ground_truth.shape == predictions.shape

    numerator = np.sum(np.abs(ground_truth - predictions))
    denom = np.sum(np.abs(ground_truth))
    wmape_score = numerator / denom

    return wmape_score

def calc_mean_cov(predictions: np.array) -> float:
    """
    Reference:
    https://en.wikipedia.org/wiki/Coefficient_of_variation
    """

    std_dev = np.std(predictions)
    mean_vals = np.mean(predictions)
    mean_cov_score = std_dev / mean_vals

    return mean_cov_score

def calc_cov_v2(predictions_mean: np.array, predictions_sd: np.array, agg_type: str='mean') -> float:
    """
    Reference:
    https://en.wikipedia.org/wiki/Coefficient_of_variation
    """
    
    cov_scores = predictions_sd / predictions_mean

    if agg_type == 'mean':
        return np.mean(cov_scores)
    elif agg_type == 'median': 
        return np.median(cov_scores)
    else:
        raise ValueError('??')

def calc_acc_ci(ground_truth: np.array,
                lower_bound: np.array,
                upper_bound: np.array):

    assert ground_truth.shape == lower_bound.shape == upper_bound.shape

    acc_vec = (lower_bound <= ground_truth) & (upper_bound >= ground_truth) 
    acc_ci_score = np.mean(acc_vec)
    return acc_ci_score

if __name__ == '__main__':
    gt = np.array([10,15,13,9,2])

    preds = np.array([5,1,23,4,1])
    lb =  np.array([9,10,12,8,1])
    ub =  np.array([11,20,15,10,3])

    print(calc_wmape(gt, preds))
    print(calc_mean_cov(preds))
    print(calc_acc_ci(gt, lb, ub))
