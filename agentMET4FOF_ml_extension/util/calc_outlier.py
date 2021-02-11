import numpy as np
from functools import reduce
from itertools import combinations

def get_outlier_arg(y_col, q1=0.25, q3=0.75):
    Q1 =  np.quantile(y_col, q1, axis=0)
    Q3 = np.quantile(y_col, q3, axis=0)
    IQR = Q3 - Q1

    outlier_args = np.argwhere((y_col <Q1-1.5*IQR) | (y_col > Q3+1.5*IQR))
    return outlier_args

def multi_intersection(*np_arrays):
    intersected = reduce(np.intersect1d, (np_arrays))
    return intersected

def multi_intersection_combo(y_outliers, level = 1):
    combi = list(combinations(np.arange(len(y_outliers)), level))
    y_outliers_levelup = [multi_intersection(*(y_outliers[i] for i in combi_i)) for combi_i in
                          combi]
    y_outliers_levelup = np.unique(np.concatenate(y_outliers_levelup))
    return y_outliers_levelup

def calc_outlier_range_(y_col, q1=0.25, q3=0.75):
    """
    A sub-function, used in calculating the (Q1,Q3,IQR) of a column
    """
    Q1 =  np.quantile(y_col, q1, axis=0)
    Q3 = np.quantile(y_col, q3, axis=0)
    IQR = Q3 - Q1

    return Q1,Q3,IQR

def calc_outlier_ranges(y_df):
    """
    Returns
    -------
    np.array of (Q1, Q3, IQR) for each column of y_df.
    Use this to determine outlier
    """

    return np.array([calc_outlier_range_(y_col) for y_col in y_df.T])

def get_num_outliers(y_row, outlier_ranges, return_sum = True):
    """
    Parameters
    ----------
    y_row : A row of N columns, which obtained from y_df
    outlier_ranges : np.array of (Q1, Q3, IQR) for each column of y_df. Use the output from `calc_outlier_ranges` method as this parameter.
    Returns
    -------
    num_outlier_cols : (int) total number of columns with outliers
    """
    num_outlier_cols = (y_row < outlier_ranges[:, 0] - 1.5 * outlier_ranges[:, 2]) | (
                y_row > outlier_ranges[:, 1] + 1.5 * outlier_ranges[:, 2]).astype(int)
    if return_sum:
        num_outlier_cols = np.sum(num_outlier_cols)
    return num_outlier_cols

def get_num_outliers_df(y_df):
    """
    1. Determines the outlier ranges
    2. For every row, determine number of columns in the y_df which is an outlier
    Note: 0 means no outlier. The higher the number, the higher the `degree of outlierness` i.e defect.
    """
    outlier_ranges = calc_outlier_ranges(y_df)
    num_outliers = np.apply_along_axis(get_num_outliers, axis=1, arr=y_df, outlier_ranges=outlier_ranges)
    return num_outliers