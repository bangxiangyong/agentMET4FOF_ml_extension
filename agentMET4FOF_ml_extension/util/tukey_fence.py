import numpy as np
import pandas as pd

# assign outlier labels
def tukey_fence_range(data, k=1.5, axis=0):
    """
    Given a pd.dataframe, returns the Tukey fences' lower and upper bounds

    Parameters
    ----------
    k : sensitivity of the fence
    axis : axis for the fence to be computed over

    Returns
    -------
    (lb,ub) : tuple
        Lower and upper bounds of the Tukey fence
    """
    q1 = np.quantile(data, 0.25, axis=axis)
    q3 = np.quantile(data, 0.75, axis=axis)
    iqr = q3 - q1

    lb = q1 - k * iqr
    ub = q3 + k * iqr

    return (lb,ub)

def tukey_fence_outlier(data : pd.DataFrame, k=1.5, axis=0, return_sum=True):
    """
    Takes in an pd Dataframe and assigns labels to outliers in the column based on Tukey Fence.
    0 = not an outlier (inlier)
    1 = (outlier < error_lb or outlier > error_ub)

    #NOTE: this is not optimised yet, much better work can be done to improve code quality.

    Returns
    -------
    outliers : np.ndarray of shape same as error_pd
    """
    lb, ub = tukey_fence_range(data, k=k, axis=axis)
    outliers = []
    for row in range(len(data)):
        temp_outlier = []
        for col in range(len(data.columns)):
            outlier_condition = 1 if (
                    data.iloc[row, col] > ub[col] or data.iloc[row, col] < lb[col]) else 0
            temp_outlier.append(outlier_condition)
        outliers.append(temp_outlier)
    outliers = np.array(outliers)

    if return_sum:
        return outliers.sum(1)
    else:
        return outliers
