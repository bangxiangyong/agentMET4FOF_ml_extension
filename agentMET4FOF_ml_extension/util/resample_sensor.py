import numpy as np
import pandas as pd

def resample_data(data_series, n=10):
    temp_df = pd.DataFrame(data_series)
    resampled_data = temp_df.groupby(np.arange(len(temp_df)) // n).mean().values.squeeze(-1)
    return resampled_data

def apply_along_sensor(data_3d, func, sensor_axis=1, **func_params):
    """
    Assuming we have a 3D data array of default (num_examples, sensor_axis, num_measurements),
    This wraps a function to create a partial function which applies `func` along the sensor_axis independently.
    This is useful, for instance, in cases where we need to apply FFT, or a resampling method along the time-series domain.

    Example: `resampled_dt = apply_along_sensor(data_3d, resample_data, sensor_axis=2, n=100)`

    """
    new_data = []
    for sensor_i in range(data_3d.shape[sensor_axis]):
        sensor_dt = np.take(data_3d, indices=sensor_i, axis=sensor_axis)
        new_data.append(np.apply_along_axis(arr=sensor_dt, func1d=func,axis=-1, **func_params))
    new_data = np.array(new_data)
    new_data = np.moveaxis(new_data, 0,sensor_axis)
    return new_data

