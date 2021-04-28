import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def plot_time_series_samples(x, y_target, metadata=None,  sensor_axis=1, figsize=(18,5), nrows=3, ncols=5):
    plots = []
    if isinstance(x, dict):
        for key in x.keys():
            plots.append(plot_time_series_samples_(x[key], y_target[key], metadata=metadata,
                                                   sensor_axis=sensor_axis,figsize=figsize,
                                                   nrows=nrows,ncols=ncols, suptitle="Axis: "+key))
    return plots

def plot_time_series_samples_(ts_x, y_target, metadata=None,  sensor_axis=1, figsize=(18,5), nrows=3, ncols=5, suptitle=""):
    if metadata is not None:
        units = metadata["units"]
        labels = metadata["labels"]
        metadata_able = True
    else:
        metadata_able = False
    num_sensors = ts_x.shape[sensor_axis]
    num_examples = ts_x.shape[0]
    num_samples = ts_x.shape[1 if sensor_axis ==2 else 2]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    num_axes = len(axes)
    for sensor_i in range(num_sensors):
        if sensor_i < num_axes:
            if sensor_axis == 1:
                ts_sample = ts_x[np.random.randint(num_examples), sensor_i]
            else:
                ts_sample = ts_x[np.random.randint(num_examples), :, sensor_i]

            axes[sensor_i].plot(np.arange(num_samples),ts_sample)
            if metadata_able:
                axes[sensor_i].set_ylabel(units[sensor_i])
                axes[sensor_i].set_title(labels[sensor_i])

    # remove unsused sensors
    if num_axes > num_sensors:
        for ax_i in range(num_sensors,num_axes):
            axes[ax_i].set_axis_off()

    fig.tight_layout()
    fig.suptitle(suptitle, y=1.00)
    return fig

def plot_uncertain_time_series(x, y_target, metadata,  sensor_axis=1, figsize=(18,5), nrows=3, ncols=5):
    plots = []
    if isinstance(x, dict):
        for key in x.keys():
            plots.append(plot_uncertain_time_series_(x[key], y_target[key], metadata,
                                                   sensor_axis=sensor_axis,figsize=figsize,
                                                   nrows=nrows,ncols=ncols, suptitle="Axis: "+key))
    return plots

def plot_uncertain_time_series_(ts_x_stats, y_target, metadata,  sensor_axis=1, figsize=(18,5), nrows=3, ncols=5, suptitle=""):
    units = metadata["units"]
    labels = metadata["labels"]

    ts_x = ts_x_stats[0] # mean
    ts_ux = ts_x_stats[1] # std
    ts_upper = ts_x+2*ts_ux
    ts_lower = ts_x-2 * ts_ux

    num_sensors = ts_x.shape[sensor_axis]
    num_examples = ts_x.shape[0]
    num_samples = ts_x.shape[1 if sensor_axis ==2 else 2]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    num_axes = len(axes)
    for sensor_i in range(num_sensors):
        if sensor_i < num_axes:
            if sensor_axis == 1:
                ts_sample_mu = ts_x[np.random.randint(num_examples), sensor_i]
                ts_sample_upper = ts_upper[np.random.randint(num_examples), sensor_i]
                ts_sample_lower = ts_lower[np.random.randint(num_examples), sensor_i]
            else:
                ts_sample_mu = ts_x[np.random.randint(num_examples), :, sensor_i]
                ts_sample_upper = ts_upper[np.random.randint(num_examples),:, sensor_i]
                ts_sample_lower = ts_lower[np.random.randint(num_examples),:, sensor_i]

            axes[sensor_i].plot(np.arange(num_samples),ts_sample_mu)
            axes[sensor_i].fill_between(np.arange(num_samples),ts_sample_upper,ts_sample_lower,alpha=0.5)

            axes[sensor_i].set_ylabel(units[sensor_i])
            axes[sensor_i].set_title(labels[sensor_i])

    # remove unsused sensors
    if num_axes > num_sensors:
        for ax_i in range(num_sensors,num_axes):
            axes[ax_i].set_axis_off()

    fig.tight_layout()
    fig.suptitle(suptitle, y=1.00)
    return fig