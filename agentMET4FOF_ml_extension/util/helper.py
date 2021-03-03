import numpy as np

def clip01(x_test, min=0, max=100):
    return np.clip(x_test,min, max)

def move_axis(x, first_axis=1,second_axis=2):
    return np.moveaxis(x,first_axis,second_axis)

def compute_mean(x, axis=0):
    return np.mean(x,axis=axis)

def compute_mean_std(x, axis=0):
    data = np.array([np.mean(x,axis=axis), np.std(x,axis=axis)])
    return data
