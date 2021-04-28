import numpy as np
import collections

def clip01(x_test, min=0, max=100):
    return np.clip(x_test,min, max)

def move_axis(x, first_axis=1,second_axis=2):
    return np.moveaxis(x,first_axis,second_axis)

def compute_mean(x, axis=0):
    return np.mean(x,axis=axis)

def compute_var(x, axis=0):
    return np.var(x,axis=axis)

def compute_mean_std(x, axis=0, return_dict=False):
    if return_dict:
        data = {"mean":np.mean(x,axis=axis), "std":np.var(x,axis=axis)}
    else:
        data = np.array([np.mean(x, axis=axis), np.std(x, axis=axis)])
    return data

def compute_mean_var(x, axis=0, return_dict=False):
    if return_dict:
        data = {"mean":np.mean(x,axis=axis), "var":np.var(x,axis=axis)}
    else:
        data = np.array([np.mean(x, axis=axis), np.var(x, axis=axis)])
    return data

# def flatten_dict(d, parent_key='', sep='_'):
#     """
#     Dives deep into every level of the dict, and combines the keys of each level to form a single long key.
#     The search depth stops when an iterable is met at that level such as numpy array.
#     For example, a dict of {"level1":{"level2":[1,2,3,4]}} becomes {"level1-level2":[1,2,3,4]}
#
#     Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
#
#     Parameters
#     ----------
#     data_dict : dict
#         The multi-level dict to be `flattened`.
#
#     Returns
#     -------
#     flattened_dict : dict
#         The single level dict
#     """
#
#     items = []
#     for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
#         if isinstance(v, dict):
#             items.extend(flatten_dict(v, new_key, sep=sep).items())
#         else:
#             items.append((new_key, v))
#     return dict(items)

def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def flatten_apply_dict(d,func, parent_key='', sep='_',):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, func(v)))
    return dict(items)
