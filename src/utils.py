from concurrent import futures
import copy
import datetime
import os
import pickle

import mergedeep

import numpy as np
import pandas as pd


def merge_dicts(a: dict, b: dict, strategy=mergedeep.Strategy.ADDITIVE) -> dict:
    a = copy.deepcopy(a)
    mergedeep.merge(a, b, strategy=strategy)
    return a


def pandas_options(max_columns=None, expand_frame_repr=False, min_rows=100, max_rows=100):
    """
    Every time I use pandas I have to look this up, so created a convencience function here.
    """
    pd.options.display.max_columns = max_columns
    pd.options.display.expand_frame_repr = expand_frame_repr
    pd.options.display.min_rows = min_rows
    pd.options.display.max_rows = max_rows


def parexec(exec_func, nblocks, nworkers=None, *args, **kwargs):
    if nworkers is None:
        nworkers = nblocks
    with futures.ThreadPoolExecutor(nworkers) as executor:
        fs = futures.wait([executor.submit(exec_func, i, *args, **kwargs) for i in range(nblocks)])
    return [f.result() for f in fs.done]


def edges_to_centers(edges, log=False):
    if log:
        edges = np.log2(edges)
    centers = edges[1:] - 0.5 * (edges[1] - edges[0])
    if log:
        centers = 2 ** centers
    return centers


def centers_to_edges(centers, log=False):
    if log:
        centers = np.log2(centers)
    if len(centers) == 1:
        dx = 1
    else:
        dx = centers[1] - centers[0]
    edges = np.r_[centers, centers[-1] + dx] - 0.5 * dx
    if log:
        edges = 2 ** edges
    return edges


def edge_meshgrid(centers_x, centers_y, logx=False, logy=False):
    return np.meshgrid(centers_to_edges(centers_x, logx), centers_to_edges(centers_y, logy))


class Timer:

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed(self):
        return datetime.datetime.now() - self.start

    def restart(self):
        elapsed = self.elapsed()
        self.start = datetime.datetime.now()
        return elapsed

    def __str__(self):
        return self.elapsed().__str__()


def pkl_save(obj, filename):
    pathname = os.path.split(filename)[0]
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=5)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result


def isnewer(src, dst):
    if os.path.exists(dst):
        return os.path.getmtime(src) > os.path.getmtime(dst)
    else:
        return True


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
