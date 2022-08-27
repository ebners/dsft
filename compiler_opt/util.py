import math
import os
import numpy as np
from typing import List, Tuple, Union
import functools
import tqdm
from joblib import Parallel
import multiprocessing
import statistics
import threading
import multiprocessing as mp


"""
This module provides generally usable functions and constants.
"""


FLOAT_REGEX = "[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
"""A regex which matches floating point numbers"""

def most_diverse_subset(baseset: List[float], out_size: int) -> List[int]:
    """
    Finds for a given set of values a subset of the given size, such that the minimal difference between two entries is maximal.

    Arguments
    ---
    - baseset: A list of values
    - out_size: The desired size of the subset. Must be in `range(0, len(baseset)+1)`
    
    Returns
    ---
    A list of indices, indicating the elements of `baseset`, which are included in the subset found.
    """
    n = len(baseset)
    assert(out_size >= 0 and out_size <= len(baseset))
    indices = sorted(range(n), key=functools.cmp_to_key(lambda l, r: baseset[l] - baseset[r]))

    if out_size == 0:
        return []
    if out_size == 1:
        return [indices[0]]
    
    while len(indices) > out_size:
        remove = 1
        min_dist = baseset[indices[-1]] - baseset[indices[0]]
        for i in range(1, len(indices)-1):
            candidate = baseset[indices[i+1]] - baseset[indices[i-1]]
            if candidate < min_dist:
                min_dist = candidate
                remove = i
        del indices[remove]
    return indices

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

_func = None

def _worker_init(func):
  global _func
  _func = func
  

def _worker(x):
  return _func(x)


def xmap(func, iterable, processes=None):
  with multiprocessing.Pool(processes, initializer=_worker_init, initargs=(func,)) as p:
    return p.map(_worker, iterable)

def remove_outliers(measurements: np.ndarray, out_percentile: float) -> np.ndarray:
    """
    Removes outliers from the given numpy array by fitting a gaussian distribution and removing points which are too improbable.

    Arguments
    ---
    - out_percentile: A value between 0 and 1, indicating the probability of false positives based on the fitte gaussian.
    """
    if len(measurements) <= 1:
        return measurements
    stddev = np.std(measurements)
    if stddev == 0:
        return measurements
    elif stddev < 0:
        print("Somehow numpy calculated a negative stddev...")
        return measurements
    dist = statistics.NormalDist(mu=np.mean(measurements), sigma=stddev)
    min_allowed = dist.inv_cdf(out_percentile/2)
    max_allowed = dist.inv_cdf(1 - out_percentile/2)
    return measurements[(measurements > min_allowed) & (measurements < max_allowed)]

def measurements_valid(measurements: np.ndarray, max_std: float) -> Tuple[bool, float]:
    """
    Returns whether the given measurements are stable or not.

    Arguments
    ---
    - measurements: A numpy array of measurements
    - max_std: The maximal standard deviation relative to the mean that would still count as stable.
    """
    # check variance
    mean = np.mean(measurements)
    rel_std = np.std(measurements) / mean
    return rel_std <= max_std, rel_std

def test_null_hyp(null_hyp: statistics.NormalDist, t, sigma_fact=2):
    """
    Tests a null hypothesis (two-sided) for a given test statistic.

    Arguments
    ---
    - null_hyp: The null hypothesis as a normal distribution
    - t: The test statistic. Can be a numpy array or a scalar
    - sigma_fact: A factor of the standard deviation to be used as an indirect significance level.
    """
    return np.abs(t - null_hyp.mean) <= sigma_fact*null_hyp.stdev

_counter = 0
_counter_lock = threading.Lock()
def new_int() -> int:
    """Returns a new unique integer. Never returns the same integer twice."""
    global _counter
    _counter_lock.acquire()
    out = _counter
    _counter += 1
    _counter_lock.release()
    return out

def get_pbar(pbar: Union[tqdm.tqdm, bool], iterations: int) -> tqdm.tqdm:
    """
    Utility function which takes a progress bar or a boolean and returns a new progress bar with the given number of iteration if the boolean is True, or else just the argument itself.
    """
    return tqdm.trange(iterations) if pbar and isinstance(pbar, bool) else pbar

_func = None

def _worker_init(func):
    global _func
    _func = func

def _worker(x):
    return _func(x)

def parallel_map(func, items: List, procs=None) -> List:
    with mp.Pool(initializer=_worker_init, initargs=(func,), processes=procs) as p:
        return p.map(_worker, items)

def get_abs_path(path: str, prefix: str):
    return path if path.startswith("/") else os.path.join(prefix, path)

def list_to_str(l: List, sep: str = " ", start: str = "[", end: str = "]") -> str:
    return start + functools.reduce(lambda x, y: str(x) + sep + str(y), l, "") + end

def str_to_list(s: str, sep: str = " ", start: str = "[", end: str = "]") -> List:
    s = s[len(start):-len(end)]
    return s.split(sep)
