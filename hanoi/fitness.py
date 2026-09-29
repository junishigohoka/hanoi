from .hanoi import *
import numpy as np
import math
from scipy.spatial.distance import squareform



def fit_neutral(z):
    """
    Computes fitness based on a uniform fitness function (i.e. neutrality).

    Arguments:
        z:      (N, n) array representing n-dimensional phenotypes of N individuals.
    Returns: (N,) array representing fitness of N individuals
    """
    return np.ones(z.shape[0], dtype = np.float64)


def fit_gaus(sigma, z_opt, z):
    """
    Computes fitness based on a Gaussian fitness function.

    Arguments:
        sigma: A scalar specifying SD around the peak
        z_opt: (n,) array specifying the coordinate of the peak in the n dimensional phenotypic space
        z:      (N, n) array representing n-dimensional phenotypes of N individuals.
    Returns: (N,) array representing fitness of N individuals
    """
    return np.array([np.exp(-(np.linalg.norm(z_i - z_opt))**2/(2 * sigma**2)) for z_i in z])


def fit_multimodal(sigma, p, z_opt, z):
    """
    Computes fitness based on a multimodal fitness function.

    Arguments:
        sigma:  (n_peaks,) array specifying SD of n_peak peaks
        p:      (n_peaks,) array specifying relative height of peaks
        z_opt:  (n_peaks, n) array specifying the coordinate of n_peak peaks in the n-dimensional phenotype space
        z:      (N, n) array representing n-dimensional phenotypes of N individuals.
    Returns: (N,) array representing fitness of N individuals
    """
    w = np.zeros(z.shape[0])
    # Loop over peaks
    for i in range(len(p)):
        w += p[i] * fit_gaus(sigma = sigma[i], z_opt = z_opt[i], z = z)
    return w


def fit_step(boxes, z):
    """
    Computes fitness based on a step fitness function.

    Arguments:
        z:     (N, n) array of n-dimensional phenotypes of N individuals.
        boxes: (m, 2, n) array of m boxes, where boxes[i, 0, :] = lower bounds, boxes[i, 1, :] = upper bounds of i-th box. 
               Fitness of phenotype within boxes is 1.

    Returns: (N,) array of 0 or 1
    """
    return np.any(
        np.all((z[:, None, :] >= boxes[:, 0, :]) & (z[:, None, :] <= boxes[:, 1, :]), axis=2),
        axis=1
    ).astype(float)



def fit_sigmoid(a, b, z):
    """
    Computes fitness based on a generaised logistic function.

    Arguments:
        a: 1D array-like of n specifying the "centre" of a linear subspace where fitness is 1/2
        b: 1D array-like of n specifying the direction and steepness of the slope orthogonal to the linear subspace where fitness is 1/2.
        z: (N, n) array representing n-dimensional phenotypes of N individuals.
    Returns: (N,) array representing fitness of N individuals
    """
    N = z.shape[0]
    n = z.shape[1]
    a = np.repeat(np.asarray(a)[None, :], N, axis = 0)
    b = np.repeat(np.asarray(b)[None, :], N, axis = 0)
    #return (1/(1 + np.exp(-b * (z - a)))**(1/n)).prod(axis = 1)
    return (1/(1 + np.exp((-b * (z - a)).sum(axis = 1))))



def wrap_pheno(z, period):
    """
    Wraps phenotype values into the range [-period/2, period/2).

    Arguments:
        z:      array of phenotype values (any range)
        period: scalar or array specifying the full cycle length
                (e.g. 24 for time-of-day in hours, 2*pi for radians)

    Returns: array of same shape as z, wrapped into [-period/2, period/2)

    Example: period=24 (hours), z=25 (i.e. 1am the next day) wraps to 1.
    """
    return np.mod(z + period / 2, period) - period / 2


def fit_gaus_circular(sigma, z_opt, z, period):
    """
    Computes fitness based on a Gaussian fitness function on wrapped (circular) phenotypes.

    Arguments:
        sigma:  A scalar specifying SD around the peak (same units as z)
        z_opt:  (n,) array specifying the peak coordinate(s), in the range of period
        z:      (N, n) array of n-dimensional circular phenotypes, in any representation
                of the period (need not be pre-wrapped)
        period: scalar or (n,) array specifying the full cycle length of each dimension
                (e.g. 24 for time-of-day in hours, 2*pi for radians)
    Returns: (N,) array representing fitness of N individuals
    """
    d = wrap_pheno(z - z_opt, period)
    return np.exp(-(d**2).sum(axis=1) / (2 * sigma**2))


def fit_step_circular(arcs, z, period):
    """
    Computes fitness based on a step fitness function on wrapped (circular) phenotypes.

    Arguments:
        z:      (N, n) array of n-dimensional circular phenotypes, in any representation
                of the period (need not be pre-wrapped)
        arcs:   (m, 2, n) array of m arcs, arcs[i, 0, :] = lower bounds, arcs[i, 1, :] = upper bounds,
                in the range of period. If lower > upper for a given dimension, the arc wraps around.
        period: scalar or (n,) array specifying the full cycle length of each dimension

    Returns: (N,) array of 0 or 1
    """
    lo = arcs[:, 0, :]
    hi = arcs[:, 1, :]
    z_ = z[:, None, :]
    arc_len = np.mod(hi - lo, period)
    offset = np.mod(z_ - lo, period)
    in_dim = offset <= arc_len
    return np.any(np.all(in_dim, axis=2), axis=1).astype(float)
