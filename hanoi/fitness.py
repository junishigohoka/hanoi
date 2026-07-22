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



def fit_linear(a, b, z):
    """
    Computes fitness based on a linear fitness function.

    Arguments:
        a: (n,) array representing slope of fitness along n traits
        b: A scalar representing intercept of fitness
        z:      (N, n) array representing n-dimensional phenotypes of N individuals.
    Returns: (N,) array representing fitness of N individuals
    """
    w_ori = z @ a + b
    # Shift fitness if negative fitness exists
    if np.all(w_ori >= 0):
        w = w_ori
    else:
        w = w_ori - w_ori.min()
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
