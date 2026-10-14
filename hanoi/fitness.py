from .hanoi import *
import numpy as np
import math
from scipy.spatial.distance import squareform



def fit_neutral(z):
    """
    Computes fitness based on a uniform fitness function (i.e. neutrality).

    Parameters
    ----------
    z : np.ndarray
        Array of shape (N, n) representing n-dimensional phenotypes of N individuals.

    Returns
    -------
    np.ndarray
        Array of shape (N,) representing fitness of N individuals.
    """
    return np.ones(z.shape[0], dtype = np.float64)


def fit_gaus(sigma, z_opt, z):
    """
    Computes fitness based on a Gaussian fitness function.

    Parameters
    ----------
    sigma : float
        Standard deviation around the peak.
    z_opt : np.ndarray
        Array of shape (n,) specifying the coordinate of the peak in the n-dimensional phenotypic space.
    z : np.ndarray
        Array of shape (N, n) representing n-dimensional phenotypes of N individuals.

    Returns
    -------
    np.ndarray
        Array of shape (N,) representing fitness of N individuals.
    """
    return np.array([np.exp(-(np.linalg.norm(z_i - z_opt))**2/(2 * sigma**2)) for z_i in z])


def fit_multimodal(sigma, p, z_opt, z):
    """
    Computes fitness based on a multimodal fitness function.

    Parameters
    ----------
    sigma : np.ndarray
        Array of shape (n_peaks,) specifying the standard deviation of each peak.
    p : np.ndarray
        Array of shape (n_peaks,) specifying the relative height of each peak.
    z_opt : np.ndarray
        Array of shape (n_peaks, n) specifying the coordinates of the peaks in the n-dimensional phenotypic space.
    z : np.ndarray
        Array of shape (N, n) representing n-dimensional phenotypes of N individuals.

    Returns
    -------
    np.ndarray
        Array of shape (N,) representing fitness of N individuals.
    """
    w = np.zeros(z.shape[0])
    # Loop over peaks
    for i in range(len(p)):
        w += p[i] * fit_gaus(sigma = sigma[i], z_opt = z_opt[i], z = z)
    return w


def fit_step(boxes, z):
    """
    Computes fitness based on a step fitness function.

    Parameters
    ----------
    boxes : np.ndarray
        Array of shape (m, 2, n) of m boxes, where boxes[i, 0, :] holds the lower bounds
        and boxes[i, 1, :] the upper bounds of the i-th box.
        Fitness of a phenotype within any box is 1, and 0 otherwise.
    z : np.ndarray
        Array of shape (N, n) of n-dimensional phenotypes of N individuals.

    Returns
    -------
    np.ndarray
        Array of shape (N,) of 0 or 1.
    """
    return np.any(
        np.all((z[:, None, :] >= boxes[:, 0, :]) & (z[:, None, :] <= boxes[:, 1, :]), axis=2),
        axis=1
    ).astype(float)



def fit_sigmoid(a, b, z):
    """
    Computes fitness based on a generalised logistic function.

    Parameters
    ----------
    a : array-like
        1D array-like of length n specifying the "centre" of a linear subspace where fitness is 1/2.
    b : array-like
        1D array-like of length n specifying the direction and steepness of the slope
        orthogonal to the linear subspace where fitness is 1/2.
    z : np.ndarray
        Array of shape (N, n) representing n-dimensional phenotypes of N individuals.

    Returns
    -------
    np.ndarray
        Array of shape (N,) representing fitness of N individuals.
    """
    N = z.shape[0]
    n = z.shape[1]
    a = np.repeat(np.asarray(a)[None, :], N, axis = 0)
    b = np.repeat(np.asarray(b)[None, :], N, axis = 0)
    #return (1/(1 + np.exp(-b * (z - a)))**(1/n)).prod(axis = 1)
    return (1/(1 + np.exp((-b * (z - a)).sum(axis = 1))))
