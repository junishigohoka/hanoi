from . import *
import numpy as np

def sample_data():
    N = 9
    n = 2
    L = 5

    M_m = np.array([[1, 0, 0, 0, 0], 
                    [0, 1, 0, 0, 0]],
                   dtype = np.float64
                   )

    M_v = np.array([[0, 0, 0.01, 0, 0], 
                    [0, 0, 0, 0.01, 0]],
                   dtype = np.float64
                   )

    M_c = np.array([[0, 0, 0, 0, -0.009]],
                   dtype = np.float64
                   )

    sample_mut_effect = hanoi.MutEffect(mean = M_m, var = M_v, cov = M_c)

    sample_geno = np.zeros((L, N))
    for i in range(L):
        sample_geno[i, i:(i+1)]  = 0
        sample_geno[i, (i+1):(i+2)] = 1
        sample_geno[i, (i+2):(i+3)] = 2

    sample_geno[:, 7] = 1
    sample_geno[:, 8] = 2

    sample_z_ref = np.array([0,0], dtype = np.float64)
    sample_c_ref = np.array([[0.01,0.00],
                      [0.00,0.01]], dtype = np.float64)


    sample_data = SampleData(mut_effect = sample_mut_effect, geno = sample_geno, z_ref = sample_z_ref, c_ref = sample_c_ref)
    return sample_data


