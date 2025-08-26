import numpy as np
cimport numpy as cnp
cimport cython
import jax.numpy as jnp

cpdef tuple sim_reproduction_c(int popsize, cnp.float64_t[:, :] genotype, cnp.float64_t[:] fitness):

    cdef int n_loci = genotype.shape[1]
    cdef cnp.float64_t[:, :] genotype_next
    cdef cnp.int64_t[:] n_offspring

    # when no individuals reproduce
    if np.sum(fitness) == 0:
        genotype_next = np.full((popsize, n_loci), np.nan)
        n_offspring = np.zeros(popsize, dtype = int)
        return np.asarray(genotype_next), np.asarray(n_offspring)

    # When at least one individuals reproduces
    # Declare variables
    cdef cnp.float64_t[:] pvals
    cdef cnp.int64_t[:] sam
    cdef int i, j, idx
    cdef cnp.float64_t bit1, bit2
    cdef cnp.float64_t fitness_sum = 0
    # Total fitness
    for i in range(popsize):
        fitness_sum += fitness[i]
    # compute binomial parameter for each individual
    pvals = np.empty(fitness.shape[0], dtype=np.float64)
    for i in range(fitness.shape[0]):
        pvals[i] = fitness[i] / fitness_sum
    n_offspring = np.random.multinomial(n = 2 * popsize, pvals = pvals) 
    # index of 2N parents
    sam = np.repeat(range(popsize), n_offspring)
    np.random.shuffle(sam)
    # Make genotype of next parents randomly take one haplotype per locus per parent
    genotype_next = np.empty((popsize, n_loci), dtype=np.float64)
    for i in range(popsize):
        idx = 2*i
        for j in range(n_loci):
            # sample two haplotypes
            bit1 = np.random.binomial(1, genotype[sam[idx], j]/2)
            bit2 = np.random.binomial(1, genotype[sam[idx+1], j]/2)
            genotype_next[i, j] = bit1 + bit2
    return np.asarray(genotype_next), np.asarray(n_offspring)
 



cpdef cnp.ndarray[cnp.float64_t, ndim=2] sim_pheno_c(object breed_val, cnp.float64_t[:] mean_0, cnp.float64_t[:,:] cov_0):
    cdef cnp.float64_t[:,:] z
    if breed_val.n == 1:
        z = np.random.normal(mean_0[0] + breed_val.mean, cov_0[0,0] + breed_val.var, (breed_val.N, breed_val.n))
        return np.asarray(z)
        #return z
    # Cholesky decomposition of the covariance matrices
    #chol = jnp.linalg.cholesky(cov_0 + breed_val.varcov)
    ## Standard normal sampling
    #z_std = np.random.standard_normal((breed_val.N, breed_val.n))
    ## Convert N vectors of n standard normal variables to the N phenotype values in n dimensions
    #z = np.einsum('ijk,ik->ij', chol, z_std) + mean_0
    #return z
