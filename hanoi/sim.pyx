#import numpy as np
#cimport numpy as cnp
#cimport cython

#cpdef tuple sim_reproduction(int popsize, cnp.ndarray[cnp.float64_t, ndim = 2] genotype, cnp.ndarray[cnp.float64_t, ndim = 1] fitness):
#
#    cdef int n_loci = genotype.shape[0]
#    cdef cnp.ndarray[cnp.float64_t, ndim = 2] genotype_next
#    cdef cnp.ndarray[cnp.int64_t, ndim = 1] n_offspring
#
#    # when no individuals reproduce
#    cdef cnp.float64_t fitness_sum = 0
#    cdef int i, j, idx
#    for i in range(popsize):
#        fitness_sum += fitness[i]
#    if fitness_sum == 0:
#        genotype_next = np.full((n_loci, popsize), np.nan)
#        n_offspring = np.zeros(popsize, dtype = int)
#        return np.asarray(genotype_next), np.asarray(n_offspring)
#
#    # When at least one individuals reproduces
#    # Total fitness
#    # Declare variables
#    cdef cnp.ndarray[cnp.float64_t, ndim = 1] pvals
#    cdef cnp.ndarray[cnp.int64_t, ndim = 1] sam1D
#    cdef cnp.ndarray[cnp.int64_t, ndim = 2] sam2D
#    cdef cnp.float64_t bit1, bit2
#    # compute binomial parameter for each individual
#    pvals = np.empty(fitness.shape[0], dtype=np.float64)
#    for i in range(fitness.shape[0]):
#        pvals[i] = fitness[i] / fitness_sum
#    n_offspring = np.random.multinomial(n = 2 * popsize, pvals = pvals) 
#    # index of 2N parents
#    sam1D = np.repeat(range(popsize), n_offspring)
#    np.random.shuffle(sam1D)
#    sam2D = sam1D.reshape(popsize, 2)
#    # Make genotype of next parents randomly take one haplotype per locus per parent
#    genotype_next = np.random.binomial(n = 1, p = genotype[:, sam2D]/2).sum(axis = 2).astype(np.float64)
#    return np.asarray(genotype_next), np.asarray(n_offspring)
 
