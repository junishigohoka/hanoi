import numpy as np
cimport numpy as cnp
cimport cython

def sim_reproduction_c(int popsize, cnp.float64_t[:, :] genotype, cnp.float64_t[:] fitness):

    cdef int n_loci = genotype.shape[0]
    cdef cnp.float64_t[:, :] genotype_next
    cdef cnp.int64_t[:] n_offspring

    # when no individuals reproduce
    if np.sum(fitness) == 0:
        genotype_next = np.full((n_loci, popsize), np.nan)
        n_offspring = np.zeros(popsize, dtype = int)
        return np.asarray(genotype_next), np.asarray(n_offspring)

    # When at least one individuals reproduces
    # Total fitness
    #cdef cnp.float64_t fitness_sum = np.sum(fitness)
    # Declare variables
    cdef cnp.float64_t[:] pvals
    cdef cnp.int64_t[:] sam
    cdef int i, j, k, idx
    cdef cnp.float64_t bit1, bit2
    cdef cnp.float64_t fitness_sum = 0
    for i in range(popsize):
        fitness_sum += fitness[i]
    #cdef cnp.int64_t[:,:] genotype_next_T = np.empty(genotype.shape, dtype = np.int64)
    #cdef int i
    # compute binomial parameter for each individual
    pvals = np.empty(fitness.shape[0], dtype=np.float64)
    for i in range(fitness.shape[0]):
        pvals[i] = fitness[i] / fitness_sum
    n_offspring = np.random.multinomial(n = 2 * popsize, pvals = pvals) 
    # index of 2N parents
    sam = np.repeat(range(popsize), n_offspring)
    np.random.shuffle(sam)
    # Make genotype of next parents randomly take one haplotype per locus per parent
    genotype_next = np.empty((n_loci, popsize), dtype=np.float64)
    for i in range(popsize):
        idx = 2*i
        for j in range(n_loci):
            # sample two haplotypes
            bit1 = np.random.binomial(1, genotype[j, sam[idx]]/2)
            bit2 = np.random.binomial(1, genotype[j, sam[idx+1]]/2)
            genotype_next[j, i] = bit1 + bit2
    return np.asarray(genotype_next), np.asarray(n_offspring)
 
