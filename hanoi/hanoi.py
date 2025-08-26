from .sim import sim_reproduction_c
import numpy as np
import math
import jax.numpy as jnp
from scipy.spatial.distance import squareform


class MutEffect:
    """
    A class to represent mutational effects on mean, variance, and covariance of traits.

    Attributes
    ----------
    mean : Mean effects. 
        Should be a 2D np.array of shape (n, L). 
        mean[i, j] holds the effect of allele 1 at locus j on trait i.
    var : Variance effects.
        Should be a 2D np.array of shape (n, L).
        var[i, j] holds the effect of allele 1 at locus j on trait i.
    cov : Covariance effects.
        Should be a 2D np.array of shape (choose(n, 2), L).
        cov[i, j] holds the effect of allele 1 at locus j on the i-th trait pair.
    n : Number of traits
    L : Number of loci


    Methods
    -------
    show() : Prints information of the MutEffect object.

    """
    def __init__(self, mean: np.ndarray, var: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.var = var
        self.n = self.mean.shape[0]
        self.L = self.mean.shape[1]
        if self.n == 1:
            self.cov = np.zeros((0, self.L))
        else:
            self.cov = cov
        for arr in [self.mean, self.var, self.cov]:
            if arr.ndim != 2:
                raise ValueError("mean, var, and cov should be 2D")
        if self.n != self.var.shape[0]:
            raise ValueError("mean and var should have the same number of rows (n)")
        if self.L != self.var.shape[1]:
            raise ValueError("mean and var should have the same number of columns (L)")
        if self.cov.shape[0] != math.comb(self.n, 2):
            raise ValueError("The number of rows of cov should be n choose 2")
    def show(self):
        print(f"Number of traits n:\n{self.n}")
        print(f"Number of trait pairs:\n{self.cov.shape[0]}")
        print(f"Number of loci L:\n{self.L}")
        print(f"Mutation effect of mean:\n{self.mean}")
        print(f"Mutation effect of variance:\n{self.var}")
        print(f"Mutation effect of covariance:\n{self.cov}")



class BreedVal:
    def __init__(self, mean: np.ndarray, var: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.var = var 
        self.cov = cov 
        self.N = mean.shape[1]
        self.n = mean.shape[0]
        if self.n == 1:
            self.varcov = np.empty((0, self.N))
        else:
            self.varcov = np.array([cov_mtx(var = self.var[i], cov = self.cov[i]) for i in range(self.N)])
        for arr in (self.var, self.cov):
            if arr.ndim != 2:
                raise ValueError("mean, var, and cov must be 2D arrays.")
        Ns = [self.var.shape[1], self.cov.shape[1]]
        if not all(N == self.N for N in Ns):
            raise ValueError("mean, var, and cov must have the same number of columns (N)")
        if self.n != var.shape[0]:
            raise ValueError("mean and var must have the same number of rows (n)")
        if math.comb(self.n, 2) != self.cov.shape[0]:
            raise ValueError("The number of rows of cov should match the number of rows of mean choose 2")
    def show(self):
        print(f"Number of traits:\n {self.n}")
        print(f"Number of trait pairs:\n {math.comb(self.n, 2)}")
        print(f"Number of individuals:\n {self.N}")
        print(f"Breeding value of mean:\n {self.mean}")
        print(f"Breeding value of variance:\n {self.var}")
        print(f"Breeding value of covariance:\n {self.cov}")
        print(f"Covariance matrices of breeding values :\n {self.varcov}")


class Generation:
    def __init__(self, genotype: np.ndarray, 
                 genotype_next: np.ndarray,
                 breed_val: BreedVal, 
                 phenotype: np.ndarray, 
                 fitness: np.ndarray,
                 n_offspring: np.ndarray
                ):
        self.genotype = genotype
        self.genotype_next = genotype_next
        self.breed_val = breed_val
        self.phenotype = phenotype
        self.fitness = fitness
        self.n_offspring = n_offspring
    def allele_freqs(self):
        return (self.genotype.mean(axis=1)/2)[np.newaxis, :]
    def allele_freqs_next(self):
        return (self.genotype_next.mean(axis=1)/2)[np.newaxis, :]


class Generations(Generation):
    def __init__(self, *args, n_gen, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gen = n_gen
    def allele_freqs(self):
        #return np.array([genotype.mean(axis=1)/2 for genotype in generations.genotype_next])
        return np.array([genotype.mean(axis=1)/2 for genotype in self.genotype])
    def allele_freqs_next(self):
        #return np.array([genotype.mean(axis=1)/2 for genotype in generations.genotype_next])
        return np.array([genotype_next.mean(axis=1)/2 for genotype_next in self.genotype_next])
    def generation(self, t):
        return Generation(genotype = self.genotype[t], 
                          breed_val = self.breed_val[t], 
                          fitness = self.fitness[t], 
                          genotype_next = self.genotype_next[t], 
                          n_offspring = self.n_offspring[t], 
                          phenotype = self.phenotype[t]
                         )
    def allele_freq_last(self):
        return self.generation(-1).allele_freqs_next()[0]
    def allele_freq_first(self):
        return self.generation(0).allele_freqs()[0]

    def show(self):
        print(f"Number of traits:\n {self.phenotype.shape[2]}")
        print(f"Number of individuals:\n {self.phenotype.shape[1]}")
        print(f"Number of generations:\n {self.phenotype.shape[0]}")




def sim_generation(mut_effect, genotype, mut_rate, fit_func, mean_0, cov_0, **kwargs):
    popsize = genotype.shape[1]
    
    # Breeding value
    A = comp_breed_val(mut_effect=mut_effect, genotype=genotype)
    
    # Phenotype
    z = sim_pheno(breed_val=A, mean_0=mean_0, cov_0 = cov_0)
    
    # Phenotype to fitness
    if fit_func == "fit_gaus":
        w = fit_gaus(z=z, **kwargs)
    if fit_func == "fit_multimodal":
        w = fit_multimodal(sigma=sigma, p=kwargs.get('p'), z_opt=z_opt, z=z)
    if fit_func == "fit_neutral":
        w = fit_neutral(z=z)
    if fit_func == "fit_step":
        ranges = kwargs.get('boxes')
        w = fit_step(z=z, boxes=ranges)
    
    # Reproduction
    genotype_next, n_offspring = sim_reproduction_c(popsize, genotype, w)
    #if n_offspring.sum() == 0:
    #    raise RuntimeError("No individuals survived")
    genotype_next = sim_mutation(genotype=genotype_next, mut_rate=mut_rate) # This still returns nans if it is nans

    return Generation(genotype = genotype, 
                      genotype_next = genotype_next,
                      breed_val = A, 
                      phenotype = z, 
                      fitness = w,
                      n_offspring = n_offspring
                     )


def sim_generations(n_gen, mut_effect, genotype, mut_rate, fit_func, mean_0, cov_0, **kwargs):
    generations_list = []
    genotype_cur = genotype
    if n_gen == 0:
        i = 0
        while not (np.all(np.isin(genotype_cur, [0,2])) or np.all(np.isnan(genotype_cur))):
        #while np.any(genotype_cur == 1) or not np.isnan(genotype_cur[0,0]):
            generations_list.append(
                    sim_generation(mut_effect = mut_effect, 
                                   genotype = genotype_cur, 
                                   mut_rate = mut_rate, 
                                   fit_func = fit_func, 
                                   mean_0 = mean_0,
                                   cov_0 = cov_0,
                                   **kwargs))
            genotype_cur = generations_list[-1].genotype_next
            i+=1
    else:
        for i in range(n_gen):
            while not np.all(np.isnan(genotype_cur)):
                generations_list.append(
                        sim_generation(mut_effect = mut_effect, 
                                       genotype = genotype_cur, 
                                       mut_rate = mut_rate, 
                                       fit_func = fit_func, 
                                       mean_0 = mean_0,
                                       cov_0 = cov_0,
                                       **kwargs))
                genotype_cur = generations_list[-1].genotype_next
    generations = Generations(genotype = np.array([generation.genotype for generation in generations_list]), 
                              genotype_next = np.array([generation.genotype_next for generation in generations_list]), 
                              breed_val = np.array([generation.breed_val for generation in generations_list]),
                              phenotype = np.array([generation.phenotype for generation in generations_list]),
                              fitness = np.array([generation.fitness for generation in generations_list]),
                              n_offspring = np.array([generation.n_offspring for generation in generations_list]),
                              n_gen = i
                              )
    return generations




def comp_breed_val(mut_effect: MutEffect, genotype: np.ndarray):
    A_m = 1/2 * mut_effect.mean @ genotype
    A_v = 1/2 * mut_effect.var @ genotype
    A_c = 1/2 * mut_effect.cov @ genotype
    return BreedVal(mean = A_m, var = A_v, cov = A_c)





def sim_mutation(genotype, mut_rate):
    # Compute transition matrix
    transition_matrix = np.array([[(1 - mut_rate)**2, 2 * mut_rate * (1 - mut_rate), mut_rate**2],
                                  [mut_rate * (1 - mut_rate), (1 - mut_rate)**2 + mut_rate**2, mut_rate * (1 - mut_rate)],
                                  [mut_rate**2, 2 * mut_rate * (1 - mut_rate), (1 - mut_rate)**2]])

    # Count the number of entries in the genotype matrix for each of the 3 genotypes
    n_sites_genotype = np.array([(genotype == i).sum() for i in range(3)])
    # Make mapping of genotype to a set of index of the genotype matrix
    idx_sites_genotype = {i : np.where(genotype == i) for i in range(3)}
    # Make realised transition count matrix, where (i, j) has number of sites with genotype i which mutate into j
    n_sites_muts = np.array([np.random.multinomial(n_sites_genotype[i], transition_matrix[i]) for i in range(3)])
    # Introduce mutations
    for i in range(3):
        geno_muts = np.repeat([0,1,2], n_sites_muts[i])
        np.random.shuffle(geno_muts)
        genotype[idx_sites_genotype[i]] = geno_muts
    # Return mutated genotype matrix
    return genotype



def sim_pheno(breed_val: BreedVal, mean_0, cov_0):
    if breed_val.n == 1:
        z = np.random.normal(mean_0 + breed_val.mean, cov_0 + breed_val.var)
        return z
    # Cholesky decomposition of the covariance matrices
    chol = jnp.linalg.cholesky(cov_0 + breed_val.varcov)
    # Standard normal sampling
    z_std = np.random.standard_normal((breed_val.N, breed_val.n))
    # Convert N vectors of n standard normal variables to the N phenotype values in n dimensions
    z = np.einsum('ijk,ik->ij', chol, z_std) + mean_0
    return z




def sim_reproduction(popsize, genotype, fitness ):
    # number of offspring per genotype of each genotype
    n_loci = genotype.shape[0]
    if fitness.sum() == 0:
        genotype_next = np.full((n_loci, popsize), np.nan)
        n_offspring = np.zeros(popsize, dtype = int)
        #raise RuntimeError("Fitness of all individuals is 0")
        return genotype_next, n_offspring
    else:
        n_offspring = np.random.multinomial(n = 2 * popsize, pvals = fitness/fitness.sum())

        # index of 2N parents
        sam = np.repeat(range(popsize), n_offspring)
        np.random.shuffle(sam)
        # Make genotype of next parents randomly take one haplotype per locus per parent
        #genotype_next_T = np.array([np.sum(np.random.binomial(1, genotype[:, sam[[2 * i, 2 * i + 1]]]/2), axis = 0) for i in range(popsize)])
        genotype_next_T = np.random.binomial(1, genotype.T[sam,:]/2 ).reshape(popsize, 2, n_loci).sum(axis = 1)
        return genotype_next_T.T, n_offspring




def fit_neutral(z):
    return np.ones(z.shape[])


def fit_gaus(sigma, z_opt, z):
    return np.array([np.exp(-(np.linalg.norm(z_i - z_opt))**2/(2 * sigma**2)) for z_i in z])


def fit_multimodal(sigma, p, z_opt, z):
    w = np.zeros(len(z))
    # Loop over peaks
    for i in range(len(sigma)):
        w += p[i] * fit_gaus(sigma[i], z_opt, z)
    return w


def fit_step(boxes, z):
    """
    z:     (N, n) array of n-dimensional phenotypes of N individuals.
    boxes: (m, 2, n) array of m boxes, where boxes[i, 0] = lower bounds, boxes[i, 1] = upper bounds. Fitness of phenotype within boxes is 1.
    Returns: (N,) array of 0 or 1
    """
    return np.any(
        np.all((z[:, None, :] >= boxes[:, 0, :]) & (z[:, None, :] <= boxes[:, 1, :]), axis=2),
        axis=1
    ).astype(float)



def cov_mtx(var, cov):
    return squareform(cov) + np.diag(var)


