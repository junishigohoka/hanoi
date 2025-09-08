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
            print(self.cov.shape[0])
            print(math.comb(self.n, 2))
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
            self.varcov = np.array([cov_mtx(var = self.var[:,i], cov = self.cov[:,i]) for i in range(self.N)])
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
        print(f"Breeding value of variance-covariance:\n {self.varcov}")


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







def comp_breed_val(mut_effect: MutEffect, genotype: np.ndarray):
    A_m = 1/2 * mut_effect.mean @ genotype
    A_v = 1/2 * mut_effect.var @ genotype
    A_c = 1/2 * mut_effect.cov @ genotype
    return BreedVal(mean = A_m, var = A_v, cov = A_c)




def cov_mtx(var, cov):
    return squareform(cov) + np.diag(var)


