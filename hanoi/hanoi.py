import numpy as np
import math
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
    """
    A class to represent breeding value of all individuals in a population

    Attributes
    ----------
    mean : Breeding value of mean $A_m$.
           Should be a 2D np.ndarray of shape (n, N).
           mean[i, j] holds the breeding value of mean of trait i for individual j.
    var : Breeding value of variance $A_v$
           Should be a 2D np.ndarray of shape (n, N).
           var[i, j] holds the breeding value of variance of trait i for individual j.
    cov : Breeding value of covariance $A_c$.
           Should be a 2D np.ndarray of shape (choose(n, 2), N).
           cov[i, j] holds the breeding value of covariance of the trait pair i for individual j.
    n : Number of traits
    N : Number of individuals


    Methods
    -------
    show() : Prints information of the BreedVal object.

    """
    def __init__(self, mean: np.ndarray, var: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.var = var 
        self.cov = cov 
        self.N = mean.shape[1]
        self.n = mean.shape[0]
        if self.n == 1:
            self.varcov = self.var.reshape((self.N, 1, 1))
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
    """
    A class to represent one generation of a population

    Attributes
    ----------
    genotype :      Genotype matrix of parents.
                    Should be a 2D np.ndarray of shape (L, N).
                    genotype[i, j] holds the genotype of individual j at locus i.
    breed_val :     Breeding value of parents.
                    Should be of type hanoi.BreedVal.
    phenotype :     Phenotype matrix of parents.
                    Should be a 2D np.ndarray of shape (N, n).
                    phenotype[i, j] holds the phenotype of individual i for trait j.
    fitness :       Fitness matrix of parents.
                    Should be a 1D np.ndarray of length N.
                    fitness[i] holds the fitness of individual i.
    n_offspring :   Number of offspring.
                    Should be a 1D np.ndarray of length N and the sum should be 2N.
                    n_offspring[i] holds the number of offspring of parent individual i.
                    Selfing is allowed, and in case of selfing, the number of offspring is counted twice.
    genotype_next : Genotype matrix of offspring.
                    Should be a 2D np.ndarray of shape (L, N).
                    genotype_next[i, j] holds the genotype of individual j at locus i.
    n : Number of traits
    N : Number of individuals
    """
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
    """
    A class to represent one generation of a population

    Attributes
    ----------
    n_gen :         Number of generations.
                    Should be positive int.
    genotype :      Genotype matrices of parents.
                    Should be a 3D np.ndarray of shape (n_gen, L, N).
                    genotype[t, i, j] holds the genotype of individual j at locus i at generation t.
    breed_val :     Breeding values of parents.
                    Should be a list of n_gen elements of hanoi.BreedVal.
                    breed_val[t] holds hanoi.BreedVal of generation t.
    phenotype :     Phenotype matrices of parents.
                    Should be a 3D np.ndarray of shape (n_gen, N, n).
                    phenotype[t, i, j] holds the phenotype of individual i for trait j at generation t.
    fitness :       Fitness matrices of parents.
                    Should be a 2D np.ndarray of shape (n_gen, N).
                    fitness[t, i] holds the fitness of individual i at generation t.
    n_offspring :   Number of offspring matrix.
                    Should be a 2D np.ndarray of shape (n_gen, N) and the sum(axis = 1) shuold be 2N.
                    n_offspring[t, i] holds the number of offspring of parent individual i at generation t.
                    Selfing is allowed, and in case of selfing, the number of offspring is counted twice.
    genotype_next : Genotype matrices of offspring.
                    Should be a 3D np.ndarray of shape (n_gen, L, N).
                    genotype_next[t, i, j] holds the genotype of individual j at locus i at generation t.
    n : Number of traits
    N : Number of individuals

    Methods
    -------
    show() : Prints information of the Generations object.
    """
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
    def allele_freqs_all(self):
        return np.vstack([self.allele_freqs(), self.allele_freq_last()])

    def show(self):
        print(f"Number of traits:\n {self.phenotype.shape[1]}")
        print(f"Number of individuals:\n {self.phenotype.shape[2]}")
        print(f"Number of generations:\n {self.phenotype.shape[0]}")




class SampleData:
    """
    A class to represent sample data

    Attributes
    ----------
    mut_effect : Sample MutEffect data.
    geno :  2D np.ndarray for sample genotype data.
    z_ref : Sample z_ref
    c_ref : Sample c_ref
    """
    def __init__(self, mut_effect, geno, z_ref, c_ref):
        self.mut_effect = mut_effect
        self.geno = geno
        self.z_ref = z_ref
        self.c_ref = c_ref



def comp_breed_val(mut_effect: MutEffect, genotype: np.ndarray):
    """
    Computes breeding value from mutation effect and genotype.

    Args:
        mut_effect : hanoi.MutEffect object representing mutation effect.
        genotype :   2D np.ndarray representing a genotype table.

    Returns: hanoi.BreedVal object representing breeding value
    """
    A_m = 1/2 * mut_effect.mean @ genotype
    A_v = 1/2 * mut_effect.var @ genotype
    A_c = 1/2 * mut_effect.cov @ genotype
    return BreedVal(mean = A_m, var = A_v, cov = A_c)


def cov_mtx(var, cov):
    """
    Computes 2D variance-covariance matrix from variance and covariance.

    Arguments:
        var: 1D array-like of n representing variance of n traits.
        cov: 1D array-like of choose(n, 2) representing covariance between n traits.

    Returns: (n, n) np.ndarray representing variance-covariance matrix.
    """
    return squareform(cov) + np.diag(var)


