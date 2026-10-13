import numpy as np
import math
from scipy.spatial.distance import squareform


class MutEffect:
    """
    A class to represent mutational effects on mean, variance, and correlation of noise of traits.

    The effects on phenotypic noise are on transformed scales
    (log-additive model; see hanoi.comp_varcov):
    an allele multiplies the variance by exp(effect) and
    shifts the Fisher z-transformed correlation (artanh(r)) by the effect.

    Attributes
    ----------
    mean : Mean effects. 
        Should be a 2D np.array of shape (n, L). 
        mean[i, j] holds the effect of allele 1 at locus j on trait i.
    ln_var : Effects on log variance (log fold-change of variance).
        Should be a 2D np.array of shape (n, L).
        ln_var[i, j] holds the effect of allele 1 at locus j on ln(variance) of trait i.
    artanh_cor : Effects on Fisher z-transformed correlation of noise, i.e. artanh(correlation).
        Should be a 2D np.array of shape (choose(n, 2), L).
        artanh_cor[i, j] holds the effect of allele 1 at locus j on artanh(correlation) of the i-th trait pair.
        The order of the trait pairs is that of scipy.spatial.distance.squareform: (1,2), (1,3), ..., (2,3), ...
    n : Number of traits
    L : Number of loci


    Methods
    -------
    show() : Prints information of the MutEffect object.

    """
    def __init__(self, mean: np.ndarray, ln_var: np.ndarray, artanh_cor: np.ndarray):
        self.mean = mean
        self.ln_var = ln_var
        self.n = self.mean.shape[0]
        self.L = self.mean.shape[1]
        if self.n == 1:
            self.artanh_cor = np.zeros((0, self.L))
        else:
            self.artanh_cor = artanh_cor
        for arr in [self.mean, self.ln_var, self.artanh_cor]:
            if arr.ndim != 2:
                raise ValueError("mean, ln_var, and artanh_cor should be 2D")
        if self.n != self.ln_var.shape[0]:
            raise ValueError("mean and ln_var should have the same number of rows (n)")
        if self.L != self.ln_var.shape[1]:
            raise ValueError("mean and ln_var should have the same number of columns (L)")
        if self.artanh_cor.shape[0] != math.comb(self.n, 2):
            print(self.artanh_cor.shape[0])
            print(math.comb(self.n, 2))
            raise ValueError("The number of rows of artanh_cor should be n choose 2")
    def show(self):
        print(f"Number of traits n:\n{self.n}")
        print(f"Number of trait pairs:\n{self.artanh_cor.shape[0]}")
        print(f"Number of loci L:\n{self.L}")
        print(f"Mutation effect of mean:\n{self.mean}")
        print(f"Mutation effect of ln(variance):\n{self.ln_var}")
        print(f"Mutation effect of artanh(correlation):\n{self.artanh_cor}")


class BreedVal:
    """
    A class to represent breeding value of all individuals in a population

    Breeding values are additive over loci. The breeding values of noise are on
    transformed scales (log-additive model; see hanoi.comp_varcov).

    Attributes
    ----------
    mean : Breeding value of mean $A_m$.
           Should be a 2D np.ndarray of shape (n, N).
           mean[i, j] holds the breeding value of mean of trait i for individual j.
    ln_var : Breeding value of ln(variance) $A_{lnv}$.
           Should be a 2D np.ndarray of shape (n, N).
           ln_var[i, j] holds the breeding value of ln(variance) of trait i for individual j.
    artanh_cor : Breeding value of artanh(correlation) $A_c$.
           Should be a 2D np.ndarray of shape (choose(n, 2), N).
           artanh_cor[i, j] holds the breeding value of artanh(correlation) of the trait pair i for individual j.
    n : Number of traits
    N : Number of individuals


    Methods
    -------
    show() : Prints information of the BreedVal object.

    """
    def __init__(self, mean: np.ndarray, ln_var: np.ndarray, artanh_cor: np.ndarray):
        self.mean = mean
        self.ln_var = ln_var
        self.artanh_cor = artanh_cor
        self.N = mean.shape[1]
        self.n = mean.shape[0]
        for arr in (self.ln_var, self.artanh_cor):
            if arr.ndim != 2:
                raise ValueError("mean, ln_var, and artanh_cor must be 2D arrays.")
        Ns = [self.ln_var.shape[1], self.artanh_cor.shape[1]]
        if not all(N == self.N for N in Ns):
            raise ValueError("mean, ln_var, and artanh_cor must have the same number of columns (N)")
        if self.n != ln_var.shape[0]:
            raise ValueError("mean and ln_var must have the same number of rows (n)")
        if math.comb(self.n, 2) != self.artanh_cor.shape[0]:
            raise ValueError("The number of rows of artanh_cor should match the number of rows of mean choose 2")
    def show(self):
        print(f"Number of traits:\n {self.n}")
        print(f"Number of trait pairs:\n {math.comb(self.n, 2)}")
        print(f"Number of individuals:\n {self.N}")
        print(f"Breeding value of mean:\n {self.mean}")
        print(f"Breeding value of ln(variance):\n {self.ln_var}")
        print(f"Breeding value of artanh(correlation):\n {self.artanh_cor}")


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
    A_lnv = 1/2 * mut_effect.ln_var @ genotype
    A_c = 1/2 * mut_effect.artanh_cor @ genotype
    return BreedVal(mean = A_m, ln_var = A_lnv, artanh_cor = A_c)


def comp_varcov(breed_val: BreedVal, v_0, r_0=None):
    """
    Computes the variance-covariance matrix of phenotypic noise of each individual
    from breeding values, using the log-additive model.

    For trait i, variance v_i = v_0[i] * exp(ln_var[i]).
    For trait pair k, correlation r_k = tanh(artanh(r_0[k]) + artanh_cor[k]).
    The covariance of traits i and j is r_k * sqrt(v_i * v_j).

    Arguments:
        breed_val : hanoi.BreedVal object representing breeding value.
        v_0 :       1D array-like of n representing baseline variance of n traits for a genotype of 0000...
                    Should be positive.
        r_0 :       1D array-like of choose(n, 2) representing baseline correlation of trait pairs
                    for a genotype of 0000... Should be in (-1, 1). Not needed if n = 1.

    Returns: 3D np.ndarray of (N, n, n) representing variance-covariance matrices.
    """
    n, N = breed_val.n, breed_val.N
    v_0 = np.atleast_1d(np.asarray(v_0, dtype = np.float64))
    if v_0.shape != (n,):
        raise ValueError("v_0 should be a 1D array of n (number of traits)")
    if np.any(v_0 <= 0):
        raise ValueError("v_0 should be positive")
    v = v_0[:, None] * np.exp(breed_val.ln_var) # (n, N)
    varcov = np.zeros((N, n, n))
    idx = np.arange(n)
    varcov[:, idx, idx] = v.T
    if n == 1:
        return varcov
    if r_0 is None:
        raise ValueError("r_0 is required if n > 1")
    r_0 = np.atleast_1d(np.asarray(r_0, dtype = np.float64))
    if r_0.shape != (math.comb(n, 2),):
        raise ValueError("r_0 should be a 1D array of choose(n, 2) (number of trait pairs)")
    if np.any(np.abs(r_0) >= 1):
        raise ValueError("r_0 should be strictly between -1 and 1")
    r = np.tanh(np.arctanh(r_0)[:, None] + breed_val.artanh_cor) # (choose(n, 2), N)
    i, j = np.triu_indices(n, k = 1) # same order as scipy.spatial.distance.squareform
    cov = r.T * np.sqrt(v[i, :] * v[j, :]).T # (N, choose(n, 2))
    varcov[:, i, j] = cov
    varcov[:, j, i] = cov
    if n > 2 and np.any(np.linalg.eigvalsh(varcov) <= 0):
        raise ValueError("Some variance-covariance matrices are not positive definite. "
                         "Pairwise correlations can be valid while the whole matrix is not when n > 2.")
    return varcov


def cov_mtx(var, cov):
    """
    Computes 2D variance-covariance matrix from variance and covariance.

    Arguments:
        var: 1D array-like of n representing variance of n traits.
        cov: 1D array-like of choose(n, 2) representing covariance between n traits.

    Returns: (n, n) np.ndarray representing variance-covariance matrix.
    """
    return squareform(cov) + np.diag(var)


