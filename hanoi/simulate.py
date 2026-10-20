from .hanoi import *
from .fitness import *
import numpy as np
import math, os
#import jax.numpy as jnp
from scipy.spatial.distance import squareform
import concurrent.futures as futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def sim_generation(mut_effect, genotype, mut_rate, fit_func, mean_0, v_0, r_0=None, **kwargs):
    """
    Simulates one generation.

    Parameters
    ----------
    mut_effect : MutEffect
        Mutation effect.
    genotype : np.ndarray
        2D array of shape (L, N) representing genotype.
    mut_rate : float
        Mutation rate per locus per generation.
    fit_func : str
        Name of the fitness function.
        One of "fit_neutral", "fit_gaus", "fit_multimodal", "fit_step", "fit_sigmoid".
    mean_0 : np.ndarray
        1D array of length n representing expectation of n traits for a genotype of 0000...
    v_0 : np.ndarray
        1D array of length n representing baseline phenotypic variance (noise) of n traits
        for a genotype of 0000...
    r_0 : np.ndarray, optional
        1D array of length choose(n, 2) representing baseline correlation of noise between
        trait pairs for a genotype of 0000... Not needed if n = 1.
    **kwargs
        Extra arguments passed to the fitness function.
        For more details, see `fit_neutral`, `fit_gaus`, `fit_multimodal`, `fit_step` and `fit_sigmoid`.

    Returns
    -------
    Generation
        The simulated generation.
    """
    popsize = genotype.shape[1]
    
    # Breeding value
    A = comp_breed_val(mut_effect=mut_effect, genotype=genotype)
    
    # Phenotype
    z = sim_pheno(breed_val=A, mean_0=mean_0, v_0 = v_0, r_0 = r_0)
    
    # Phenotype to fitness
    if fit_func == "fit_gaus":
        w = fit_gaus(z=z, **kwargs)
    if fit_func == "fit_multimodal":
        w = fit_multimodal(z=z, **kwargs)
    if fit_func == "fit_neutral":
        w = fit_neutral(z=z)
    if fit_func == "fit_step":
        boxes = kwargs.get('boxes')
        w = fit_step(z=z, **kwargs)
    if fit_func == "fit_sigmoid":
        w = fit_sigmoid(z=z, **kwargs)
    
    # Germline mutation
    if mut_rate > 0:
        genotype_mut = sim_mutation(genotype=genotype, mut_rate=mut_rate) # This still returns nans if it is nans
    else:
        genotype_mut = genotype
    # Reproduction
    genotype_next, n_offspring = sim_reproduction(popsize, genotype_mut, w)
    #if n_offspring.sum() == 0:
    #    raise RuntimeError("No individuals survived")
    return Generation(genotype = genotype,
                      genotype_next = genotype_next,
                      breed_val = A, 
                      phenotype = z, 
                      fitness = w,
                      n_offspring = n_offspring
                     )


def sim_generations(n_gen, mut_effect, genotype, mut_rate, mean_0, v_0, fit_func, r_0 = None, seed = None, record = True, **kwargs):
    """
    Simulates multiple generations.

    Parameters
    ----------
    n_gen : int
        Number of generations to simulate.
        n_gen = 0 runs the simulation until all loci are fixed in the population.
    mut_effect : MutEffect
        Mutation effect.
    genotype : np.ndarray
        2D array of shape (L, N) representing the initial genotype table.
    mut_rate : float
        Mutation rate per locus per generation.
    mean_0 : np.ndarray
        1D array of length n representing expectation of n traits for a genotype of 0000...
    v_0 : np.ndarray
        1D array of length n representing baseline phenotypic variance (noise) of n traits
        for a genotype of 0000...
    fit_func : str
        Name of the fitness function.
        One of "fit_neutral", "fit_gaus", "fit_multimodal", "fit_step", "fit_sigmoid".
    r_0 : np.ndarray, optional
        1D array of length choose(n, 2) representing baseline correlation of noise between
        trait pairs for a genotype of 0000... Not needed if n = 1.
    seed : int, optional
        Seed of the random number generator.
    record : bool, default True
        Whether the intermediate generations are recorded.
    **kwargs
        Extra arguments passed to the fitness function.

    Returns
    -------
    Generations or list
        If `record` is True, a `Generations` object.
        If `record` is False, a list of 2: the first element is the number of generations
        simulated, and the second is the `Generation` object of the last generation.
    """
    if seed is not None:
        np.random.seed(seed)
    if record:
        generations_list = []
        genotype_cur = genotype
        if n_gen == 0:
            i = 0
            while not (np.all(np.all(genotype_cur == 0, axis = 1) | np.all(genotype_cur == 2, axis = 1)) or np.all(np.isnan(genotype_cur))):
            #while not (np.all(np.isin(genotype_cur, [0,2])) or np.all(np.isnan(genotype_cur))):
            #while np.any(genotype_cur == 1) or not np.isnan(genotype_cur[0,0]):
                generations_list.append(
                    sim_generation(mut_effect = mut_effect, 
                                   genotype = genotype_cur, 
                                   mut_rate = mut_rate, 
                                   fit_func = fit_func, 
                                   mean_0 = mean_0,
                                   v_0 = v_0,
                                   r_0 = r_0,
                                   **kwargs))
                genotype_cur = generations_list[-1].genotype_next
                i+=1
        else:
            for i in range(n_gen):
                if np.all(np.isnan(genotype_cur)):
                    break
                generations_list.append(
                    sim_generation(mut_effect = mut_effect, 
                                   genotype = genotype_cur, 
                                   mut_rate = mut_rate, 
                                   fit_func = fit_func, 
                                   mean_0 = mean_0,
                                   v_0 = v_0,
                                   r_0 = r_0,
                                   **kwargs))
                genotype_cur = generations_list[-1].genotype_next
        generations = Generations(genotype = np.array([generation.genotype for generation in generations_list]), 
                                  genotype_next = np.array([generation.genotype_next for generation in generations_list]), 
                                  breed_val = np.array([generation.breed_val for generation in generations_list]),
                                  phenotype = np.array([generation.phenotype for generation in generations_list]),
                                  fitness = np.array([generation.fitness for generation in generations_list]),
                                  n_offspring = np.array([generation.n_offspring for generation in generations_list]),
                                  n_gen = len(generations_list)
                                  )
        return generations
    else: # i.e. record is False
        genotype_cur = genotype
        if n_gen == 0:
            i = 0
            while not (np.all(np.all(genotype_cur == 0, axis = 1) | np.all(genotype_cur == 2, axis = 1))  or np.all(np.isnan(genotype_cur))) :
                gen = sim_generation(mut_effect = mut_effect, 
                               genotype = genotype_cur, 
                               mut_rate = mut_rate, 
                               fit_func = fit_func, 
                               mean_0 = mean_0,
                               v_0 = v_0,
                               r_0 = r_0,
                               **kwargs)
                genotype_cur = gen.genotype_next
                i+=1
        else:
            for i in range(n_gen):
                if np.all(np.isnan(genotype_cur)):
                    break
                gen = sim_generation(mut_effect = mut_effect, 
                                     genotype = genotype_cur, 
                                     mut_rate = mut_rate, 
                                     fit_func = fit_func, 
                                     mean_0 = mean_0,
                                     v_0 = v_0,
                                     r_0 = r_0,
                                     **kwargs)
                genotype_cur = gen.genotype_next
            i+=1 # I know this is ugly...
        return [i, gen]



def sim_mutation(genotype, mut_rate):
    """
    Simulates mutation.

    Parameters
    ----------
    genotype : np.ndarray
        2D array representing the genotype table before mutation.
    mut_rate : float
        Mutation rate per locus per generation.

    Returns
    -------
    np.ndarray
        2D array representing the genotype table after mutation.
    """
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
    genotype_mut = genotype.copy()
    for i in range(3):
        geno_muts = np.repeat([0,1,2], n_sites_muts[i])
        np.random.shuffle(geno_muts)
        genotype_mut[idx_sites_genotype[i]] = geno_muts
    # Return mutated genotype matrix
    return genotype_mut



def sim_pheno(breed_val: BreedVal, mean_0, v_0, r_0=None):
    """
    Simulates phenotype.

    Phenotypic noise follows a log-additive model:
    the variance of trait i of an individual is v_0[i] * exp(breed_val.ln_var[i]), and
    the correlation of noise of trait pair k is tanh(artanh(r_0[k]) + breed_val.artanh_cor[k]).

    Parameters
    ----------
    breed_val : BreedVal
        Breeding value.
    mean_0 : np.ndarray
        1D array of length n representing expectation of n traits for a genotype of 0000...
    v_0 : np.ndarray
        1D array of length n representing baseline variance (noise) of n traits
        for a genotype of 0000...
    r_0 : np.ndarray, optional
        1D array of length choose(n, 2) representing baseline correlation of noise between
        trait pairs for a genotype of 0000... Not needed if n = 1.

    Returns
    -------
    np.ndarray
        2D array of shape (N, n) representing phenotype.
        [i, j] holds the phenotype of individual i for trait j.
    """
    varcov = comp_varcov(breed_val=breed_val, v_0=v_0, r_0=r_0) # (N, n, n)
    if breed_val.n == 1:
        loc = mean_0 + breed_val.mean               # (1, N)
        sd = np.sqrt(varcov[:, 0, 0])               # (N,)
        z = np.random.normal(loc[0], sd)[:, None]   # (N, 1)
        return z
    z = np.array([np.random.multivariate_normal(mean_0 + breed_val.mean[:,i], varcov[i]) for i in range(breed_val.N)])
    return z




def sim_reproduction(popsize, genotype, fitness):
    """
    Simulates reproduction.

    Parameters
    ----------
    popsize : int
        Population size N.
    genotype : np.ndarray
        2D array of shape (L, N) representing genotype of parents.
    fitness : np.ndarray
        1D array of length N representing fitness of parents.

    Returns
    -------
    genotype_next : np.ndarray
        2D array of shape (L, N) representing genotype of the offspring.
    n_offspring : np.ndarray
        1D array of length N whose sum should be 2N.
        n_offspring[i] holds the number of offspring of parent individual i.
    """
    # number of offspring per genotype of each genotype
    n_loci = genotype.shape[0]
    if fitness.sum() == 0:
        genotype_next = np.full((n_loci, popsize), np.nan)
        n_offspring = np.zeros(popsize, dtype = int)
        #raise RuntimeError("Fitness of all individuals is 0")
        return genotype_next, n_offspring
    n_offspring = np.random.multinomial(n = 2 * popsize, pvals = fitness/fitness.sum())

    # index of 2N parents
    sam = np.repeat(range(popsize), n_offspring)
    # Shuffle the 2N parents
    np.random.shuffle(sam)
    # Reshape parents so that they are in pairs
    sam = sam.reshape(popsize, 2)
    # Mating
    genotype_next = np.random.binomial(1, genotype[:,sam]/2 ).sum(axis = 2)
    return genotype_next, n_offspring


def run_replicates(n_reps, max_workers = os.cpu_count(), seeds = None, **kwargs):
    """
    Runs `hanoi.sim_generations` many times in parallel processes.

    Parameters
    ----------
    n_reps : int
        Number of times to run `hanoi.sim_generations`.
    max_workers : int, default os.cpu_count()
        Number of CPUs to use for parallelisation.
    seeds : list of int, optional
        Seeds of the replicates. Its length must equal `n_reps`.
        If omitted, random seeds are generated.
    **kwargs
        Extra arguments passed to `hanoi.sim_generations`.

    Returns
    -------
    list
        A list of `n_reps` results of `hanoi.sim_generations`.
        With `record=True` (default), each is a `Generations` object.
        With `record=False`, each is a list of 2: the number of generations
        and the `Generation` object of the final generation.
    """
    results = [None] * n_reps

    # generate seeds if not provided
    if seeds is None:
        ss = np.random.SeedSequence()
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_reps)]
    elif len(seeds) != n_reps:
        raise ValueError("Length of seeds must equal n_reps")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Schedule all tasks
        futures = {
            executor.submit(sim_generations, seed=seed, **kwargs): i
            for i, seed in enumerate(seeds)
        }
        
        # Collect results as they finish
        for fut in as_completed(futures):
            idx = futures[fut]          # get the original index of this future
            results[idx] = fut.result() # store in the correct slot
    return results




def sim_genotype_coalescent(L, N, mu, seed=None):
    """
    Generates a genotype table based on coalescent.

    Parameters
    ----------
    L : int
        Number of loci.
    N : int
        Diploid population size.
    mu : float
        Mutation rate.
    seed : int or None, optional
        Random seed. If None (default), the result is not repeatable.

    Returns
    -------
    np.ndarray
        Array of shape (L, N) with the genotype of N individuals at L loci.

    """
    # random number generator; `rng` can be a seed (for repeatable results) or None
    rng = np.random.default_rng(seed)

    # theta = 4 N mu, as in coalescent theory
    theta = 4 * N * mu

    # prob[i] = probability that a locus has i copies of allele 1, for i = 0, 1, ..., 2N - 1
    #prob = np.empty(2 * N)

    # for i = 0, 1, ..., 2N - 1 the expected fraction of loci is theta / i
    prob = np.zeros(2*N)
    prob[1:] = theta / np.arange(1, 2 * N)
    # the sum of theta / i was too big, so the probability of class 0 became negative
    if prob.sum() > 1:
        raise ValueError("theta * sum(1/i) > 1: mu is too large")
    # fill monomorphic
    prob[0] = 1 - prob.sum()


    # distribute L loci over the classes; counts[i] = number of loci with i copies
    counts = rng.multinomial(L, prob)

    # one number per locus: how many copies of allele 1 it has
    # e.g. counts = [3, 2, 1] gives copies = [0, 0, 0, 1, 1, 2]
    copies = np.repeat(np.arange(2 * N), counts)
    copies = rng.permutation(copies)

    # empty genotype table: L rows (loci), N columns (individuals)
    genotype = np.zeros((L, N), dtype=int)

    for locus in range(L):
        i = copies[locus]                                    # copies of allele 1 at this locus
        chosen = rng.choice(2 * N, size=i, replace=False)    # which gene copies carry allele 1
        for c in chosen:
            genotype[locus, c // 2] += 1                            # add 1 to the individual who owns gene copy c
    return genotype
