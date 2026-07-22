from .hanoi import *
from .fitness import *
import numpy as np
import math, os
#import jax.numpy as jnp
from scipy.spatial.distance import squareform
import concurrent.futures as futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def sim_generation(mut_effect, genotype, mut_rate, fit_func, mean_0, varcov_0, circular=False, period=None, **kwargs):
    """
    Simulates one generation.

    Arguments:
        mut_effect : hanoi.MutEffect object representing mutation effect.
        genotype :   2D np.ndarray of (L, N) representing genotype.
        mut_rate :   float representing mutation rate per locus per generation.
        fit_func :   Fitness function.
                     "fit_neutral", "fit_gaus", "fit_multimodal", "fit_step", "fit_sigmoid",
                     "fit_gaus_circular", "fit_step_circular"
        mean_0 :     2D np.ndarray of (n, N) representing expectation of n traits for a genotype of 0000...
        varcov_0 :   2D np.ndarray of (n, n) representing baseline variance-covariance matrix for a genotype of 0000....
        circular :   bool. If True, phenotype is wrapped into [-period/2, period/2) to represent
                     a circular (periodic) trait. See hanoi.sim_pheno.
        period :    scalar or (n,) array. Required if circular=True. See hanoi.sim_pheno.
        **kwargs :   Extra arguments passed to fitness function. For more details, see help of fit_neutral, fit_gaus, fit_multimodal, fit_step, fit_sigmoid, fit_gaus_circular, fit_step_circular.
    Returns: hanoi.Generation object representing simulated generation.
    """
    popsize = genotype.shape[1]

    # Breeding value
    A = comp_breed_val(mut_effect=mut_effect, genotype=genotype)

    # Phenotype
    z = sim_pheno(breed_val=A, mean_0=mean_0, varcov_0 = varcov_0, circular=circular, period=period)

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
    if fit_func == "fit_gaus_circular":
        w = fit_gaus_circular(z=z, period=period, **kwargs)
    if fit_func == "fit_step_circular":
        w = fit_step_circular(z=z, period=period, **kwargs)

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


def sim_generations(n_gen, mut_effect, genotype, mut_rate, mean_0, varcov_0, fit_func, n_gen_max = np.inf, seed = None, record = True, **kwargs):
    """
    Simulates multiple generations.

    Arguments:
        n_gen :      Number of generations to simulate. 
                     n_gen = 0 runs simulation until mutations at all loci are fixed in the population.
        n_gen_max:   Maximum number of generations to simulate.
                     If n_gen > 0, n_gen_max is set to n_gen.
                     If n_gen == 0, simulation will stop if mutations do not fix before n_gen_max.
        mut_effect : hanoi.MutEffect representing the mutation effect.
        genotype :   2D np.ndarray of (L, N) representing the initial genotype table.
        mut_rate :   A float representing mutation rate per locus per generation.
        mean_0 :     2D np.ndarray of (n, N) representing expectation of n traits for a genotype of 0000...
        varcov_0 :   2D np.ndarray of (n, n) representing baseline variance-covariance matrix for a genotype of 0000....
        fit_func :   str representing the fitness function.
                     Should be one of "fit_neutral", "fit_gaus", "fit_multimodal", "fit_step", "fit_sigmoid",
                     "fit_gaus_circular", "fit_step_circular".
        seed :       int used as seed.
        record :     bool representing whether the intermediate generations are recorded.
        **kwargs :   Extra arguments passed to hanoi.sim_generation, including circular/period and
                     the fitness function's own arguments.
    Returns:
        If record is True, a hanoi.Generations object is returned.
        If record is False, a list of 2 is returned:
        The first element is n_gen, and the second is hanoi.Generation object representing the last generation.
    """
    if seed is not None:
        np.random.seed(seed)
    if n_gen > 0:
        n_gen_max = n_gen
    if record:
        generations_list = []
        genotype_cur = genotype
        if n_gen == 0:
            i = 0
            while not (np.all(np.all(genotype_cur == 0, axis = 1) | np.all(genotype_cur == 2, axis = 1)) or np.all(np.isnan(genotype_cur)) or i == n_gen_max):
                generations_list.append(
                    sim_generation(mut_effect = mut_effect, 
                                   genotype = genotype_cur, 
                                   mut_rate = mut_rate, 
                                   fit_func = fit_func, 
                                   mean_0 = mean_0,
                                   varcov_0 = varcov_0,
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
                                   varcov_0 = varcov_0,
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
            while not (np.all(np.all(genotype_cur == 0, axis = 1) | np.all(genotype_cur == 2, axis = 1))  or np.all(np.isnan(genotype_cur)) or i == n_gen_max) :
                gen = sim_generation(mut_effect = mut_effect, 
                               genotype = genotype_cur, 
                               mut_rate = mut_rate, 
                               fit_func = fit_func, 
                               mean_0 = mean_0,
                               varcov_0 = varcov_0,
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
                                     varcov_0 = varcov_0,
                                     **kwargs)
                genotype_cur = gen.genotype_next
            i+=1 # I know this is ugly...
        return [i, gen]



def sim_mutation(genotype, mut_rate):
    """
    Simulates mutation.

    Arguments :
        genotype : 2D np.ndarray representing genotye table before mutation.
        mut_rate : float representing mutation rate per locus per generation.

    Returns: 2D np.ndarray representing genotype table after mutation.
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



def sim_pheno(breed_val: BreedVal, mean_0, varcov_0, circular=False, period=None):
    """
    Simulates phenotype.

    Arguments:
        breed_val : hanoi.BreedVal object representing breeding value.
        mean_0 :    2D np.ndarray of (n, N) representing expectation of n traits for a genotype of 0000...
        varcov_0 :  2D np.ndarray of (n, n) representing baseline variance-covariance matrix for a genotype of 0000....
        circular :  bool. If True, phenotype is wrapped into [-period/2, period/2) after sampling,
                    to represent a circular (periodic) trait (e.g. time of day, angle).
        period :    scalar or (n,) array. Required if circular=True.
                    Specifies the full cycle length of the phenotype
                    (e.g. 24 for time-of-day in hours, 2*pi for radians).
                    Ignored if circular=False.

    Returns: 2D np.ndarray of (N, n) representing phenotype.
             [i, j] represents the phenotype of individual i for trait j.
             If circular=True, values are wrapped into [-period/2, period/2);
             e.g. with period=24, a raw value of 25 (1am the next day) is
             wrapped to 1.
    """
    if breed_val.n == 1:
        z = np.random.normal(mean_0 + breed_val.mean, np.sqrt(varcov_0 + breed_val.var)).T
    else:
        z = np.array([np.random.multivariate_normal(mean_0 + breed_val.mean[:,i], varcov_0 + breed_val.varcov[i]) for i in range(breed_val.N)])
    if circular:
        z = wrap_pheno(z, period)
    return z


def sim_reproduction(popsize, genotype, fitness):
    """
    Simulates reproduction.

    Arguments:
        popsize :  int N representing population size
        genotype : 2D np.ndarray of (L, N) representing genotype of parents.
        fitness :  1D np.ndarray of N representing fitness of parents.

    Returns: A tuple (genotype_next, n_offspring)
        genotype_next : 2D np.ndarray of (L, N) representing genotype of the offspring
        n_offspring :  Number of offspring.
                        Should be a 1D np.ndarray of length N and the sum should be 2N.
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
    Run `hanoi.sim_generations` many times.

    Parameters
    ----------
    n_reps :       int
                   Number of times to run `hanoi.sim_generations`
    max_workers :  int
                   Number of CPUs to use for parallelisation. <os.cpu_count()>
    seeds :        int
    **kwargs :     dict
                   Extra arguments passed to `hanoi.sim_generations`.


    Returns
    -------
    A list of n_reps. 
    With `record = True` (default), it returns a list of n_reps objects of type `hanoi.Generation`.
    With `record = False, it returns a list of n_reps list of 2 (i.e. shape (n_reps, 2)). The first element of each list of two is the number of generations, and the second element is a 2D np.ndarray representing the genotype table of the final generation.
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

