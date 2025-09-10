from .hanoi import *
from .fitness import *
import numpy as np
import math
#import jax.numpy as jnp
from scipy.spatial.distance import squareform


def sim_generation(mut_effect, genotype, mut_rate, fit_func, mean_0, varcov_0, **kwargs):
    popsize = genotype.shape[1]
    
    # Breeding value
    A = comp_breed_val(mut_effect=mut_effect, genotype=genotype)
    
    # Phenotype
    z = sim_pheno(breed_val=A, mean_0=mean_0, varcov_0 = varcov_0)
    
    # Phenotype to fitness
    if fit_func == "fit_gaus":
        w = fit_gaus(z=z, **kwargs)
    if fit_func == "fit_multimodal":
        w = fit_multimodal(sigma=sigma, p=kwargs.get('p'), z_opt=z_opt, z=z)
    if fit_func == "fit_neutral":
        w = fit_neutral(z=z)
    if fit_func == "fit_step":
        boxes = kwargs.get('boxes')
        w = fit_step(z=z, boxes=boxes)
    
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


def sim_generations(n_gen, mut_effect, genotype, mut_rate, mean_0, varcov_0, fit_func, **kwargs):
    generations_list = []
    genotype_cur = genotype
    if n_gen == 0:
        i = 0
        while not (np.all(np.all(genotype_cur == 0, axis = 1) | np.all(genotype_cur == 2, axis = 1) )):
        #while not (np.all(np.isin(genotype_cur, [0,2])) or np.all(np.isnan(genotype_cur))):
        #while np.any(genotype_cur == 1) or not np.isnan(genotype_cur[0,0]):
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
                              n_gen = i + 1
                              )
    return generations




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
    genotype_mut = genotype.copy()
    for i in range(3):
        geno_muts = np.repeat([0,1,2], n_sites_muts[i])
        np.random.shuffle(geno_muts)
        genotype_mut[idx_sites_genotype[i]] = geno_muts
    # Return mutated genotype matrix
    return genotype_mut



def sim_pheno(breed_val: BreedVal, mean_0, varcov_0):
    if breed_val.n == 1:
        z = np.random.normal(mean_0 + breed_val.mean, varcov_0 + breed_val.var).T
        return z
    ## Cholesky decomposition of the covariance matrices
    #chol = jnp.linalg.cholesky(varcov_0 + breed_val.varcov)
    ### Standard normal sampling
    #z_std = np.random.standard_normal((breed_val.N, breed_val.n))
    ## Convert N vectors of n standard normal variables to the N phenotype values in n dimensions
    #z = np.einsum('ijk,ik->ij', chol, z_std) + mean_0
    z = np.array([np.random.multivariate_normal(mean_0 + breed_val.mean[:,i], varcov_0 + breed_val.varcov[i]) for i in range(breed_val.N)])
    return z




def sim_reproduction(popsize, genotype, fitness):
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



