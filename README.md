![](logo/hanoi_2D.png)


# hanoi
A Python package for population genetic simulation with Heritable Additive Noise.

## What is HANOI for?

Classical quantitative genetics assumes that alleles shift the *mean* of a trait. HANOI (**H**eritable **A**dditive **NO**ise) additionally lets alleles change the trait's **variance** (and the **correlation** of noise between traits). That is, the noise of the phenotype is itself heritable, so selection can act on how variable, or how correlated, traits are, e.g. bet-hedging, canalisation or developmental instability.

HANOI is a forward-in-time, Wright–Fisher-style simulator of a diploid population of size `N` with `L` biallelic loci and `n` traits:

1. **Breeding values.** Each allele 1 adds a mutation effect to the breeding value of the mean, of `ln(variance)` and of `artanh(correlation)`. Effects are additive across loci: `A = 1/2 * effect @ genotype`.
2. **Phenotype.** Each individual's phenotype is drawn from a (multivariate) normal distribution with mean `mean_0 + A_mean`. The noise is *log-additive*: the variance of each trait is `v_0 * exp(A_ln_var)` (always positive), and the correlation of each trait pair is `tanh(artanh(r_0) + A_artanh_cor)` (always between -1 and 1). The covariance of two traits is `r * sqrt(v_i * v_j)`.
3. **Fitness.** A fitness function maps phenotype to fitness.
4. **Mutation and reproduction.** Germline mutation (0 ↔ 1 ↔ 2 allele counts), then 2N offspring slots are drawn multinomially in proportion to fitness and paired at random.

Typical uses are tracking allele frequencies, fixation times and fixation outcomes of variance-modifying alleles under different selection regimes.

## Installation

```bash
pip install hanoi
```

Requires Python with `numpy>=2.0` and `scipy>=1.15`.

## Quick start

```python
import numpy as np
import hanoi

L, N, n = 2, 100, 1          # loci, individuals, traits

# Mutation effect of allele 1, each of shape (n, L); artanh_cor is (choose(n, 2), L)
M = hanoi.MutEffect(
    mean      =np.array([[0.0, 0.0]]),
    ln_var    =np.array([[0.0, 0.5]]),   # locus 2 multiplies the variance by exp(0.5)
    artanh_cor=np.zeros((0, L)),         # no trait pairs when n = 1
)

# Genotype table of shape (L, N); entries are allele-1 counts (0, 1, 2)
G = np.zeros((L, N))
G[0, :50] = 1
G[1, :50] = 1

mean_0 = np.array([0.0])            # trait mean of genotype 00...0
v_0    = np.array([0.1])            # baseline variance (noise) of each trait
r_0    = None                       # baseline correlation of each trait pair, (choose(n, 2),); not needed when n = 1

sim = hanoi.sim_generations(
    n_gen=100,                      # 0 = run until all loci are fixed
    mut_effect=M,
    genotype=G,
    mut_rate=0.0,                   # per locus per generation
    mean_0=mean_0,
    v_0=v_0,
    r_0=r_0,
    fit_func="fit_gaus",            # see the list below
    z_opt=np.array([0.0]),          # extra arguments go to the fitness function
    sigma=1.0,
    seed=1,
)

sim.n_gen
sim.allele_freqs()                  # (n_gen, L) allele frequencies over time
sim.phenotype.shape                 # (n_gen, N, n)
sim.generation(0).fitness           # inspect a single generation
```

Small ready-made inputs are available from `hanoi.sample_data_1()` (1 locus, 1 trait) and `hanoi.sample_data_2()` (5 loci, 2 traits). They return an object with `mut_effect`, `geno`, `z_ref` and `c_ref`.

### Conventions

| Object | Shape | Meaning |
|---|---|---|
| `genotype` | `(L, N)` | allele-1 count (0/1/2) of individual `j` at locus `i` |
| `MutEffect.mean` / `.ln_var` | `(n, L)` | effect of allele 1 at each locus on the mean / on `ln(variance)` of each trait |
| `MutEffect.artanh_cor` | `(choose(n, 2), L)` | effect on `artanh(correlation)` of each trait pair, in `scipy.spatial.distance.squareform` order |
| `v_0` | `(n,)` | baseline variance of each trait; must be positive |
| `r_0` | `(choose(n, 2),)` | baseline correlation of each trait pair; must be strictly between -1 and 1 |
| `phenotype` | `(N, n)` | phenotype of individual `i` for trait `j` |
| `fitness`, `n_offspring` | `(N,)` | per-parent fitness and offspring number (sums to `2N`) |

Use `hanoi.comp_varcov(breed_val, v_0, r_0)` to get the `(N, n, n)` variance-covariance matrices of phenotypic noise. With three or more traits, valid pairwise correlations do not guarantee a valid matrix, so an error is raised if any matrix is not positive definite.

## Main functions

- `hanoi.sim_generations(n_gen, ..., record=True)` simulates many generations. With `record=True` it returns a `Generations` object, with `record=False` a list `[n_gen, last_generation]`, which is lighter on memory. `n_gen=0` runs until every locus is fixed (or `n_gen_max` is reached).
- `hanoi.sim_generation(...)` simulates a single generation and returns a `Generation`.
- `hanoi.run_replicates(n_reps, max_workers=..., seeds=None, **kwargs)` runs `sim_generations` many times in parallel processes; `kwargs` are passed on to `sim_generations`.
- Building blocks: `comp_breed_val`, `sim_pheno`, `sim_mutation`, `sim_reproduction`.

```python
sims = hanoi.run_replicates(
    n_reps=100, n_gen=0,
    mut_effect=M, genotype=G, mut_rate=0.0,
    mean_0=mean_0, v_0=v_0, r_0=r_0,
    fit_func="fit_neutral", record=False,
)
fixation_times = [s[0] for s in sims]
```

## Fitness functions

Choose one with `fit_func=`; its own arguments are passed as extra keyword arguments.

| `fit_func` | Arguments | Description |
|---|---|---|
| `"fit_neutral"` | – | fitness is 1 for everyone |
| `"fit_gaus"` | `sigma`, `z_opt` | Gaussian peak at `z_opt` |
| `"fit_multimodal"` | `sigma`, `p`, `z_opt` | weighted sum of Gaussian peaks |
| `"fit_step"` | `boxes` | fitness 1 inside any of the `(m, 2, n)` boxes, else 0 |
| `"fit_sigmoid"` | `a`, `b` | logistic slope with centre `a` and steepness/direction `b` |
| `"fit_gaus_circular"` | `sigma`, `z_opt` | Gaussian on a periodic trait (needs `circular=True, period=...`) |
| `"fit_step_circular"` | `arcs` | step function on a periodic trait (needs `circular=True, period=...`) |

### Circular traits

For periodic traits (time of day, angles), pass `circular=True` and `period` (e.g. `24`, or `2*np.pi`). Phenotypes are then wrapped into `[-period/2, period/2)`.

## Note on the return of fully lethal generations

If every individual has fitness 0, the next generation's genotype table is filled with `NaN`, and `sim_generations` stops early.

## License

MIT
