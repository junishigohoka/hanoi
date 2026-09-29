![logo/hanoi_2D.png]


# hanoi
A Python package for population genetic simulation with Heritable Additive Noise.

## What is HANOI for?

Classical quantitative genetics assumes that alleles shift the *mean* of a trait. HANOI (**H**eritable **A**dditive **NO**ise) additionally lets alleles change the trait's **variance** (and the **covariance** between traits). That is, the noise of the phenotype is itself heritable, so selection can act on how variable, or how correlated, traits are, e.g. bet-hedging, canalisation or developmental instability.

HANOI is a forward-in-time, Wright–Fisher-style simulator of a diploid population of size `N` with `L` biallelic loci and `n` traits:

1. **Breeding values.** Each allele 1 adds a mutation effect to the baseline mean, variance and covariance. Effects are additive across loci: `A = 1/2 * effect @ genotype`.
2. **Phenotype.** Each individual's phenotype is drawn from a (multivariate) normal distribution with mean `mean_0 + A_mean` and variance-covariance `varcov_0 + A_varcov`.
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

# Mutation effect of allele 1, each of shape (n, L); cov is (choose(n, 2), L)
M = hanoi.MutEffect(
    mean=np.array([[0.0, 0.0]]),
    var =np.array([[0.0, 0.5]]),   # locus 1 increases phenotypic variance
    cov =np.zeros((0, L)),         # no trait pairs when n = 1
)

# Genotype table of shape (L, N); entries are allele-1 counts (0, 1, 2)
G = np.zeros((L, N))
G[0, :50] = 1
G[1, :50] = 1

mean_0   = np.array([0.0])          # trait mean of genotype 00...0
varcov_0 = np.array([[0.1]])        # baseline variance-covariance matrix

sim = hanoi.sim_generations(
    n_gen=100,                      # 0 = run until all loci are fixed
    mut_effect=M,
    genotype=G,
    mut_rate=0.0,                   # per locus per generation
    mean_0=mean_0,
    varcov_0=varcov_0,
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
| `MutEffect.mean` / `.var` | `(n, L)` | effect of allele 1 at each locus on each trait |
| `MutEffect.cov` | `(choose(n, 2), L)` | effect on the covariance of each trait pair |
| `phenotype` | `(N, n)` | phenotype of individual `i` for trait `j` |
| `fitness`, `n_offspring` | `(N,)` | per-parent fitness and offspring number (sums to `2N`) |

Use `hanoi.cov_mtx(var, cov)` to build a variance-covariance matrix, e.g. `hanoi.cov_mtx([0, 1], [0])` for two traits.

## Main functions

- `hanoi.sim_generations(n_gen, ..., record=True)` simulates many generations. With `record=True` it returns a `Generations` object, with `record=False` a list `[n_gen, last_generation]`, which is lighter on memory. `n_gen=0` runs until every locus is fixed (or `n_gen_max` is reached).
- `hanoi.sim_generation(...)` simulates a single generation and returns a `Generation`.
- `hanoi.run_replicates(n_reps, max_workers=..., seeds=None, **kwargs)` runs `sim_generations` many times in parallel processes; `kwargs` are passed on to `sim_generations`.
- Building blocks: `comp_breed_val`, `sim_pheno`, `sim_mutation`, `sim_reproduction`.

```python
sims = hanoi.run_replicates(
    n_reps=100, n_gen=0,
    mut_effect=M, genotype=G, mut_rate=0.0,
    mean_0=mean_0, varcov_0=varcov_0,
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
