from scipy import stats, linalg
import numpy as np
import pandas as pd
import os
from scipy.special import logsumexp, gammaln, logit, softmax
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def multinomial_rvs(counts, p, rng=None):
    """
    Simulate multinomial sampling of D dimensional probability distribution

    Arguments:
        counts: number of draws from distribution - int or array of the 
                ints (N)
        p: probability  - array of the floats (D, N)
        rng: np.random.default_rng() object, Optional
    Returns:
        Multinomial sample
    """

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(counts, (np.ndarray)):
        counts = np.full(p[0, ...].shape, counts)

    out = np.zeros(np.shape(p), dtype=int)
    ps = np.cumsum(p[::-1, ...], axis=0)[::-1, ...]
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0

    for i in range(p.shape[0]-1):
        binsample = rng.binomial(counts, condp[i, ...])
        out[i, ...] = binsample
        counts -= binsample

    out[-1, ...] = counts

    return out

def initialise_cancer(NSIM, rng=None):
    """
    Initialise a cancer, assigning fCpG states assuming fCpGs are homozygous 
    at t=0

    Arguments:
        tau: age when population began expanding exponentially - float
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        NSIM: number of fCpG loci to simulate - int
        rng: np.random.default_rng() object, Optional
    Returns:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
    """

    if rng is None:
        rng = np.random.default_rng()

    # assume fCpG's are homozygous (de)methylated at t=0
    mkw = np.zeros((3, NSIM), dtype = int)
    idx = np.arange(NSIM)
    np.random.shuffle(idx)
    mkw[0, idx[:NSIM//2]] = 1
    mkw[2, idx[NSIM//2:]] = 1

    return mkw

def diploid_evolve(mkw, mu, gamma, time_interval):
    # generate rate matrix for 2 alleles
    RateMatrix = np.array([[-2*gamma, mu, 0], 
                            [2*gamma, -(mu+gamma), 2*mu], 
                            [0, gamma, -2*mu]])
    
    # solve ODEs to calculate p(m, k, w)
    ProbStates = linalg.expm(RateMatrix * time_interval) @ mkw

    # randomly assign state according to probabilities
    m_cancer, k_cancer, w_cancer = multinomial_rvs(1, ProbStates, rng)

    return m_cancer, k_cancer, w_cancer

def trisomy_evolve(mudw, mu, gamma, time_interval):
    # generate rate matrix for 3 alleles
    RateMatrix = np.array([[-3*gamma, mu, 0, 0], 
                            [3*gamma, -(mu+2*gamma), 2*mu, 0], 
                            [0, 2*gamma, -(2*mu+gamma), 3*mu],
                            [0, 0, gamma, -3*mu]])
    
    ProbStates = linalg.expm(RateMatrix * time_interval) @ mudw

    m_cancer, u_cancer, d_cancer, w_cancer = multinomial_rvs(1, ProbStates, rng)

    return m_cancer, u_cancer, d_cancer, w_cancer


rng = np.random.default_rng()

NSIM = 10000

mu = 0.01
gamma = 0.01
tau = 50

# If we wanted to just plot the probability of the states depending on the
# different starting conditions
RateMatrix = np.array([[-2*gamma, mu, 0], 
                        [2*gamma, -(mu+gamma), 2*mu], 
                        [0, gamma, -2*mu]])

# solve ODEs to calculate p(m, k, w)
ProbStatesCond = linalg.expm(RateMatrix * tau)

positions_diploid = np.array([1, 0.5, 0])

title_dict = {0:"Homozygous methylated\nat t=0",
              1:"Heterozygous methylated\nat t=0",
              2:"Homozygous demethylated\nat t=0"}

fig, axes = plt.subplots(2, 2, sharey = True, sharex = True)
axes_flat = np.ravel(axes)
for i in range(len(axes_flat)):
    if i < np.shape(ProbStatesCond)[0]:
        axes_flat[i].bar(positions_diploid, ProbStatesCond[:, i], width = 0.1,
                     alpha = 0.4)
        axes_flat[i].set_title(title_dict[i])
    else:
        axes_flat[i].remove()
sns.despine()

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                 left=False, right=False)
plt.xlabel('Fraction methylated')
plt.ylabel('Probability')
plt.tight_layout()

# Now calculate NSIM diploid fCpGs and evolve them according to the above
m_cancer, k_cancer, w_cancer = diploid_evolve(initialise_cancer(NSIM, rng),
                                              mu, gamma, tau)

betaOut = (k_cancer + 2*m_cancer) / 2

fig, ax = plt.subplots()
plt.hist(betaOut, bins = np.linspace(0, 1, 31), alpha = 0.4)
plt.xlabel('Fraction methylated')
plt.ylabel('Probability')
sns.despine()
plt.tight_layout()

u_cancer = rng.binomial(n = k_cancer, p = 0.5)
d_cancer = k_cancer - u_cancer

mudw = np.row_stack([m_cancer, u_cancer, d_cancer, w_cancer])

betaTri = (d_cancer + 2 * u_cancer + 3 * m_cancer) / 3
plt.hist(betaTri, bins = np.linspace(0, 1, 31), alpha = 0.4)

time_interval = 50

m_cancer2, u_cancer2, d_cancer2, w_cancer2 = trisomy_evolve(mudw, mu, gamma, time_interval)

betaTri2 = (d_cancer2 + 2 * u_cancer2 + 3 * m_cancer2) / 3
plt.hist(betaTri2, bins = np.linspace(0, 1, 31), alpha = 0.4)

plt.legend(['Diploid', 'Immediately after Trisomy',
            'A long time after Trisomy'])