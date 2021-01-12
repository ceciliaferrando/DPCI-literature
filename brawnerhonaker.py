# Brawner-Honaker method as implemented by Wu et al. (2020) in R
# https://github.com/wxindu/dp-conf-int/blob/master/algorithms/replications/alg0_Brawner_Honaker.R

import numpy as np
from scipy import stats
from scipy.special import comb

def compute_epsilon(rho, delta = 1e-6):
    rho + 2 * np.sqrt(rho * np.log((np.pi * rho) / delta))

def partition_bootstrap(data):
    N = len(data)
    draws = np.random.multinomial(N, [1/N]*N, size=1)
    partitions = []
    for i in range(8):
        partitions[i] = data[draws == (i-1)]
    return partitions

def define_p_i(i, N):
    return comb(N, i) * ((1/N)**i) * (1-1/N)**(N - i)

def define_rho_i(i, rho, N):
    return (i * rho) / define_p_i(i, N)

def define_sigma_i(sensitivity, i, rho, N):
    p_i = define_p_i(i, N)
    return i * p_i * (sensitivity ** 2 / (2 * rho))


def calculate_partition_mean(X_i, i, rho, range, N):
    sensitivity = range / N
    sigma_i = define_sigma_i(sensitivity, i, rho, N)
    m_i = i * np.sum(X_i) / N
    return m_i + np.random.normal(loc=0, scale=np.sqrt(sigma_i), size=1)

def bootstrap_priv_mean(data, rho, range):
    N = len(data)
    partitions = partition_bootstrap(data)
    M_i_vec = [calculate_partition_mean(partitions[i], i, rho, range, N) for i in range(8)]  #check this
    return np.sum(M_i_vec)

def estimate_var(k_boot_means, alpha_prime, N, rho, range):
    sensitivity = range / N
    k = len(k_boot_means)
    c_a_prime = stats.chi2.ppf(alpha_prime, k - 1)   #check this
    return np.var(k_boot_means) - (sensitivity ** 2) / (2 * rho) * ((k * c_a_prime) / (k - 1) - 1)

def epsilon_rho_equiv():
    return {epsilons: [.01, .1, .2, .25, .5, .75, 1, 2],
               rhos: [.000009, .0004, .0012, .0018, .0062, .013, .022, .075]}


def construct_boot_ci(data, k, epsilon, range, alpha, alpha_prime):

    rho = epsilon_rho_equiv[rhos][epsilon_rho_equiv[epsilons] == epsilon]
    N = length(data)
    boot_vec = [None] * k

    for i in range(1, k+1):
        boot_vec[i-1] = bootstrap_priv_mean(data, rho / k, range)

    mean_est = np.mean(boot_vec)
    var_est = np.max(0, estimate_var(boot_vec, alpha_prime, N, rho, range))
    se_est = np.sqrt(var_est)
    z = stats.norm.ppf(1 - alpha / 2)
    return [mean_est - z * se_est, mean_est + z * se_est]