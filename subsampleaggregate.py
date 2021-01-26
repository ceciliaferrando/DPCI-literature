import numpy as np
import matplotlib.pyplot as plt

def subsample(data, k):
    """
        k: number of groups to divide datasets into
    """
    np.random.shuffle(data)
    subsamples = np.split(data, k)
    Z = np.array([np.mean(group) for group in subsamples])
    return subsamples, Z


def WidenedWinsorize(Z, epsilon, xmin, xmax):
    """
    :param Z: vector of subsampled estimates
    :param epsilon: DP param
    :param xmin: lowerbound of data range
    :param xmax: upperbound of data range
    :return: winsorized mean of Z
    """
    k = len(Z)
    eta = 1 / 10  # as in Smith 2011
    rad = k ** (1 / 3 + eta)  # as in Smith 2011
    Z = np.sort(Z)

    # use algo from Du et al to release quantiles
    alpha_hat = PrivateQuantile(Z, 1 / 4, epsilon / 4, xmin, xmax)
    beta_hat = PrivateQuantile(Z, 3 / 4, epsilon / 4, xmin, xmax)

    mu_crude = (alpha_hat + beta_hat) / 2
    iqr_crude = np.abs(beta_hat - alpha_hat)

    # widen and clamp
    u = mu_crude + 4 * rad * iqr_crude
    l = mu_crude - 4 * rad * iqr_crude
    Z[Z < l] = l
    Z[Z > u] = u

    mu = np.mean(Z)
    Y = np.random.laplace(np.abs(u - l) / (2 * epsilon * k))

    #winsorized (DP) mu
    W_mu = mu + Y

    return W_mu

def Utility(m, i):
    """
    From Du et al 2020, Algorithm 5
    :param m: index of element corresponding to true quantile
    :param i: given index
    :return: utility
    """
    return i + 1 - m if i<m else m-1


def PrivateQuantile(Z, q, eps, xmin, xmax):
    """
    From Du et al 2020
    :param Z: vector of subsampled estimates
    :param q: quantile of interest (between 0 and 1)
    :param eps: DP param
    :param xmin: lowerbound of data range
    :param xmax: upperbound of data range
    :return: private quantile
    """

    N = len(Z)
    m = np.floor((N - 1) * q + 1.5)
    Z[Z < xmin] = xmin
    Z[Z > xmax] = xmax
    Z_dict = {i: Z[i - 1] for i in range(1, len(Z) + 1)}
    Z_dict[0] = np.min(Z)  # see Du et al, I found Smith confusing
    Z_dict[len(Z) + 1] = np.max(Z)  # see Du et al, I found Smith confusing

    ps = []
    prob_sum = 0

    for i in range(len(Z) + 1):
        # p_i = (Z_dict[i+1] - Z_dict[i]) * np.exp(eps * Utility(m, i))   #du
        p_i = (Z_dict[i + 1] - Z_dict[i]) * np.exp(-eps * np.abs(i - q * k))  # smith
        prob_sum += p_i
        ps.append(p_i)

    prob_vec = [p_i / prob_sum for p_i in ps]
    j = np.random.choice(a=list(range(len(Z) + 1)), size=1, p=prob_vec)[0]
    out = np.random.uniform(low=Z_dict[j], high=Z_dict[j + 1], size=1)

    return (out)

if __name__ == "__main__":

    # simple experiment with a Normal
    theta = 0
    stddev = 3
    # define hyperparams
    Lmda = 3  # how do we choose it for the simulations? not sure why Smith bounds between 0 and Lmda at some point
    epsilon = 1.0
    k = 50

    W_mu_vec = []

    for trial in range(5000):
        n = 10000
        data = np.random.normal(theta, stddev, n)
        subsamples, Z = subsample(data, k)
        W_mu = WidenedWinsorize(Z, epsilon, -Lmda, +Lmda)
        W_mu_vec.append(W_mu)

    print("average of winsorized means", np.mean(W_mu_vec))