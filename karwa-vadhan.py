# Karwa Vadhan as implemented by Wu et al. (2020) in R
# https://github.com/wxindu/dp-conf-int/blob/master/algorithms/replications/alg1_Vadhan.R

import numpy as np

def maxi(v):
    l = 1
    for i in range(2,len(v)):
        if v[l] < v[i]:
            l = i
    return l

def variance(x, m):
    return (1/(len(x) - 1)) * np.sum((x-m)**2)

def pub_interval(db, a):
    m = np.mean(db)
    radius = np.sqrt(variance(db, m)/len(db)) * stats.norm.ppf(1-a/2)
    return [m-radius, M+radius]

def pub_range(db):
    return [np.min(db), np.max(db)]

def pub_histogram_learner(db, bins):
    # outputs a normalized histogram of db separated by the intervals in bins
    db = sorted(db)
    probs = np.zeros((len(bins) - 1, ))
    db_i = 1

    while db[db_i] < bins[1]:
        db_i += 1
        if db_i > len(db):
            return probs/np.sum(probs)

    for i in range(1, len(probs)):
        while db[db_i] < bins[i + 1]:
            probs[i] += 1
            db_i += 1
            if db_i > len(db):
                return probs / np.sum(probs)

    return probs/np.sum(probs)

def priv_histogram_learner(db, bins, e):
    probs = pub_histogram_learner(db, bins)
    return probs + np.random.laplace(0, 2 / e / len(db), len(probs))

def priv_std(db, a, e, stdmin, stdmax):
    bins_base = np.floor(np.log2(stdmin) - 2)
    bins = 2**np.array((list(range(bins_base, np.ceil(np.log2(stdmax) + 2)))))   #check
    y = np.array(list(range(1,np.floor(len(db)/2))))
    for i in range(1,len(y)):
        y[i] = np.abs(db[2 * i] - db[2 * i - 1])

    l = maxi(priv_histogram_learner(y, bins, e))
    return 2 ** ((l + bins_base - 1) + 2)

def priv_mean(db, e, std, r):
    rn = np.ceil(r / std)
    bins_base = -rn
    bins = np.array((list(range(bins_base,(rn + 1))))) - .5) * std
    l = maxi(priv_histogram_learner(db, bins, e))
    return ((l + bins_base - 1) * std)


def priv_range(db, a1, a2, e1, e2, stdmin, stdmax, r):
    priv_std_ = priv_std(db, a1, e1, stdmin, stdmax)
    priv_mean_ = priv_mean(db, e2, priv_std, r)
    radius = 4 * priv_std_ * np.sqrt(np.log(len(db) / a2))
    return [priv_mean_ - radius, priv_mean_ + radius]


def priv_karwa_vadhan(db, a0, a1, a2, a3, e1, e2, e3, stdmin, stdmax, r):
    n = len(db)
    # priv_range is the most significant source of error
    xrange = priv_range(db, a3 / 2, a3 / 2, e3 / 2, e3 / 2, stdmin, stdmax, r)
    xmin = xrange[1]
    xmax = xrange[2]
    xdist = xmax - xmin

    # clamp
    db[db < xmin] = xmin
    db[db > xmax] = xmax

    mean_var = xdist / (e1 * n)
    priv_mean_ = np.mean(db) + np.random.laplace(0, mean_var, 1)
    if (priv_mean_ < xmin):
        priv_mean_ = xmin
    elif (priv_mean_ > xmax):
        priv_mean_ = xmax

    var_var = xdist ** 2 / (e2 * (n - 1))
    # priv_var <- public_var + extra_var + lap_noise
    priv_var = variance(db, priv_mean) + var_var * np.log(1 / a2) + np.random.laplace(0, var_var, 1)
    if (priv_var < 0 or priv_var > stdmax):
        priv_var = stdmax

    # mean_var*log(1/a1) is the second most significant source of error
    priv_radius = np.sqrt(priv_var / n) * stats.t.ppf(1 - a0 / 2, n - 1) + mean_var * np.log(1 / a1)

    return [priv_mean - priv_radius, priv_mean + priv_radius]


def priv_karwa_vadhan_ci(db, a, e, stdmin, stdmax, xmin, xmax):
    return (priv_vadhan(db, a / 4, a / 4, a / 4, a / 4, e / 3, e / 3, e / 3, stdmin, stdmax, max(abs(xmax), abs(xmin))))




