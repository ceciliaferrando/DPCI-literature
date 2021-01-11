import numpy as np
import argparse
from karwavadhan import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DP CI reference algorithms')
    parser.add_argument('alg', type=str, help='vadhan or brawner or dorazio')
    parser.add_argument('n', type=int, help='number of data points')
    parser.add_argument('e', type=float, help='privacy parameter epsilon')
    parser.add_argument('e', type=float, help='privacy parameter epsilon')

    true_param = 0
    n_experiments = 500
    is_true_param_covered = [0]*n_experiments
    moes = [0]*n_experiments

    for experiment in range(n_experiments):

        n = 10000
        #db = np.random.normal(0, 1, n)
        #np.save("db.csv", db)
        #db = np.array([0, 0.2, 0.1, 0.05, -0.05, -1])
        db = np.loadtxt("db.txt")
        a = 0.05    #significance level
        e = 0.1
        stdmin, stdmax = 0.2, 5 #see original code
        r = 32                  #check how they do it in the paper
        xmin, xmax = -r, r
        ci = karwa_vadhan_ci(db, a, e, stdmin, stdmax, xmin, xmax)
        if ci[0] <= true_param <= ci[1]:
            is_true_param_covered[experiment] = 1
        moe = (ci[1]-ci[0])/2
        moes[experiment] = moe

    print("mean MoE", np.mean(moes))
    print("coverage", np.mean(is_true_param_covered))