import numpy as np
import argparse
from karwavadhan import *
from brawnerhonaker import *

parser = argparse.ArgumentParser(description='DP CI reference algorithms')
parser.add_argument('--alg', type=str, default='brawnerhonaker', help='karwavadhan or brawnerhonaker or dorazioetal')
parser.add_argument('--n', type=int, default=1000, help='number of data points')
parser.add_argument('--e', type=float, default=0.1, help='privacy parameter epsilon')
parser.add_argument('--r', type=float, default=32,
                    help="Upper bound of the data range. Range symmetric with respect to 0. Lowerbound set to minus this")
parser.add_argument('--a', type=float, default=0.05, help='significance level')
parser.add_argument('--iters', type=int, default=500, help='number of iterations for computing CI coverage')
parser.add_argument('--center', type=float, default=0.0, help='significance level')
args = parser.parse_args()

if __name__ == "__main__":

    alg = args.alg
    true_param = args.center
    n = args.n
    e = args.e
    r = args.r
    a = args.a
    iters = args.iters

    is_true_param_covered = [0]*iters
    moes = [0]*iters

    for iter in range(iters):

        db = np.random.normal(0, 1, n)

        if alg == 'karwavadhan':
            stdmin, stdmax = 0.2, 5 #see original code
            xmin, xmax = -r, r
            ci = karwa_vadhan_ci(db, a, e, stdmin, stdmax, xmin, xmax)
            if ci[0] <= true_param <= ci[1]:
                is_true_param_covered[iter] = 1
            moe = (ci[1]-ci[0])/2
            moes[iter] = moe
        elif alg == 'brawnerhonaker':
            stdmin, stdmax = 0.2, 5  # see original code
            xmin, xmax = -r, r
            ci = construct_boot_ci(db, 50, e, 2*r, a, .05)
            if ci[0] <= true_param <= ci[1]:
                is_true_param_covered[iter] = 1
            moe = (ci[1] - ci[0]) / 2
            moes[iter] = moe


    print("mean MoE", np.mean(moes))
    print("coverage", np.mean(is_true_param_covered))