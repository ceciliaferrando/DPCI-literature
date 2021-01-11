# DPCI-literature
Reproducing algorithms for differentially private confidence intervals in the literature

How to run an experiment for DP CI of a standard normal:

run "run.py" from cmd line with the following arguments:

'--alg', type=str, default='vadhan', help='vadhan or brawner or dorazio or pb'

'--n', type=int, default=10000, help='number of data points'

'--e', type=float, default=0.1, help='privacy parameter epsilon'

'--r', type=float, default=32, help="Upper bound of the data range. Range symmetric with respect to 0. Lowerbound set to minus this"

'--a', type=float, default=0.05, help='significance level'

'--iters', type=int, default=1000, help='number of iterations for computing CI coverage'

'--center', type=float, default=0.0, help='significance level'
