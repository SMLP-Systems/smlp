#!/usr/bin/python3.14
# Reference: https://shgo.readthedocs.io/en/latest/docs/README.html#1-endres-sc--sandrock-c-focke-ww-2018-a-simplicial-homology-algorithm-for-lipschitz-optimisation-journal-of-global-optimization
from shgo import shgo
from numpy import sqrt
from hashlib import sha256

def f(x):  # (cattle-feed)
    return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]

def g1(x):
    return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0

def g2(x):
    return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21
            - 1.645 * sqrt(0.28*x[0]**2 + 0.19*x[1]**2
                         + 20.5*x[2]**2 + 0.62*x[3]**2)
            ) # >=0

def h1(x):
    return x[0] + x[1] + x[2] + x[3] - 1  # == 0

def main():
    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2},
            {'type': 'eq', 'fun': h1})
    bounds = [(0, 1.0),]*4
    res = shgo(f, bounds, iters=3, constraints=cons)
    pprint_res=f'x1={res.x[0]:.3f},x2={res.x[1]:.3f},x3={res.x[2]:.3f},x4={res.x[3]:.3f},f={res.fun:.3f}\n' + \
               f'g1={g1(res.x):.3f},g2={g2(res.x):.3f},g3={h1(res.x):.3f}'
    print(pprint_res)
    return sha256(pprint_res.encode()).hexdigest();

if __name__ == "__main__":
    print(main())
