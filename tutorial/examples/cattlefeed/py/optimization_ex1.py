#!/usr/bin/python3.14
# Reference: https://shgo.readthedocs.io/en/latest/docs/README.html#1-endres-sc--sandrock-c-focke-ww-2018-a-simplicial-homology-algorithm-for-lipschitz-optimisation-journal-of-global-optimization
from shgo import shgo
from numpy import sqrt, dot, sum
from hashlib import sha256

def main():
    v=[[24.55,26.75,39,40.50],[2.3,5.6,11.1,1.3],
       [12,11.9,41.8,52.1],[0.28,0.19,20.5,0.62]]

    g1 = lambda x: dot(x,v[1]) - 5
    g2 = lambda x: dot(x,v[2]) - 1.645 * sqrt(dot([x**2 for x in x],v[3])) - 21
    h1 = lambda x: sum(x) - 1

    cons = ({'type': 'ineq', 'fun': g1},
            {'type': 'ineq', 'fun': g2},
            {'type': 'eq',   'fun': h1})
    #shgo(f, bounds, iters, constraints)
    res = shgo(lambda x: dot(x,v[0]), [(0, 1.0),]*4, iters=3, constraints=cons)
    pprint_res=f'x1={res.x[0]:.3f},x2={res.x[1]:.3f},x3={res.x[2]:.3f},x4={res.x[3]:.3f},f={res.fun:.3f}\n' + \
               f'g1={g1(res.x):.3f},g2={g2(res.x):.3f},g3={h1(res.x):.3f}'
    print(pprint_res)
    return sha256(pprint_res.encode()).hexdigest();

if __name__ == "__main__":
    print(main())
