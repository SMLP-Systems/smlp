#!/usr/bin/python3.12
from transform_variables import dtlz4_objectives_t
from mpmath import mpf, mp, nan
mp.dps = 500
dim = 3
N = 101
p = []
[p.append(mpf(i)/mpf(100)) for i in range(0,N)]
print("X0,X1,X2,F1,F2")
for i in range(0,N):
    for j in range(0,N):
        for k in range(0,N):
            y = [p[i],p[j],p[k]]
            print(f"{float(y[0]):.16g},{float(y[1]):.16g},{float(y[2]):.16g},",end="")
            objectives_t = dtlz4_objectives_t(y)
            print(f"{float(objectives_t[0]):.16g},{float(objectives_t[1]):.16g}")
