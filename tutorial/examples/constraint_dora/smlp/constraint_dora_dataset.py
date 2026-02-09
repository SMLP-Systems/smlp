#!/usr/bin/python3.12
"""
Constrained Optimization Example using scipy.optimize
Classic Lagrange Multiplier Problem (appears in many optimization textbooks)

Problem:
    Minimize: f(x1, x2) = (x1 - 2)^2 + (x2 - 1)^2
    Subject to: x1^2 + x2^2 - 1 <= 0  (inside unit circle)
    
    Geometrically: Find the point on the unit circle closest to (2, 1)

Analytical Solution (using Lagrange multipliers):
    Setting ∇f = λ∇g where g(x1,x2) = x1² + x2² - 1:
    - 2(x1-2) = 2λx1  →  x1 = 2/(1+λ)
    - 2(x2-1) = 2λx2  →  x2 = 1/(1+λ)
    - Substituting into constraint: 5 = (1+λ)²
    - Solving: λ = √5 - 1
    
Expected Result:
    Optimal point: x1 = 2/√5 ≈ 0.894427, x2 = 1/√5 ≈ 0.447214
    Minimum value: f = (√5 - 1)² ≈ 1.527864
    
Reference: Wolfram Alpha
https://www.wolframalpha.com/input?i=Minimize%3A+f%28x1%2C+x2%29+%3D+%28x1+-+2%29%5E2+%2B+%28x2+-+1%29%5E2+subject+to+x1%5E2+%2B+x2%5E2+-+1+%3C%3D+0
"""

from numpy import linspace, meshgrid
from pandas import read_csv, concat
from gzip import open as gzopen

def main():
# Initial guess
    rng = range(0, 1000)
    x1_start, x1_stop = (-1.5, 2.5)
    x2_start, x2_stop = (-1.5, 2.0)
    x1 = linspace(x1_start, x1_stop, rng.stop)
    x2 = linspace(x2_start, x2_stop, rng.stop)
    X1, X2 = meshgrid(x1, x2)
    Z = (X1 - 2)**2 + (X2 - 1)**2
    with gzopen('dataset.txt.gz',"wt") as ds:
        ds.write("X1 X2 Y1\n")
        [[ds.write(f"{X1[i][j]} {X2[i][j]} {Z[i][j]}\n") for j in rng] for i in rng]
if __name__ == "__main__":
    main()
