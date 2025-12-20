#!/usr/bin/python3.12
"""
C3-DTLZ4 Objective Functions
Based on the DTLZ test suite with DTLZ4 objectives and C3 constraints.

Reference:
- Deb et al. (2002) - Scalable Test Problems for Evolutionary Multiobjective Optimization
- Jain & Deb (2014) - An Evolutionary Many-Objective Optimization Algorithm (for constraints)
"""

from math import pi, sin, cos
from pandas import read_csv
from sys import argv

def dtlz4_objectives(x, n_objectives=2, alpha=100.0):
    """
    DTLZ4 objective functions.
    
    DTLZ4 is similar to DTLZ2 but uses a parameter alpha to control
    the density of solutions. Higher alpha creates a more biased density.
    
    Args:
        x: Decision variables [x0, x1, ..., x_{n-1}], all in [0, 1]
        n_objectives: Number of objectives (m)
        alpha: Density parameter (default: 100.0)
    
    Returns:
        List of objective values [f1, f2, ..., fm]
    
    Formula:
        g(x_M) = sum((x_i - 0.5)^2 for i in x_M)
        
        For m objectives:
        f_1 = (1 + g) * cos(x_1^alpha * π/2) * cos(x_2^alpha * π/2) * ... * cos(x_{m-1}^alpha * π/2)
        f_2 = (1 + g) * cos(x_1^alpha * π/2) * cos(x_2^alpha * π/2) * ... * sin(x_{m-1}^alpha * π/2)
        ...
        f_{m-1} = (1 + g) * cos(x_1^alpha * π/2) * sin(x_2^alpha * π/2)
        f_m = (1 + g) * sin(x_1^alpha * π/2)
        
    where x_M are the last k = n - m + 1 variables used for the g function.
    """
    n_vars = len(x)
    k = n_vars - n_objectives + 1  # Number of variables in g function
    
    # Calculate g function using the last k variables
    x_M = x[n_objectives - 1:]  # Last k variables
    g = sum((xi - 0.5) ** 2 for xi in x_M)
    
    # Calculate objectives
    objectives = []
    
    for i in range(n_objectives):
        f = 1.0 + g
        
        # Multiply by cosine terms
        for j in range(n_objectives - i - 1):
            f *= cos(x[j] ** alpha * pi / 2.0)
        
        # Multiply by sine term (if not the first objective)
        if i > 0:
            f *= sin(x[n_objectives - i - 1] ** alpha * pi / 2.0)
        
        objectives.append(f)
    
    return objectives


def dtlz4_2obj(x, alpha=100.0):
    """
    DTLZ4 for exactly 2 objectives.
    
    Args:
        x: Decision variables [x0, x1, ..., x_{n-1}], all in [0, 1]
           Typically n = 11 for 2-objective DTLZ4
        alpha: Density parameter (default: 100.0)
    
    Returns:
        Tuple of (f1, f2)
    
    Formula:
        g = sum((x_i - 0.5)^2 for i = 1 to n-1)
        f1 = (1 + g) * cos(x_0^alpha * π/2)
        f2 = (1 + g) * sin(x_0^alpha * π/2)
    """
    n_vars = len(x)
    
    # g function uses all variables except x[0]
    g = sum((x[i] - 0.5) ** 2 for i in range(1, n_vars))
    
    # Calculate objectives
    f1 = (1.0 + g) * cos(x[0] ** alpha * pi / 2.0)
    f2 = (1.0 + g) * sin(x[0] ** alpha * pi / 2.0)
    
    return f1, f2

def c3dtlz4_constraints(objectives):
    """
    C3-DTLZ4 constraint functions.
    
    For each objective f_i:
        c_i = -(sum of all f_j²) + 0.75 * f_i + 1.0
    
    Args:
        objectives: List of objective values [f1, f2, ..., fm]
    
    Returns:
        List of constraint values [c1, c2, ..., cm]
        Feasible if ALL constraints ≤ 0
    """
    sum_squares = sum(f**2 for f in objectives)
    constraints = [-sum_squares + f * 0.75 + 1.0 for f in objectives]
    return constraints


def evaluate_c3dtlz4(x, n_objectives=2, alpha=100.0):
    """
    Complete evaluation of C3-DTLZ4 problem.
    
    Args:
        x: Decision variables [x0, x1, ..., x_{n-1}], all in [0, 1]
        n_objectives: Number of objectives
        alpha: Density parameter (default: 100.0)
    
    Returns:
        Dictionary with:
            - 'objectives': List of objective values
            - 'constraints': List of constraint values
            - 'feasible': Boolean indicating if solution is feasible
    """
    objectives = dtlz4_objectives(x, n_objectives, alpha)
    constraints = c3dtlz4_constraints(objectives)
    feasible = all(c <= 0 for c in constraints)
    
    return {
        'objectives': objectives,
        'constraints': constraints,
        'feasible': feasible
    }

def main(csv: str = "results.csv"):
    df = read_csv(csv,sep=',')
    print("N,X0,X1,X2,F1,F2,C1,C2")
    for i in range(0,df.shape[0]):
        x=(df['X0'][i], df['X1'][i], df['X2'][i])
        result = evaluate_c3dtlz4(x)
        print(f"{df['N'][i]:3d},{df['X0'][i]},{df['X1'][i]},{df['X2'][i]},{result['objectives'][0]},{result['objectives'][1]},{result['constraints'][0]},{result['constraints'][1]}")
    return 0

if __name__ == "__main__":
    results = "results.csv" if len(argv) < 2 else argv[1]
    exit(main(results))
