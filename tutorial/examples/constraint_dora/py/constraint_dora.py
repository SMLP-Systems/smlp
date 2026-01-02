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

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from hashlib import sha256
from os import popen
from pandas import read_csv, concat

# Define the objective function
def objective(x):
    """Objective function: (x1 - 2)^2 + (x2 - 1)^2"""
    return (x[0] - 2)**2 + (x[1] - 1)**2

# Define the constraint
def constraint(x):
    """Constraint: x1^2 + x2^2 <= 1 (must return >= 0 for feasibility)"""
    return 1 - x[0]**2 - x[1]**2

def brute_force(f_data, f_constraint):
    # Read both files
    df1 = read_csv(f_data, sep=r'\s+')
    df2 = read_csv(f_constraint, sep=r'\s+')
    # Paste (merge side by side) - add suffixes to avoid duplicate column names
    df1.columns = [f"{col}_1" for col in df1.columns]
    df2.columns = [f"{col}_2" for col in df2.columns]
    df = concat([df1, df2], axis=1)

    # Sort by column 6 (index 5)
    df = df.sort_values(by=df.columns[5])

    # Filter out rows where first column starts with 'X'
    df = df[~df.iloc[:, 0].astype(str).str.startswith('X')]

    # Filter rows where column 6 > 0
    df = df[df.iloc[:, 5] > 0]

    # Sort by column 3 (index 2)
    df = df.sort_values(by=df.columns[2])

    # Get the first row
    result = df.head(1)

    # Print the result
    return result.to_string(index=False, header=False)

def main(n: int = 400, rootpath: str = ".") -> int:
# Initial guess
    # Create grid for contour plot and calculate result using brute force method
    rng = range(0, n)
    x1_start, x1_stop = (-1.5, 2.5)
    x2_start, x2_stop = (-1.5, 2.0)
    x1 = np.linspace(x1_start, x1_stop, rng.stop)
    x2 = np.linspace(x2_start, x1_stop, rng.stop)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1 - 2)**2 + (X2 - 1)**2
    C = 1 - X1**2 - X2**2
    dataset=rootpath + "/dataset.txt"
    with open(dataset,"w") as ds:
        ds.write("X1 X2 Y1\n")
        [[ds.write(f"{X1[i][j]} {X2[i][j]} {Z[i][j]}\n") for j in rng] for i in rng]
    constraint_set=rootpath + "/constraint.txt"
    with open(constraint_set,"w") as cs:
        cs.write("X1 X2 Y1\n")
        [[cs.write(f"{X1[i][j]} {X2[i][j]} {C[i][j]}\n") for j in rng] for i in rng]
    #Brute force solution
    brute_force_result=brute_force(dataset, constraint_set).split()
    print("=" * 60)
    print(f"Brute force result: x1 = {float(brute_force_result[0]):.5f} x2 = {float(brute_force_result[1]):.5f} f(x*) = {float(brute_force_result[2]):.5f}") 
    dataset_polar=rootpath + "/dataset_polar.txt"
    R = np.sqrt(X1*X1 + X2*X2)
    THETA = np.arctan2(X2,X1)
    X11 = R * np.cos(THETA)
    X21 = R * np.sin(THETA)
    Z1 = (X11 - 2)**2 + (X21 - 1)**2
    C1 = 1 - R**2
    with open(dataset_polar,"w") as ds:
        ds.write("X1 X2 Y1\n")
        [[ds.write(f"{R[i][j]} {THETA[i][j]} {Z1[i][j]}\n") for j in rng] for i in rng]
    constraint_set_polar=rootpath + "/constraint_polar.txt"
    with open(constraint_set_polar,"w") as cs:
        cs.write("X1 X2 Y1\n")
        [[cs.write(f"{R[i][j]} {THETA[i][j]} {C1[i][j]}\n") for j in rng] for i in rng]
    #Brute force solution in polar coordinates
    brute_force_result_polar=brute_force(dataset_polar, constraint_set_polar).split()
    print("=" * 60)
    print(f"Brute force result polar: r = {float(brute_force_result_polar[0]):.5f} theta = {float(brute_force_result_polar[1]):.5f} f(x*) = {float(brute_force_result_polar[2]):.5f}") 
    brute_force_result_polar_x = float(brute_force_result_polar[0])*np.cos(float(brute_force_result_polar[1]))
    brute_force_result_polar_y = float(brute_force_result_polar[0])*np.sin(float(brute_force_result_polar[1]))
    print(f"Brute force result polar Cartesian coordinate system: x1 = {float(brute_force_result_polar_x):.5f} theta = {float(brute_force_result_polar_y):.5f} f(x*) = {float(brute_force_result_polar[2]):.5f}") 
    print("=" * 60)

    x0 = np.array([0.5, 0.5])
    
    # Define constraint dictionary for scipy
    con = {'type': 'ineq', 'fun': constraint}
    
    # Solve the optimization problem
    result = minimize(objective, x0, method='SLSQP', constraints=con)
    
    # Print results
    print("=" * 60)
    print("CONSTRAINED OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"\nConstraint value: {constraint(result.x):.6f} (should be >= 0)")
    print(f"Constraint satisfied: {constraint(result.x) >= -1e-6}")
    result_slsqp_pprinted= f"\nOptimal solution: x1 = {result.x[0]:.6f} x2 = {result.x[1]:.6f} f(x*) = {result.fun:.6f}"
    print(result_slsqp_pprinted)
    print("\nExpected result (analytical solution):")
    print(f"  2/√5 = {2/np.sqrt(5):.6f}, 1/√5 = {1/np.sqrt(5):.6f}")
    print(f"  (√5 - 1)² = {(np.sqrt(5) - 1)**2:.6f}")
    print("=" * 60)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contours of objective function
    contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)
    
    # Plot constraint boundary (unit circle)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', linewidth=2, label='Constraint boundary')
    ax.fill(circle_x, circle_y, alpha=0.1, color='red', label='Feasible region')
    
    # Plot unconstrained optimum
    ax.plot(2, 1, 'bs', markersize=12, label='Unconstrained optimum (2, 1)')
    
    # Plot constrained optimum
    ax.plot(result.x[0], result.x[1], 'go', markersize=12, 
            label=f'Constrained optimum ({result.x[0]:.3f}, {result.x[1]:.3f})')
    
    # Plot initial guess
    ax.plot(x0[0], x0[1], 'mx', markersize=12, label=f'Initial guess ({x0[0]}, {x0[1]})')
    
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Constrained Optimization: Minimize f(x1,x2) subject to x1² + x2² ≤ 1', 
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_ylim(-1.5, 2.0)
    
    plt.tight_layout()
    timer = fig.canvas.new_timer(interval=5000, callbacks=[(plt.close, [], {})])
    timer.start()
    plt.show()
    return sha256(result_slsqp_pprinted.encode()).hexdigest()

if __name__ == "__main__":
    print(main())
