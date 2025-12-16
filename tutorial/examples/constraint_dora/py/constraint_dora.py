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
    Minimum value: f = (√5 - 1)² ≈ 3.236068
    
Reference: Wolfram Alpha
https://www.wolframalpha.com/input?i=Minimize%3A+f%28x1%2C+x2%29+%3D+%28x1+-+2%29%5E2+%2B+%28x2+-+1%29%5E2+subject+to+x1%5E2+%2B+x2%5E2+-+1+%3C%3D+0
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from hashlib import sha256

# Define the objective function
def objective(x):
    """Objective function: (x1 - 2)^2 + (x2 - 1)^2"""
    return (x[0] - 2)**2 + (x[1] - 1)**2

# Define the constraint
def constraint(x):
    """Constraint: x1^2 + x2^2 <= 1 (must return >= 0 for feasibility)"""
    return 1 - x[0]**2 - x[1]**2

def main():
# Initial guess
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
    
    # Create grid for contour plot
    x1 = np.linspace(-1.5, 2.5, 400)
    x2 = np.linspace(-1.5, 2.0, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1 - 2)**2 + (X2 - 1)**2
    
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
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1.5, 2.0)
    
    plt.tight_layout()
    timer = fig.canvas.new_timer(interval=5000, callbacks=[(plt.close, [], {})])
    timer.start()
    plt.show()
    return sha256(result_slsqp_pprinted.encode()).hexdigest()

if __name__ == "__main__":
    print(main())
