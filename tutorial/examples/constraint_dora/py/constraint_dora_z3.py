#!/usr/bin/python3.12
from z3 import *
import numpy as np
from scipy.optimize import minimize

print("=" * 60)
print("METHOD 1: Z3 Optimizer (for comparison)")
print("=" * 60)

# Create Z3 optimizer
opt = Optimize()

# Define real variables
x1 = Real('x1')
x2 = Real('x2')

# Define the objective function: (x1-2)^2 + (x2-1)^2
objective = (x1 - 2)**2 + (x2 - 1)**2

# Add constraint: x1^2 + x2^2 <= 1 (inside unit circle)
opt.add(x1**2 + x2**2 <= 1)

# Minimize the objective
opt.set("timeout",30000)
opt.minimize(objective)

# Solve
if opt.check() == sat:
    model = opt.model()
    x1_val = model[x1]
    x2_val = model[x2]
    
    # Convert to float for display
    x1_float = float(x1_val.as_decimal(10).rstrip('?'))
    x2_float = float(x2_val.as_decimal(10).rstrip('?'))
    obj_val = (x1_float - 2)**2 + (x2_float - 1)**2
    
    print(f"Z3 Solution:")
    print(f"  x1 = {x1_float:.6f}")
    print(f"  x2 = {x2_float:.6f}")
    print(f"  f(x1, x2) = {obj_val:.6f}")
    print(f"  Constraint check: x1^2 + x2^2 = {x1_float**2 + x2_float**2:.6f}")
else:
    print("No solution found by Z3")

print("\n" + "=" * 60)
print("METHOD 2: Scipy (Recommended for this problem)")
print("=" * 60)

# Define objective function
def objective_func(x):
    return (x[0] - 2)**2 + (x[1] - 1)**2

# Define constraint: x1^2 + x2^2 - 1 <= 0
def constraint_func(x):
    return 1 - (x[0]**2 + x[1]**2)  # Must be >= 0

constraint = {'type': 'ineq', 'fun': constraint_func}

# Initial guess
x0 = [0.5, 0.5]

# Solve using SLSQP (Sequential Least Squares Programming)
result = minimize(objective_func, x0, method='SLSQP', constraints=constraint)

print(f"Scipy Solution:")
print(f"  x1 = {result.x[0]:.6f}")
print(f"  x2 = {result.x[1]:.6f}")
print(f"  f(x1, x2) = {result.fun:.6f}")
print(f"  Constraint check: x1^2 + x2^2 = {result.x[0]**2 + result.x[1]**2:.6f}")
print(f"  Success: {result.success}")

print("\n" + "=" * 60)
print("ANALYTICAL SOLUTION")
print("=" * 60)

# The optimal point lies on the boundary of the circle
# in the direction from origin to (2,1)
target = np.array([2, 1])
direction = target / np.linalg.norm(target)
optimal = direction  # Scale to unit circle

print(f"Analytical Solution:")
print(f"  x1 = {optimal[0]:.6f}")
print(f"  x2 = {optimal[1]:.6f}")
print(f"  f(x1, x2) = {objective_func(optimal):.6f}")
print(f"  Constraint check: x1^2 + x2^2 = {optimal[0]**2 + optimal[1]**2:.6f}")

print("\n" + "=" * 60)
print("EXPLANATION")
print("=" * 60)
print("The minimum occurs at the point on the unit circle")
print("closest to (2, 1). This is found by moving from the")
print("origin toward (2, 1) until hitting the circle boundary.")
print("=" * 60)
