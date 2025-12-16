#!/usr/bin/python3.12
"""
Binh and Korn (BNH) Multi-Objective Optimization Problem
Reference: https://pymoo.org/problems/multi/bnh.html
Problem Definition:
-------------------
Minimize:
    f₁(x) = 4x₁² + 4x₂²
    f₂(x) = (x₁ - 5)² + (x₂ - 5)²

Subject to:
    C₁(x) = (x₁ - 5)² + x₂² ≤ 25
    C₂(x) = (x₁ - 8)² + (x₂ + 3)² ≥ 7.7
    0 ≤ x₁ ≤ 5
    0 ≤ x₂ ≤ 3

Characteristics:
----------------
- 2 objectives (minimize both)
- 2 constraints
- 2 decision variables
- The Pareto-optimal solutions are marked by bold continuous curves
- Constraints do not make any solution in the unconstrained Pareto-optimal 
  front infeasible, but introduce additional difficulty in solving
"""

import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# 1. Define the Binh-Korn problem (BNH)
problem = get_problem("bnh")

# 2. Configure the optimization algorithm (NSGA-II is a standard choice)
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# 3. Define termination criteria (e.g., number of generations)
termination = get_termination("n_gen", 200)

# 4. Run the optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=False)

# Get the non-dominated solutions (approximated Pareto front)
F = res.F

# 5. Plot the results
plt.figure(figsize=(10, 6))

# Plot the NSGA-II approximation
plt.scatter(F[:, 0], F[:, 1], label="NSGA-II Approximation", alpha=0.7, s=50)

# Optional: Add the true Pareto front if available for comparison
# pymoo provides the true front for the BNH problem
true_pareto = problem.pareto_front()
plt.plot(true_pareto[:, 0], true_pareto[:, 1], 'r-', label="True Pareto Front", linewidth=2, alpha=0.7)

plt.title("Binh and Korn Problem Pareto Front")
plt.xlabel("Objective 1 ($f_1$)")
plt.ylabel("Objective 2 ($f_2$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
