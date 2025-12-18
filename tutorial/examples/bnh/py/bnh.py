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

import numpy as np
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from inspect import getsource

def main(rootpath: str = ".") -> int:
    # 1. Define the Binh-Korn problem (BNH)
    problem = get_problem("bnh")
    
    # 2. Problem definition
    print("## Problem definition")
    print("```")
    print('"Boundary definition"')
    print(getsource(problem.__init__))
    print('"Functions (F) and inequality constraints (G) definitions"')
    print(getsource(problem._evaluate))
    print('"Expected Pareto front"')
    print(getsource(problem._calc_pareto_front))
    print("```")
    
    # 3. Configure the optimization algorithm (NSGA-II is a standard choice)
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # 4. Define termination criteria (e.g., number of generations)
    termination = get_termination("n_gen", 200)
    
    # 5. Run the optimization
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=False)
    
    # Get the non-dominated solutions (approximated Pareto front)
    F = res.F  # Objective values
    X = res.X  # Decision variables
    
    # Get the true Pareto front in objective space
    true_pareto = problem.pareto_front()
    
    # Calculate the true Pareto front in decision variable space
    # Use a much smaller sample to avoid memory issues
    n_samples = 100  # Reduced from 1000
    x1_range = np.linspace(0, 5, n_samples)
    x2_range = np.linspace(0, 3, n_samples)
    
    # Sample only feasible points near constraints instead of full grid
    X_samples = []
    for x1 in x1_range:
        for x2 in x2_range:
            # Check constraints
            g1 = (x1-5)**2 + x2**2 - 25
            g2 = 7.7 - (x1-8)**2 - (x2+3)**2
            if g1 <= 0 and g2 <= 0:  # Feasible
                X_samples.append([x1, x2])
    
    X_feasible = np.array(X_samples)
    
    # Evaluate objectives for feasible points
    F_feasible = np.zeros((len(X_feasible), 2))
    for i, x in enumerate(X_feasible):
        F_feasible[i, 0] = 4*x[0]**2 + 4*x[1]**2
        F_feasible[i, 1] = (x[0]-5)**2 + (x[1]-5)**2
    
    # Find non-dominated solutions
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    pareto_indices = nds.do(F_feasible, only_non_dominated_front=True)
    X_true_pareto = X_feasible[pareto_indices]
    
    # Sort by x1 for plotting
    sort_idx = np.argsort(X_true_pareto[:, 0])
    X_true_pareto = X_true_pareto[sort_idx]
    
    # 6. Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Plot 1: Objective Space (Pareto Front) =====
    ax1.scatter(F[:, 0], F[:, 1], label="NSGA-II Approximation", alpha=0.7, s=50, c='blue')
    ax1.plot(true_pareto[:, 0], true_pareto[:, 1], 'r-', label="True Pareto Front", linewidth=2, alpha=0.7)
    ax1.set_title("Pareto Front in Objective Space", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Objective 1: $f_1(x) = 4x_1^2 + 4x_2^2$", fontsize=11)
    ax1.set_ylabel("Objective 2: $f_2(x) = (x_1-5)^2 + (x_2-5)^2$", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Decision Variable Space =====
    # Add constraint visualization first (so shading is behind points)
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Constraint 1: (x₁ - 5)² + x₂² ≤ 25 (circle centered at (5,0) with radius 5)
    c1_x = 5 + 5*np.cos(theta)
    c1_y = 5*np.sin(theta)
    ax2.fill(c1_x, c1_y, color='green', alpha=0.15, label='Feasible for $C_1$ (inside)')
    ax2.plot(c1_x, c1_y, 'g--', linewidth=2, alpha=0.8, label='$C_1$ boundary')
    
    # Constraint 2: (x₁ - 8)² + (x₂ + 3)² ≥ 7.7 (circle centered at (8,-3) with radius √7.7)
    c2_x = 8 + np.sqrt(7.7)*np.cos(theta)
    c2_y = -3 + np.sqrt(7.7)*np.sin(theta)
    ax2.fill(c2_x, c2_y, color='red', alpha=0.15, label='Infeasible for $C_2$ (inside)')
    ax2.plot(c2_x, c2_y, 'm--', linewidth=2, alpha=0.8, label='$C_2$ boundary')
    
    # Plot the actual decision variables (x₁, x₂) from NSGA-II
    ax2.scatter(X[:, 0], X[:, 1], label="NSGA-II Solutions", alpha=0.7, s=60, c='blue', 
                edgecolors='darkblue', linewidth=1, zorder=5)
    
    
    with open(rootpath + "/NSGA2_pareto.txt","w") as nsga2_pareto:
            nsga2_pareto.write("X1 X2 F1 F2\n") 
            [nsga2_pareto.write(f"{X[i][0]} {X[i][1]} {F[i][0]} {F[i][1]}\n") for i in range(0,len(X))]
    
    # Plot the true Pareto front in decision variable space
    ax2.plot(X_true_pareto[:, 0], X_true_pareto[:, 1], 'r-', linewidth=2.5, 
             label='True Pareto Front', alpha=0.8, zorder=6)
    
    # Add variable bounds
    ax2.axhline(y=0, color='orange', linewidth=2, linestyle=':', alpha=0.6)
    ax2.axhline(y=3, color='orange', linewidth=2, linestyle=':', alpha=0.6)
    ax2.axvline(x=0, color='orange', linewidth=2, linestyle=':', alpha=0.6)
    ax2.axvline(x=5, color='orange', linewidth=2, linestyle=':', alpha=0.6)
    ax2.text(5.2, 3.2, 'bounds', color='orange', fontsize=9)
    
    ax2.set_title("Pareto Solutions in Decision Variable Space", fontsize=14, fontweight='bold')
    ax2.set_xlabel("$x_1$", fontsize=12)
    ax2.set_ylabel("$x_2$", fontsize=12)
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-7, 6)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timer = fig.canvas.new_timer(interval=5000, callbacks=[(plt.close, [], {})])
    timer.start()
    plt.show()
    return 0
if __name__ == "__main__":
    print(main())
    
