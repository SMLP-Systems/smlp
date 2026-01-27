#!/usr/bin/python3.12
from pysmt.shortcuts import Symbol, Real, Solver, And, Plus, Times, Minus, LE, GE
from pysmt.typing import REAL
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from math import inf
from hashlib import sha256
from base64 import b64encode
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def solve_bnh_pysmt(rootpath: str = ".", timeout: float=5000) -> str:
    """
    Solve the BNH multi-objective optimization problem using PySMT with Z3.
    Uses multiple approaches to find Pareto optimal solutions.
    """
    
    logging.info("=" * 60)
    logging.info("BNH Multi-Objective Optimization Problem with PySMT")
    logging.info("=" * 60)
    
    # Store Pareto solutions
    pareto_solutions = []
    
    # Approach 1: Minimize f1 with constraint satisfaction
    logging.info("\n1. Optimization: Minimize f1")
    logging.info("-" * 60)
    
    result = minimize_objective_1()
    if result:
        pareto_solutions.append(result)
        x1_val, x2_val, f1_val, f2_val = result
        logging.info(f"x1 = {x1_val:.6f}")
        logging.info(f"x2 = {x2_val:.6f}")
        logging.info(f"f1 = {f1_val:.6f}")
        logging.info(f"f2 = {f2_val:.6f}")
    
    # Approach 2: Minimize f2 with constraint satisfaction
    logging.info("\n2. Optimization: Minimize f2")
    logging.info("-" * 60)
    
    result = minimize_objective_2()
    if result:
        pareto_solutions.append(result)
        x1_val, x2_val, f1_val, f2_val = result
        logging.info(f"x1 = {x1_val:.6f}")
        logging.info(f"x2 = {x2_val:.6f}")
        logging.info(f"f1 = {f1_val:.6f}")
        logging.info(f"f2 = {f2_val:.6f}")
    
    # Approach 3: Weighted sum method with different weights
    logging.info("\n3. Weighted Sum Method (various weights):")
    logging.info("-" * 60)
    
    weights = [(0.2, 0.8), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.8, 0.2)]
    
    for w1, w2 in weights:
        result = minimize_weighted(w1, w2)
        if result:
            x1_val, x2_val, f1_val, f2_val = result
            logging.info(f"\nWeights (w1={w1}, w2={w2}):")
            logging.info(f"  x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
            logging.info(f"  f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
            pareto_solutions.append(result)
    
    # Verify constraints for all solutions
    logging.info("\n4. Constraint Verification:")
    logging.info("-" * 60)
    for i, (x1_val, x2_val, f1_val, f2_val) in enumerate(pareto_solutions):
        c1 = (x1_val - 5)**2 + x2_val**2
        c2 = (x1_val - 8)**2 + (x2_val + 3)**2
        logging.info(f"Solution {i+1}:")
        logging.info(f"  C1 = {c1:.6f} (should be ≤ 25): {'✓' if c1 <= 25 else '✗'}")
        logging.info(f"  C2 = {c2:.6f} (should be ≥ 7.7): {'✓' if c2 >= 7.7 else '✗'}")
    
    # Plot Pareto front
    plot_results(pareto_solutions, timeout)

    solution_string = ' '.join(str(item) for item in pareto_solutions)
    logging.info(solution_string)
    return sha256(solution_string.encode()).hexdigest()


def get_constraints(x1, x2):
    """Define the constraints for the BNH problem using PySMT."""
    constraints = []
    
    # Variable bounds
    constraints.append(GE(x1, Real(0)))
    constraints.append(LE(x1, Real(5)))
    constraints.append(GE(x2, Real(0)))
    constraints.append(LE(x2, Real(3)))
    
    # C1: (x1 - 5)^2 + x2^2 <= 25
    x1_minus_5 = Plus(x1, Real(-5))
    term1 = Times(x1_minus_5, x1_minus_5)
    term2 = Times(x2, x2)
    c1_lhs = Plus(term1, term2)
    constraints.append(LE(c1_lhs, Real(25)))
    
    # C2: (x1 - 8)^2 + (x2 + 3)^2 >= 7.7
    x1_minus_8 = Plus(x1, Real(-8))
    x2_plus_3 = Plus(x2, Real(3))
    term3 = Times(x1_minus_8, x1_minus_8)
    term4 = Times(x2_plus_3, x2_plus_3)
    c2_lhs = Plus(term3, term4)
    constraints.append(GE(c2_lhs, Real(7.7)))
    
    return And(constraints)


def evaluate_objectives(x1_val, x2_val):
    """Evaluate both objective functions."""
    f1 = 4 * x1_val**2 + 4 * x2_val**2
    f2 = (x1_val - 5)**2 + (x2_val - 5)**2
    return f1, f2


def minimize_objective_1():
    """Minimize f1 = 4*x1^2 + 4*x2^2 using binary search."""
    x1 = Symbol("x1", REAL)
    x2 = Symbol("x2", REAL)
    
    constraints = get_constraints(x1, x2)
    
    # Binary search on f1
    min_bound = 0.0
    max_bound = 400.0  # Upper bound for f1
    epsilon = 0.01
    best_solution = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # f1 = 4*x1^2 + 4*x2^2 <= mid
        x1_squared = Times(x1, x1)
        x2_squared = Times(x2, x2)
        f1_expr = Plus(Times(Real(4), x1_squared), Times(Real(4), x2_squared))
        obj_constraint = LE(f1_expr, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                
                best_solution = (x1_val, x2_val, f1_val, f2_val)
                max_bound = mid
            else:
                min_bound = mid
    
    return best_solution


def minimize_objective_2():
    """Minimize f2 = (x1-5)^2 + (x2-5)^2 using binary search."""
    x1 = Symbol("x1_f2", REAL)
    x2 = Symbol("x2_f2", REAL)
    
    constraints = get_constraints(x1, x2)
    
    # Binary search on f2
    min_bound = 0.0
    max_bound = 100.0  # Upper bound for f2
    epsilon = 0.01
    best_solution = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # f2 = (x1-5)^2 + (x2-5)^2 <= mid
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        obj_constraint = LE(f2_expr, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                
                best_solution = (x1_val, x2_val, f1_val, f2_val)
                max_bound = mid
            else:
                min_bound = mid
    
    return best_solution


def minimize_weighted(w1, w2):
    """Minimize weighted sum: w1*f1 + w2*f2 using binary search."""
    x1 = Symbol(f"x1_w{w1}", REAL)
    x2 = Symbol(f"x2_w{w1}", REAL)
    
    constraints = get_constraints(x1, x2)
    
    # Binary search on weighted objective
    min_bound = 0.0
    max_bound = 500.0  # Upper bound for weighted sum
    epsilon = 0.01
    best_solution = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
        # f1 = 4*x1^2 + 4*x2^2
        x1_squared = Times(x1, x1)
        x2_squared = Times(x2, x2)
        f1_expr = Plus(Times(Real(4), x1_squared), Times(Real(4), x2_squared))
        
        # f2 = (x1-5)^2 + (x2-5)^2
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        
        # Weighted objective: w1*f1 + w2*f2 <= mid
        weighted = Plus(Times(Real(w1), f1_expr), Times(Real(w2), f2_expr))
        obj_constraint = LE(weighted, Real(mid))
        
        full_constraint = And(constraints, obj_constraint)
        
        with Solver(name="z3") as solver:
            solver.add_assertion(full_constraint)
            
            if solver.solve():
                model = solver.get_model()
                x1_val = float(model.get_value(x1).constant_value())
                x2_val = float(model.get_value(x2).constant_value())
                f1_val, f2_val = evaluate_objectives(x1_val, x2_val)
                
                best_solution = (x1_val, x2_val, f1_val, f2_val)
                max_bound = mid
            else:
                min_bound = mid
    
    return best_solution


def plot_results(solutions, timeout):
    """Plot the Pareto front in objective space."""
    if not solutions:
        return
    
    f1_vals = [s[2] for s in solutions]
    f2_vals = [s[3] for s in solutions]
    
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(f1_vals, f2_vals, c='red', s=100, marker='o', 
                edgecolors='black', linewidths=2, zorder=3)
    
    for i, (x1, x2, f1, f2) in enumerate(solutions):
        plt.annotate(f'{i+1}', (f1, f2), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    plt.xlabel('f₁(x) = 4x₁² + 4x₂²', fontsize=12)
    plt.ylabel('f₂(x) = (x₁-5)² + (x₂-5)²', fontsize=12)
    plt.title('BNH Problem: Pareto Front Approximation (PySMT/Z3)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    
    plt.show()
    
    logging.info("\n5. Pareto Front Plot Generated")
    logging.info("-" * 60)
    logging.info("\n" + "=" * 60)
    logging.info(f"Found {len(solutions)} Pareto optimal solutions")
    logging.info("=" * 60)


if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else float(argv[2])
    print(solve_bnh_pysmt(rootpath, timeout))
