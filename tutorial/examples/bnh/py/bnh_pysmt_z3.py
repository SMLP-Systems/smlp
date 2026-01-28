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
    
    # Store Pareto solutions with method labels
    pareto_solutions = []
    
    # Approach 1: Minimize f1 with constraint satisfaction
    logging.info("\n1. Optimization: Minimize f1")
    logging.info("-" * 60)
    
    result = minimize_objective_1()
    if result:
        pareto_solutions.append((*result, 'minimize_f1'))
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
        pareto_solutions.append((*result, 'minimize_f2'))
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
            pareto_solutions.append((*result, 'weighted'))
    
    # Approach 4: Epsilon Constraint Method
    logging.info("\n4. Epsilon Constraint Method (Pareto Front Simulation):")
    logging.info("-" * 60)
    
    epsilon_solutions = epsilon_constraint_method(num_points=10)
    for result in epsilon_solutions:
        if result:
            x1_val, x2_val, f1_val, f2_val = result
            logging.info(f"  x1 = {x1_val:.6f}, x2 = {x2_val:.6f}")
            logging.info(f"  f1 = {f1_val:.6f}, f2 = {f2_val:.6f}")
            pareto_solutions.append((*result, 'epsilon'))
    
    # Verify constraints for all solutions
    logging.info("\n5. Constraint Verification:")
    logging.info("-" * 60)
    for i, solution in enumerate(pareto_solutions):
        x1_val, x2_val, f1_val, f2_val, method = solution
        c1 = (x1_val - 5)**2 + x2_val**2
        c2 = (x1_val - 8)**2 + (x2_val + 3)**2
        logging.info(f"Solution {i+1} ({method}):")
        logging.info(f"  C1 = {c1:.6f} (should be ≤ 25): {'✓' if c1 <= 25 else '✗'}")
        logging.info(f"  C2 = {c2:.6f} (should be ≥ 7.7): {'✓' if c2 >= 7.7 else '✗'}")
    
    # Plot Pareto front
    plot_results(pareto_solutions, timeout)

    solution_string = ' '.join(str(item[:4]) for item in pareto_solutions)
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


def epsilon_constraint_method(num_points=10):
    """
    Generate Pareto front using epsilon constraint method.
    
    Strategy: Minimize f1 while constraining f2 to different epsilon values.
    This systematically explores the trade-off between objectives.
    
    Args:
        num_points: Number of points to generate along the Pareto front
        
    Returns:
        List of solutions (x1, x2, f1, f2)
    """
    solutions = []
    
    # First, find the range of f2 values
    # Get minimum f2 (optimizing f2 alone)
    x1 = Symbol("x1_eps_range", REAL)
    x2 = Symbol("x2_eps_range", REAL)
    constraints = get_constraints(x1, x2)
    
    # Find minimum f2
    min_bound = 0.0
    max_bound = 100.0
    epsilon = 0.01
    min_f2 = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
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
                _, f2_val = evaluate_objectives(x1_val, x2_val)
                min_f2 = f2_val
                max_bound = mid
            else:
                min_bound = mid
    
    # Find maximum f2 (when minimizing f1)
    x1 = Symbol("x1_eps_max", REAL)
    x2 = Symbol("x2_eps_max", REAL)
    constraints = get_constraints(x1, x2)
    
    min_bound = 0.0
    max_bound = 400.0
    epsilon = 0.01
    max_f2 = None
    
    iteration = 0
    max_iterations = 50
    
    while max_bound - min_bound > epsilon and iteration < max_iterations:
        iteration += 1
        mid = (min_bound + max_bound) / 2
        
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
                _, f2_val = evaluate_objectives(x1_val, x2_val)
                max_f2 = f2_val
                max_bound = mid
            else:
                min_bound = mid
    
    if min_f2 is None or max_f2 is None:
        logging.info("  Could not determine f2 range")
        return solutions
    
    logging.info(f"  f2 range: [{min_f2:.6f}, {max_f2:.6f}]")
    
    # Generate epsilon values
    epsilon_values = np.linspace(min_f2, max_f2, num_points)
    
    # For each epsilon value, minimize f1 subject to f2 <= epsilon
    for eps_val in epsilon_values:
        # Convert numpy float to Python float for PySMT
        eps_val = float(eps_val)
        x1 = Symbol(f"x1_eps_{eps_val}", REAL)
        x2 = Symbol(f"x2_eps_{eps_val}", REAL)
        
        constraints = get_constraints(x1, x2)
        
        # Add constraint: f2 <= epsilon
        x1_minus_5 = Plus(x1, Real(-5))
        x2_minus_5 = Plus(x2, Real(-5))
        term1 = Times(x1_minus_5, x1_minus_5)
        term2 = Times(x2_minus_5, x2_minus_5)
        f2_expr = Plus(term1, term2)
        f2_constraint = LE(f2_expr, Real(eps_val))
        
        constraints_with_epsilon = And(constraints, f2_constraint)
        
        # Binary search to minimize f1
        min_bound = 0.0
        max_bound = 400.0
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
            f1_constraint = LE(f1_expr, Real(mid))
            
            full_constraint = And(constraints_with_epsilon, f1_constraint)
            
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
        
        if best_solution:
            solutions.append(best_solution)
    
    return solutions


def plot_results(solutions, timeout):
    """Plot the Pareto front in objective space with different colors per method."""
    if not solutions:
        return
    
    # Separate solutions by method
    methods = {}
    for solution in solutions:
        x1, x2, f1, f2, method = solution
        if method not in methods:
            methods[method] = {'f1': [], 'f2': [], 'solutions': []}
        methods[method]['f1'].append(f1)
        methods[method]['f2'].append(f2)
        methods[method]['solutions'].append((x1, x2, f1, f2))
    
    # Define colors and labels for each method
    method_styles = {
        'minimize_f1': {'color': 'blue', 'marker': 's', 'label': 'Minimize f₁', 'size': 120},
        'minimize_f2': {'color': 'green', 'marker': 's', 'label': 'Minimize f₂', 'size': 120},
        'weighted': {'color': 'red', 'marker': 'o', 'label': 'Weighted Sum', 'size': 100},
        'epsilon': {'color': 'purple', 'marker': '^', 'label': 'Epsilon Constraint', 'size': 100}
    }
    
    fig = plt.figure(figsize=(12, 7))
    
    # Plot each method with its own color and marker
    for method, data in methods.items():
        style = method_styles.get(method, {'color': 'gray', 'marker': 'o', 'label': method, 'size': 100})
        plt.scatter(data['f1'], data['f2'], 
                   c=style['color'], 
                   s=style['size'], 
                   marker=style['marker'],
                   edgecolors='black', 
                   linewidths=1.5, 
                   label=style['label'],
                   alpha=0.8,
                   zorder=3)
    
    # Add annotations
    for i, solution in enumerate(solutions):
        x1, x2, f1, f2, method = solution
        plt.annotate(f'{i+1}', (f1, f2), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.xlabel('f₁(x) = 4x₁² + 4x₂²', fontsize=12)
    plt.ylabel('f₂(x) = (x₁-5)² + (x₂-5)²', fontsize=12)
    plt.title('BNH Problem: Pareto Front Approximation (PySMT/Z3)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    if not inf == timeout:
        timer = fig.canvas.new_timer(interval=timeout, callbacks=[(plt.close, [], {})])
        timer.start()
    
    plt.show()
    
    logging.info("\n6. Pareto Front Plot Generated")
    logging.info("-" * 60)
    logging.info("\n" + "=" * 60)
    logging.info(f"Found {len(solutions)} Pareto optimal solutions")
    for method, data in methods.items():
        logging.info(f"  - {method_styles.get(method, {}).get('label', method)}: {len(data['solutions'])} points")
    logging.info("=" * 60)


if __name__ == "__main__":
    rootpath = "." if len(argv) < 2 else argv[1]
    timeout = inf if len(argv) < 3 else float(argv[2])
    print(solve_bnh_pysmt(rootpath, timeout))
